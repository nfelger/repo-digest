from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
import pytest
from repo_digest.github import GitHubClient, PR, Issue, Commit, RepoActivity


SINCE = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)


def make_client(token="test-token"):
    return GitHubClient(token)


def test_fetch_merged_prs_returns_prs_after_since(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", side_effect=[
        # pulls list
        [
            {
                "number": 42,
                "title": "Add dark mode",
                "body": "Implements dark mode toggle",
                "user": {"login": "alice"},
                "labels": [{"name": "enhancement"}],
                "merged_at": "2026-04-20T10:00:00Z",
            },
            {
                "number": 41,
                "title": "Old PR",
                "body": "",
                "user": {"login": "bob"},
                "labels": [],
                "merged_at": "2026-04-10T10:00:00Z",  # before since
            },
        ],
        # files for PR 42
        [{"filename": "src/theme.py"}, {"filename": "tests/test_theme.py"}],
        # reviews for PR 42
        [{"body": "LGTM"}, {"body": ""}],
    ])

    prs = client.fetch_merged_prs("org", "repo", SINCE)

    assert len(prs) == 1
    assert prs[0].number == 42
    assert prs[0].title == "Add dark mode"
    assert prs[0].author == "alice"
    assert prs[0].labels == ["enhancement"]
    assert prs[0].files_changed == ["src/theme.py", "tests/test_theme.py"]
    assert prs[0].review_bodies == ["LGTM"]


def test_fetch_merged_prs_empty_when_none_in_window(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", return_value=[])
    prs = client.fetch_merged_prs("org", "repo", SINCE)
    assert prs == []
