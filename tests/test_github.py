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


def test_fetch_open_prs(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", side_effect=[
        [
            {
                "number": 50,
                "title": "WIP: new auth flow",
                "body": "Draft implementation",
                "user": {"login": "carol"},
                "labels": [{"name": "wip"}],
                "created_at": "2026-04-21T08:00:00Z",
            },
            {
                "number": 49,
                "title": "Old open PR",
                "body": "",
                "user": {"login": "dave"},
                "labels": [],
                "created_at": "2026-04-01T08:00:00Z",  # before since
            },
        ],
        # files for PR 50
        [{"filename": "src/auth.py"}],
    ])

    prs = client.fetch_open_prs("org", "repo", SINCE)

    assert len(prs) == 1
    assert prs[0].number == 50
    assert prs[0].author == "carol"
    assert prs[0].files_changed == ["src/auth.py"]
    assert prs[0].review_bodies == []


def test_fetch_closed_issues_excludes_prs(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", return_value=[
        {
            "number": 10,
            "title": "Bug: crash on login",
            "body": "Steps to reproduce...",
            "user": {"login": "eve"},
            "labels": [{"name": "bug"}],
            "closed_at": "2026-04-19T12:00:00Z",
        },
        {
            "number": 11,
            "title": "PR disguised as issue",
            "body": "",
            "user": {"login": "frank"},
            "labels": [],
            "closed_at": "2026-04-20T12:00:00Z",
            "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/11"},
        },
    ])

    issues = client.fetch_closed_issues("org", "repo", SINCE)

    assert len(issues) == 1
    assert issues[0].number == 10
    assert issues[0].labels == ["bug"]


def test_fetch_open_issues(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", return_value=[
        {
            "number": 20,
            "title": "Feature request: export to CSV",
            "body": "Would be great to export data",
            "user": {"login": "grace"},
            "labels": [{"name": "enhancement"}],
            "created_at": "2026-04-22T09:00:00Z",
        },
        {
            "number": 19,
            "title": "Old issue",
            "body": "",
            "user": {"login": "henry"},
            "labels": [],
            "created_at": "2026-04-10T09:00:00Z",  # before since
        },
    ])

    issues = client.fetch_open_issues("org", "repo", SINCE)

    assert len(issues) == 1
    assert issues[0].number == 20


def test_fetch_commits_excludes_merge_commits(mocker):
    client = make_client()
    mocker.patch.object(client, "_get_paginated", return_value=[
        {
            "sha": "abc1234",
            "commit": {
                "message": "fix: null pointer in login",
                "author": {"name": "ivan"},
            },
            "author": {"login": "ivan"},
            "parents": [{"sha": "prev1"}],
        },
        {
            "sha": "def5678",
            "commit": {
                "message": "Merge pull request #42 from feature/dark-mode",
                "author": {"name": "GitHub"},
            },
            "author": None,
            "parents": [{"sha": "prev1"}, {"sha": "prev2"}],  # merge commit
        },
    ])

    commits = client.fetch_commits("org", "repo", SINCE)

    assert len(commits) == 1
    assert commits[0].sha == "abc1234"
    assert commits[0].author == "ivan"
    assert "null pointer" in commits[0].message


def test_fetch_repo_activity(mocker):
    client = make_client()
    mocker.patch.object(client, "fetch_merged_prs", return_value=[])
    mocker.patch.object(client, "fetch_open_prs", return_value=[])
    mocker.patch.object(client, "fetch_closed_issues", return_value=[])
    mocker.patch.object(client, "fetch_open_issues", return_value=[])
    mocker.patch.object(client, "fetch_commits", return_value=[])

    activity = client.fetch_repo_activity("org/repo", SINCE)

    assert activity.repo == "org/repo"
    client.fetch_merged_prs.assert_called_once_with("org", "repo", SINCE)
    client.fetch_commits.assert_called_once_with("org", "repo", SINCE)
