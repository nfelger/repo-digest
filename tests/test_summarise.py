from unittest.mock import MagicMock, patch
from repo_digest.github import PR, Issue, Commit, RepoActivity
from repo_digest.summarise import format_activity, summarise_repo


def make_activity():
    return RepoActivity(
        repo="org/myrepo",
        merged_prs=[
            PR(
                number=1,
                title="Add login page",
                body="Implements JWT auth",
                author="alice",
                labels=["feature"],
                files_changed=["src/login.py", "tests/test_login.py"],
                review_bodies=["Looks good"],
            )
        ],
        open_prs=[
            PR(
                number=2,
                title="WIP: dashboard",
                body="",
                author="bob",
                labels=[],
                files_changed=["src/dashboard.py"],
                review_bodies=[],
            )
        ],
        closed_issues=[
            Issue(number=10, title="Fix crash", body="Crashes on null input", author="carol", labels=["bug"])
        ],
        open_issues=[
            Issue(number=11, title="Add dark mode", body="User request", author="dave", labels=["enhancement"])
        ],
        commits=[
            Commit(sha="abc1234", message="chore: update deps", author="eve")
        ],
    )


def test_format_activity_includes_pr_title():
    activity = make_activity()
    text = format_activity(activity)
    assert "Add login page" in text
    assert "alice" in text


def test_format_activity_includes_open_pr():
    activity = make_activity()
    text = format_activity(activity)
    assert "WIP: dashboard" in text
    assert "bob" in text


def test_format_activity_includes_issues():
    activity = make_activity()
    text = format_activity(activity)
    assert "Fix crash" in text
    assert "Add dark mode" in text


def test_format_activity_includes_commits():
    activity = make_activity()
    text = format_activity(activity)
    assert "update deps" in text
    assert "eve" in text


def test_format_activity_empty_repo():
    activity = RepoActivity(repo="org/empty")
    text = format_activity(activity)
    assert isinstance(text, str)


def test_summarise_repo_calls_litellm(mocker):
    activity = make_activity()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "## org/myrepo\n\nAlice shipped login."
    mocker.patch("repo_digest.summarise.litellm.completion", return_value=mock_response)

    result = summarise_repo(activity, model="ollama/gemma4", since_str="Apr 17")

    import repo_digest.summarise as s
    s.litellm.completion.assert_called_once()
    call_kwargs = s.litellm.completion.call_args
    assert call_kwargs.kwargs["model"] == "ollama/gemma4"
    assert isinstance(result, str)
    assert "Alice shipped login" in result
