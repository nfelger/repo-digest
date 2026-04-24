import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from repo_digest.cli import app
from repo_digest.github import RepoActivity
from repo_digest.summarise import Digest

runner = CliRunner()


def make_mock_digest():
    return Digest(
        overview="## Overview\n\nTwo teams shipped auth work.",
        repo_summaries={
            "org/repo1": "## org/repo1\n\nAlice shipped login.",
        },
    )


def test_missing_since_shows_error():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_run_outputs_overview(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("repos:\n  - org/repo1\n")
    mocker.patch("repo_digest.cli.fetch_all_repos", return_value=[RepoActivity(repo="org/repo1")])
    mocker.patch("repo_digest.cli.build_digest", return_value=make_mock_digest())
    mocker.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"})

    result = runner.invoke(app, ["--since", "7d", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "Overview" in result.output
    assert "Two teams shipped auth work" in result.output


def test_run_outputs_per_repo(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("repos:\n  - org/repo1\n")
    mocker.patch("repo_digest.cli.fetch_all_repos", return_value=[RepoActivity(repo="org/repo1")])
    mocker.patch("repo_digest.cli.build_digest", return_value=make_mock_digest())
    mocker.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"})

    result = runner.invoke(app, ["--since", "7d", "--config", str(config_file)])

    assert "Alice shipped login" in result.output


def test_missing_github_token_shows_error(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("repos:\n  - org/repo1\n")
    mocker.patch.dict(os.environ, {}, clear=True)
    # Ensure GITHUB_TOKEN not in env
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    mocker.patch.dict(os.environ, env, clear=True)

    result = runner.invoke(app, ["--since", "7d", "--config", str(config_file)])

    assert result.exit_code != 0
    assert "GITHUB_TOKEN" in result.output
