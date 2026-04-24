# Repo Digest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that fetches GitHub activity across multiple repos and produces an LLM-generated markdown digest in the terminal.

**Architecture:** Stateless CLI — user passes `--since` and the tool reads repos from a config file, fetches GitHub data in parallel, runs a two-stage LLM summarisation (per-repo then cross-repo synthesis), and renders the result as markdown via `rich`. No database, no persistent state.

**Tech Stack:** Python 3.11+, typer, httpx, litellm, rich, pyyaml, pytest, pytest-mock

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, `repo-digest` CLI entry point |
| `repo_digest/__init__.py` | Empty package marker |
| `repo_digest/config.py` | Load and validate `~/.repo-digest/config.yaml` → `Config` dataclass |
| `repo_digest/since.py` | Parse `--since` string (`7d`, `2w`, `1m`, ISO date) → `datetime` |
| `repo_digest/github.py` | GitHub REST API client — data classes, per-repo fetching, parallel multi-repo |
| `repo_digest/summarise.py` | Format activity as text, call LiteLLM per-repo, synthesise cross-repo |
| `repo_digest/cli.py` | typer app — parse args, orchestrate, render with rich |
| `tests/test_config.py` | Config loading tests |
| `tests/test_since.py` | Date parsing tests |
| `tests/test_github.py` | GitHub client tests (mocked HTTP) |
| `tests/test_summarise.py` | Summarisation tests (mocked litellm) |
| `tests/test_cli.py` | CLI integration tests |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `repo_digest/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "repo-digest"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer[all]",
    "httpx",
    "litellm",
    "rich",
    "pyyaml",
]

[project.scripts]
repo-digest = "repo_digest.cli:app"

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-mock",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package and test markers**

```bash
mkdir -p repo_digest tests
touch repo_digest/__init__.py tests/__init__.py
```

- [ ] **Step 3: Install in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors, `repo-digest` command available in PATH.

- [ ] **Step 4: Verify pytest runs cleanly**

```bash
pytest
```

Expected: `no tests ran` — 0 errors, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml repo_digest/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding"
```

---

## Task 2: Config Loading

**Files:**
- Create: `repo_digest/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
from pathlib import Path
import pytest
from repo_digest.config import Config, LLMConfig, load_config


def test_load_config_repos(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("repos:\n  - org/repo1\n  - org/repo2\n")
    config = load_config(config_file)
    assert config.repos == ["org/repo1", "org/repo2"]


def test_load_config_default_model(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("repos:\n  - org/repo1\n")
    config = load_config(config_file)
    assert config.llm.model == "ollama/gemma4"


def test_load_config_custom_model(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "repos:\n  - org/repo1\nllm:\n  model: anthropic/claude-sonnet-4-20250514\n"
    )
    config = load_config(config_file)
    assert config.llm.model == "anthropic/claude-sonnet-4-20250514"


def test_load_config_missing_repos(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  model: gpt-4o\n")
    with pytest.raises(KeyError):
        load_config(config_file)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'repo_digest.config'`

- [ ] **Step 3: Implement `repo_digest/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path.home() / ".repo-digest" / "config.yaml"


@dataclass
class LLMConfig:
    model: str = "ollama/gemma4"


@dataclass
class Config:
    repos: list[str]
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    llm_data = data.get("llm", {})
    return Config(
        repos=data["repos"],
        llm=LLMConfig(model=llm_data.get("model", "ollama/gemma4")),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/config.py tests/test_config.py
git commit -m "feat: config loading"
```

---

## Task 3: `--since` Date Parsing

**Files:**
- Create: `repo_digest/since.py`
- Create: `tests/test_since.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_since.py`:

```python
from datetime import datetime, timezone, date
import pytest
from repo_digest.since import parse_since


def test_parse_days():
    dt = parse_since("7d")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 6 <= delta.days <= 7


def test_parse_weeks():
    dt = parse_since("2w")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 13 <= delta.days <= 14


def test_parse_months():
    dt = parse_since("1m")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 29 <= delta.days <= 30


def test_parse_iso_date():
    dt = parse_since("2026-04-15")
    assert dt.date() == date(2026, 4, 15)
    assert dt.tzinfo is not None


def test_parse_invalid_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_since("yesterday")


def test_result_is_utc():
    dt = parse_since("3d")
    assert dt.tzinfo == timezone.utc
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_since.py -v
```

Expected: `ModuleNotFoundError: No module named 'repo_digest.since'`

- [ ] **Step 3: Implement `repo_digest/since.py`**

```python
import re
from datetime import datetime, timedelta, timezone


def parse_since(since: str) -> datetime:
    """Parse --since value to a UTC-aware datetime.

    Accepts: '7d', '2w', '1m' (relative) or '2026-04-15' (ISO date).
    """
    now = datetime.now(tz=timezone.utc)

    match = re.fullmatch(r"(\d+)([dwm])", since)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        if unit == "d":
            return now - timedelta(days=n)
        elif unit == "w":
            return now - timedelta(weeks=n)
        else:  # m
            return now - timedelta(days=n * 30)

    try:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise ValueError(
            f"Cannot parse --since value: {since!r}. "
            "Use '7d', '2w', '1m', or an ISO date like '2026-04-15'."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_since.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/since.py tests/test_since.py
git commit -m "feat: --since date parsing"
```

---

## Task 4: GitHub API Client — Data Types and Merged PR Fetching

**Files:**
- Create: `repo_digest/github.py`
- Create: `tests/test_github.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_github.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_github.py -v
```

Expected: `ModuleNotFoundError: No module named 'repo_digest.github'`

- [ ] **Step 3: Implement data classes and `fetch_merged_prs` in `repo_digest/github.py`**

```python
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import httpx


GITHUB_API = "https://api.github.com"


@dataclass
class PR:
    number: int
    title: str
    body: str
    author: str
    labels: list[str]
    files_changed: list[str]
    review_bodies: list[str]


@dataclass
class Issue:
    number: int
    title: str
    body: str
    author: str
    labels: list[str]


@dataclass
class Commit:
    sha: str
    message: str
    author: str


@dataclass
class RepoActivity:
    repo: str
    merged_prs: list[PR] = field(default_factory=list)
    open_prs: list[PR] = field(default_factory=list)
    closed_issues: list[Issue] = field(default_factory=list)
    open_issues: list[Issue] = field(default_factory=list)
    commits: list[Commit] = field(default_factory=list)


class GitHubClient:
    def __init__(self, token: str):
        self._client = httpx.Client(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _get_paginated(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages from a GitHub API path."""
        params = {**(params or {}), "per_page": 100}
        results = []
        url: str | None = path
        while url:
            if url.startswith("http"):
                resp = self._client.get(url)
            else:
                resp = self._client.get(url, params=params)
            resp.raise_for_status()
            results.extend(resp.json())
            params = None
            url = None
            for part in resp.headers.get("link", "").split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break
        return results

    def fetch_merged_prs(self, owner: str, repo: str, since: datetime) -> list[PR]:
        items = self._get_paginated(
            f"/repos/{owner}/{repo}/pulls",
            {"state": "closed", "sort": "updated", "direction": "desc"},
        )
        prs = []
        for item in items:
            if not item.get("merged_at"):
                continue
            merged_at = datetime.fromisoformat(item["merged_at"].replace("Z", "+00:00"))
            if merged_at < since:
                break
            number = item["number"]
            files = [
                f["filename"]
                for f in self._get_paginated(f"/repos/{owner}/{repo}/pulls/{number}/files")
            ]
            reviews = [
                r["body"]
                for r in self._get_paginated(f"/repos/{owner}/{repo}/pulls/{number}/reviews")
                if r.get("body")
            ]
            prs.append(PR(
                number=number,
                title=item["title"],
                body=item.get("body") or "",
                author=item["user"]["login"],
                labels=[la["name"] for la in item.get("labels", [])],
                files_changed=files,
                review_bodies=reviews,
            ))
        return prs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_github.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/github.py tests/test_github.py
git commit -m "feat: github client data types and merged PR fetching"
```

---

## Task 5: GitHub API Client — Remaining Endpoints

**Files:**
- Modify: `repo_digest/github.py`
- Modify: `tests/test_github.py`

- [ ] **Step 1: Add tests for open PRs, issues, commits, and full repo activity fetch**

Append to `tests/test_github.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_github.py -v
```

Expected: the 5 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement remaining methods in `repo_digest/github.py`**

Add these methods to `GitHubClient` after `fetch_merged_prs`:

```python
    def fetch_open_prs(self, owner: str, repo: str, since: datetime) -> list[PR]:
        items = self._get_paginated(
            f"/repos/{owner}/{repo}/pulls", {"state": "open"}
        )
        prs = []
        for item in items:
            created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            if created_at < since:
                continue
            number = item["number"]
            files = [
                f["filename"]
                for f in self._get_paginated(f"/repos/{owner}/{repo}/pulls/{number}/files")
            ]
            prs.append(PR(
                number=number,
                title=item["title"],
                body=item.get("body") or "",
                author=item["user"]["login"],
                labels=[la["name"] for la in item.get("labels", [])],
                files_changed=files,
                review_bodies=[],
            ))
        return prs

    def fetch_closed_issues(self, owner: str, repo: str, since: datetime) -> list[Issue]:
        items = self._get_paginated(
            f"/repos/{owner}/{repo}/issues",
            {"state": "closed", "sort": "updated", "direction": "desc"},
        )
        issues = []
        for item in items:
            if "pull_request" in item:
                continue
            closed_at = datetime.fromisoformat(item["closed_at"].replace("Z", "+00:00"))
            if closed_at < since:
                break
            issues.append(Issue(
                number=item["number"],
                title=item["title"],
                body=item.get("body") or "",
                author=item["user"]["login"],
                labels=[la["name"] for la in item.get("labels", [])],
            ))
        return issues

    def fetch_open_issues(self, owner: str, repo: str, since: datetime) -> list[Issue]:
        items = self._get_paginated(
            f"/repos/{owner}/{repo}/issues",
            {"state": "open", "sort": "created", "direction": "desc"},
        )
        issues = []
        for item in items:
            if "pull_request" in item:
                continue
            created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            if created_at < since:
                break
            issues.append(Issue(
                number=item["number"],
                title=item["title"],
                body=item.get("body") or "",
                author=item["user"]["login"],
                labels=[la["name"] for la in item.get("labels", [])],
            ))
        return issues

    def fetch_commits(self, owner: str, repo: str, since: datetime) -> list[Commit]:
        since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        items = self._get_paginated(
            f"/repos/{owner}/{repo}/commits",
            {"sha": "main", "since": since_str},
        )
        commits = []
        for item in items:
            if len(item.get("parents", [])) > 1:
                continue  # skip merge commits
            author = item["commit"]["author"]["name"]
            if item.get("author"):
                author = item["author"]["login"]
            commits.append(Commit(
                sha=item["sha"][:7],
                message=item["commit"]["message"].splitlines()[0],
                author=author,
            ))
        return commits

    def fetch_repo_activity(self, repo: str, since: datetime) -> RepoActivity:
        owner, name = repo.split("/", 1)
        return RepoActivity(
            repo=repo,
            merged_prs=self.fetch_merged_prs(owner, name, since),
            open_prs=self.fetch_open_prs(owner, name, since),
            closed_issues=self.fetch_closed_issues(owner, name, since),
            open_issues=self.fetch_open_issues(owner, name, since),
            commits=self.fetch_commits(owner, name, since),
        )
```

Also add this module-level function after the class:

```python
def fetch_all_repos(repos: list[str], since: datetime, token: str) -> list[RepoActivity]:
    """Fetch activity for all repos in parallel (up to 5 concurrent)."""
    client = GitHubClient(token)
    with ThreadPoolExecutor(max_workers=min(len(repos), 5)) as executor:
        futures = [executor.submit(client.fetch_repo_activity, repo, since) for repo in repos]
        return [f.result() for f in futures]
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_github.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/github.py tests/test_github.py
git commit -m "feat: github client complete"
```

---

## Task 6: Summarisation — Format Activity + Per-Repo Summary

**Files:**
- Create: `repo_digest/summarise.py`
- Create: `tests/test_summarise.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_summarise.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_summarise.py -v
```

Expected: `ModuleNotFoundError: No module named 'repo_digest.summarise'`

- [ ] **Step 3: Implement `repo_digest/summarise.py` with `format_activity` and `summarise_repo`**

```python
from datetime import datetime
from dataclasses import dataclass

import litellm

from repo_digest.github import RepoActivity


@dataclass
class Digest:
    overview: str
    repo_summaries: dict[str, str]  # repo -> markdown summary


def format_activity(activity: RepoActivity) -> str:
    """Format repo activity into a text block for LLM consumption."""
    parts: list[str] = []

    if activity.merged_prs:
        parts.append("## Merged Pull Requests")
        for pr in activity.merged_prs:
            parts.append(f"\n**#{pr.number}: {pr.title}** by @{pr.author}")
            if pr.labels:
                parts.append(f"Labels: {', '.join(pr.labels)}")
            if pr.body:
                parts.append(pr.body[:500])
            if pr.files_changed:
                parts.append(f"Files changed: {', '.join(pr.files_changed[:15])}")
            if pr.review_bodies:
                parts.append("Reviews: " + " | ".join(rb[:200] for rb in pr.review_bodies))

    if activity.open_prs:
        parts.append("\n## Open Pull Requests")
        for pr in activity.open_prs:
            parts.append(f"\n**#{pr.number}: {pr.title}** by @{pr.author}")
            if pr.body:
                parts.append(pr.body[:300])
            if pr.files_changed:
                parts.append(f"Files changed: {', '.join(pr.files_changed[:10])}")

    if activity.closed_issues:
        parts.append("\n## Closed Issues")
        for issue in activity.closed_issues:
            label_str = f" [{', '.join(issue.labels)}]" if issue.labels else ""
            parts.append(f"\n**#{issue.number}: {issue.title}**{label_str} by @{issue.author}")
            if issue.body:
                parts.append(issue.body[:300])

    if activity.open_issues:
        parts.append("\n## Open Issues")
        for issue in activity.open_issues:
            label_str = f" [{', '.join(issue.labels)}]" if issue.labels else ""
            parts.append(f"\n**#{issue.number}: {issue.title}**{label_str} by @{issue.author}")
            if issue.body:
                parts.append(issue.body[:300])

    if activity.commits:
        parts.append("\n## Commits to Main")
        for commit in activity.commits:
            parts.append(f"- {commit.sha}: {commit.message} (@{commit.author})")

    return "\n".join(parts)


def summarise_repo(activity: RepoActivity, model: str, since_str: str) -> str:
    """Produce a markdown summary for one repo using the LLM."""
    activity_text = format_activity(activity)

    prompt = f"""You are summarising recent GitHub activity for an engineering leader who wants to stay informed without reading code.

Repository: {activity.repo}
Period: {since_str} to today

{activity_text}

Write a concise markdown summary (200-400 words) covering:
- What was built or changed (be specific: name the features, fixes, or refactors)
- What is currently in progress
- Notable problems, discussions, or decisions
- Who is active and on what

Be specific. Name people and things. Do not use generic filler."""

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_summarise.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/summarise.py tests/test_summarise.py
git commit -m "feat: activity formatting and per-repo summarisation"
```

---

## Task 7: Summarisation — Cross-Repo Synthesis

**Files:**
- Modify: `repo_digest/summarise.py`
- Modify: `tests/test_summarise.py`

- [ ] **Step 1: Add tests for synthesis and full digest**

Append to `tests/test_summarise.py`:

```python
from repo_digest.summarise import synthesise, build_digest
from concurrent.futures import ThreadPoolExecutor


def test_synthesise_calls_litellm(mocker):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "## Overview\n\nTwo repos shipped auth features."
    mocker.patch("repo_digest.summarise.litellm.completion", return_value=mock_response)

    repo_summaries = {
        "org/repo1": "## org/repo1\n\nAlice shipped login.",
        "org/repo2": "## org/repo2\n\nBob fixed auth bug.",
    }
    result = synthesise(repo_summaries, model="ollama/gemma4", since_str="Apr 17")

    import repo_digest.summarise as s
    s.litellm.completion.assert_called_once()
    assert "Two repos shipped auth features" in result


def test_build_digest(mocker):
    activities = [make_activity(), RepoActivity(repo="org/repo2")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "mocked summary"
    mocker.patch("repo_digest.summarise.litellm.completion", return_value=mock_response)

    digest = build_digest(activities, model="ollama/gemma4", since_str="Apr 17")

    assert isinstance(digest, Digest)
    assert "org/myrepo" in digest.repo_summaries
    assert "org/repo2" in digest.repo_summaries
    assert isinstance(digest.overview, str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_summarise.py::test_synthesise_calls_litellm tests/test_summarise.py::test_build_digest -v
```

Expected: `ImportError: cannot import name 'synthesise'`

- [ ] **Step 3: Add `synthesise` and `build_digest` to `repo_digest/summarise.py`**

Append to the bottom of `repo_digest/summarise.py`:

```python
def synthesise(repo_summaries: dict[str, str], model: str, since_str: str) -> str:
    """Produce a cross-repo overview from per-repo summaries."""
    summaries_text = "\n\n".join(
        f"### {repo}\n{summary}" for repo, summary in repo_summaries.items()
    )

    prompt = f"""You are writing an executive overview for an engineering leader who follows multiple repos in their org.

Period: {since_str} to today

Here are summaries of recent activity across repositories:

{summaries_text}

Write a concise markdown overview (100-200 words):
- Major themes or patterns across repos
- Notable work or developments worth highlighting
- Anything that stands out

This appears at the top of the digest. Be specific and concise."""

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def build_digest(
    activities: list[RepoActivity], model: str, since_str: str
) -> Digest:
    """Run the full two-stage summarisation pipeline."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(activities), 5)) as executor:
        futures = {
            activity.repo: executor.submit(summarise_repo, activity, model, since_str)
            for activity in activities
        }
        repo_summaries = {repo: future.result() for repo, future in futures.items()}

    overview = synthesise(repo_summaries, model=model, since_str=since_str)
    return Digest(overview=overview, repo_summaries=repo_summaries)
```

- [ ] **Step 4: Run all summarise tests**

```bash
pytest tests/test_summarise.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add repo_digest/summarise.py tests/test_summarise.py
git commit -m "feat: cross-repo synthesis and full digest pipeline"
```

---

## Task 8: CLI Integration

**Files:**
- Create: `repo_digest/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'repo_digest.cli'`

- [ ] **Step 3: Implement `repo_digest/cli.py`**

```python
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from repo_digest.config import DEFAULT_CONFIG_PATH, load_config
from repo_digest.github import fetch_all_repos
from repo_digest.since import parse_since
from repo_digest.summarise import build_digest

app = typer.Typer(help="Summarise recent activity across GitHub repos using an LLM.")
console = Console()


@app.command()
def main(
    since: str = typer.Option(..., "--since", help="Start of window: '7d', '2w', '1m', or ISO date like '2026-04-15'"),
    config: Optional[Path] = typer.Option(None, "--config", help=f"Config file path (default: {DEFAULT_CONFIG_PATH})"),
) -> None:
    config_path = config or DEFAULT_CONFIG_PATH

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        typer.echo("Error: GITHUB_TOKEN environment variable is not set.", err=True)
        raise typer.Exit(code=1)

    try:
        since_dt = parse_since(since)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    since_str = since_dt.strftime("%b %-d, %Y")

    with console.status("Fetching GitHub activity..."):
        activities = fetch_all_repos(cfg.repos, since_dt, token)

    with console.status("Generating digest..."):
        digest = build_digest(activities, model=cfg.llm.model, since_str=since_str)

    console.print(Markdown(f"# Repo Digest ({since_str} – today)\n\n## Overview\n\n{digest.overview}"))
    for repo, summary in digest.repo_summaries.items():
        console.print(Markdown(f"---\n\n## {repo}\n\n{summary}"))
```

- [ ] **Step 4: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Smoke test the CLI help**

```bash
repo-digest --help
```

Expected: usage message showing `--since` and `--config` options.

- [ ] **Step 6: Commit**

```bash
git add repo_digest/cli.py tests/test_cli.py
git commit -m "feat: CLI integration — wire everything together"
```

---

## Done

The tool is fully implemented. To use it:

1. Create `~/.repo-digest/config.yaml` with your repos and optional LLM config.
2. Set `GITHUB_TOKEN` in your environment.
3. Run: `repo-digest --since 7d`
