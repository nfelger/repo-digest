from dataclasses import dataclass, field
from datetime import datetime
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
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
            # params are encoded in the next-page URL from the Link header
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
                # sort is by updated, not merged_at — a post-merge comment could push an
                # old PR to the top, so skip rather than break
                continue
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


def fetch_all_repos(repos: list[str], since: datetime, token: str) -> list[RepoActivity]:
    """Fetch activity for all repos in parallel (up to 5 concurrent)."""
    client = GitHubClient(token)
    with ThreadPoolExecutor(max_workers=min(len(repos), 5)) as executor:
        futures = [executor.submit(client.fetch_repo_activity, repo, since) for repo in repos]
        return [f.result() for f in futures]
