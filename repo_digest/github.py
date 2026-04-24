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
            trust_env=False,
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
