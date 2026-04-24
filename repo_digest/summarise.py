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
        for commit in activity.commits[:50]:
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
