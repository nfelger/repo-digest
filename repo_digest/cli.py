import os
from pathlib import Path

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
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Config file path"),
) -> None:
    config_path = config

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
