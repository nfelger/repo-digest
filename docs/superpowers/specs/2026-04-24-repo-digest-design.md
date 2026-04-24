# Repo Digest — Design Spec

## Purpose

A CLI tool that summarises recent activity across multiple GitHub repos using LLMs. Designed for engineering leaders and senior ICs who follow repos in their org but don't have time to read every PR and commit. Produces a rich, readable digest of who did what, what's being built, what problems are surfacing, and what practices are in play.

## CLI Interface

```
repo-digest --since 7d
repo-digest --since 2026-04-15
repo-digest --config path/to/config.yaml
```

- `--since` (required): accepts relative durations (`7d`, `2w`, `1m`) or absolute ISO dates.
- `--config` (optional): path to config file. Defaults to `~/.repo-digest/config.yaml`.
- No persistent state. Scheduling is external (cron, etc.).

## Config File

```yaml
# ~/.repo-digest/config.yaml

repos:
  - org/repo-one
  - org/repo-two
  - other-org/some-repo

llm:
  model: ollama/gemma4    # litellm model string
```

- `repos`: list of GitHub repos in `owner/repo` format.
- `llm.model`: LiteLLM model string. Provider is encoded in the string (e.g. `anthropic/claude-sonnet-4-20250514`, `ollama/gemma4`, `gpt-4o`). Defaults to `ollama/gemma4`.
- API keys follow standard env var conventions per provider (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Not configured in the file.
- GitHub token via the standard `GITHUB_TOKEN` env var.

## GitHub Data Fetching

For each repo, fetch activity within the `--since` window via GitHub REST API:

- **Merged PRs** — title, description, author, changed files, review comments
- **Opened PRs** (still open) — title, description, author, changed files, review comments
- **Closed issues** — title, labels, discussion
- **Opened issues** — title, labels, discussion
- **Commits to main** (not already covered by merged PRs) — message, author, files changed

Uses `httpx` for HTTP requests. Repos are fetched in parallel.

## LLM Integration

Uses LiteLLM (`litellm.completion()`) as the provider abstraction. No custom wrapper needed. Supports Ollama, Anthropic, OpenAI, and 140+ other providers via a single interface.

Simple prompt in, text out. No streaming, no tool use, no multi-turn.

## Summarisation Pipeline

Two-stage process:

**Stage 1 — Per-repo summary:** For each repo, assemble fetched GitHub data into a prompt. The prompt asks the LLM to summarise: what was built, what's in progress, notable discussions, who's active and on what. Repos are summarised in parallel.

**Stage 2 — Cross-repo synthesis:** All per-repo summaries are fed into a single prompt that produces a high-level digest — themes, notable activity, anything that stands out across repos.

Prompts are plain strings in the code. No template engine.

## Terminal Output

LLM output is markdown, rendered to the terminal via `rich.markdown`. The overall structure:

- Cross-repo overview first (from stage 2)
- Per-repo detail sections below (from stage 1)

No custom formatting logic — the LLM structures the markdown.

## Project Structure

```
repo_digest/
    __init__.py
    cli.py              # CLI entry point (typer)
    config.py            # YAML config loading
    github.py            # GitHub API fetching
    summarise.py         # Prompt construction + two-stage pipeline
```

Single flat package. No subpackages.

## Dependencies

- **typer** — CLI framework
- **httpx** — HTTP client (GitHub API + Ollama via LiteLLM)
- **litellm** — LLM provider abstraction
- **rich** — markdown rendering in terminal
- **pyyaml** — config file parsing

## What This Is Not

- Not opinionated — it summarises activity, it doesn't judge practices. The reader draws their own conclusions.
- Not a dashboard — it's a CLI that produces a point-in-time digest.
- Not stateful — no database, no history, no tracking of previous runs.
- Cross-repo trends are a future extension, not part of this design.
