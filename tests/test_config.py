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
