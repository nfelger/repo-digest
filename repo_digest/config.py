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
    # Only pass known fields so unrecognised YAML keys don't raise TypeError
    llm_kwargs = {k: v for k, v in llm_data.items() if k in {"model"}}
    return Config(
        repos=data["repos"],
        llm=LLMConfig(**llm_kwargs),
    )
