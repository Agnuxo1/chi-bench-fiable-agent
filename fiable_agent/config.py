"""Agent configuration (YAML): models, RAG, checklist, critic and loop.

Security rules:
  * Credentials NEVER go in the YAML: only via environment variables
    (``ANTHROPIC_API_KEY``). ``load_config`` rejects keys named like secrets
    and values shaped like tokens.
  * The value of a credential is never printed or logged.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_NAME = "default.yaml"

# IDs taken from repo/chi-bench/configs/prices.yaml (keys "anthropic/<id>").
MODEL_ALIASES: dict[str, str] = {
    "sonnet": "anthropic/claude-sonnet-5",
    "opus": "anthropic/claude-opus-5",
    "opus-4-7": "anthropic/claude-opus-4-7",
    "haiku": "anthropic/claude-haiku-4-5",
}

API_KEY_ENV = "ANTHROPIC_API_KEY"
CONFIG_ENV = "FIABLE_AGENT_CONFIG"
MODEL_ENV = "FIABLE_AGENT_MODEL"

_FORBIDDEN_KEYS = {
    "api_key",
    "anthropic_api_key",
    "openai_api_key",
    "hf_token",
    "token",
    "access_token",
    "password",
    "secret",
}
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
)


class ConfigError(ValueError):
    """Invalid or unsafe configuration."""


@dataclass
class ModelConfig:
    agent: str = "sonnet"
    critic: str | None = None  # None => same model as the agent (consistent cost)
    max_output_tokens: int = 8192
    temperature: float | None = None
    # Extra parameters passed as-is to messages.create (e.g. thinking).
    extra_request_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagConfig:
    enabled: bool = True
    backend: str = "bm25"  # bm25 | faiss | null
    handbook_dir: str = "/workspace/skills/managed-care-operations-handbook"
    top_k: int = 4
    chunk_chars: int = 1800
    max_files: int = 5000
    inject_on_start: bool = True
    max_inject_chars: int = 6000


@dataclass
class ChecklistConfig:
    enabled: bool = True
    enforce_finish_gate: bool = True
    max_finish_nudges: int = 3


@dataclass
class CriticConfig:
    enabled: bool = True
    judges: list[str] = field(default_factory=lambda: ["contract", "grounding", "policy"])
    mode: str = "llm"  # llm | heuristic  (affects only the policy judge)
    max_blocks_per_action: int = 2
    policy_max_tokens: int = 700


@dataclass
class LoopConfig:
    max_turns: int = 80
    max_tool_result_chars: int = 40000
    wall_clock_budget_s: float = 1500.0
    prompt_caching: bool = True


@dataclass
class AgentConfig:
    profile: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    checklist: ChecklistConfig = field(default_factory=ChecklistConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    prices_file: str | None = None

    @property
    def agent_model(self) -> str:
        return resolve_model(self.model.agent)

    @property
    def critic_model(self) -> str:
        return resolve_model(self.model.critic or self.model.agent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model(name: str) -> str:
    """Alias (sonnet/opus/haiku) -> id with provider prefix ``anthropic/<id>``."""
    if not name or not str(name).strip():
        raise ConfigError("model name is empty")
    key = str(name).strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if "/" not in key:
        return f"anthropic/{key}"
    return key


def api_model_id(model: str) -> str:
    """Id expected by the Anthropic API (without the ``anthropic/`` prefix)."""
    resolved = resolve_model(model)
    return resolved.split("/", 1)[1] if resolved.startswith("anthropic/") else resolved


def _assert_no_secrets(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).strip().lower() in _FORBIDDEN_KEYS:
                raise ConfigError(
                    f"config key '{path}{key}' looks like a credential; use environment variables"
                )
            _assert_no_secrets(value, f"{path}{key}.")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_secrets(value, f"{path}{i}.")
    elif isinstance(node, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(node):
                raise ConfigError(f"config value at '{path.rstrip('.')}' looks like a secret token")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular 'extends' in {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    _assert_no_secrets(data)
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        data = _deep_merge(_read_yaml(parent_path, _seen + (path,)), data)
    return data


def _build(data: dict[str, Any]) -> AgentConfig:
    known = {"profile", "model", "rag", "checklist", "critic", "loop", "prices_file"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    try:
        cfg = AgentConfig(
            profile=str(data.get("profile", "default")),
            model=ModelConfig(**(data.get("model") or {})),
            rag=RagConfig(**(data.get("rag") or {})),
            checklist=ChecklistConfig(**(data.get("checklist") or {})),
            critic=CriticConfig(**(data.get("critic") or {})),
            loop=LoopConfig(**(data.get("loop") or {})),
            prices_file=data.get("prices_file"),
        )
    except TypeError as exc:  # unknown field inside a section
        raise ConfigError(str(exc)) from exc
    if cfg.rag.backend not in {"bm25", "faiss", "null"}:
        raise ConfigError(f"rag.backend must be bm25|faiss|null, got {cfg.rag.backend!r}")
    if cfg.critic.mode not in {"llm", "heuristic"}:
        raise ConfigError(f"critic.mode must be llm|heuristic, got {cfg.critic.mode!r}")
    bad = set(cfg.critic.judges) - {"contract", "grounding", "policy"}
    if bad:
        raise ConfigError(f"unknown judges: {sorted(bad)}")
    resolve_model(cfg.model.agent)
    return cfg


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> AgentConfig:
    """Load a YAML file (with optional ``extends:``) and apply overrides and environment.

    Order: file -> ``overrides`` -> ``FIABLE_AGENT_MODEL`` (if set).
    ``path`` may be a packaged profile name (``"opus"``) or a file path.
    """
    env = dict(os.environ if env is None else env)
    if path is None:
        path = env.get(CONFIG_ENV) or DEFAULT_CONFIG_NAME
    candidate = Path(str(path))
    if not candidate.exists():
        name = candidate.name if candidate.suffix else f"{candidate.name}.yaml"
        candidate = CONFIG_DIR / name
    if not candidate.exists():
        raise ConfigError(f"config not found: {path}")
    data = _read_yaml(candidate)
    if overrides:
        _assert_no_secrets(overrides)
        data = _deep_merge(data, overrides)
    if env.get(MODEL_ENV):
        data = _deep_merge(data, {"model": {"agent": env[MODEL_ENV]}})
    return _build(data)


def has_api_key(env: dict[str, str] | None = None) -> bool:
    """True if the credential is present (never returns or prints its value)."""
    env = os.environ if env is None else env
    return bool((env.get(API_KEY_ENV) or "").strip())

