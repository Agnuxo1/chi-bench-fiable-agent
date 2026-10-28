"""Pluggable LLM client.

* ``AnthropicClient`` - Messages API (``anthropic`` SDK, already present in the venv of
  the ``chi-bench:agent`` image). The key is read from the environment by the SDK
  (``ANTHROPIC_API_KEY``); this module never touches or prints it.
* ``ScriptedLLM`` - network-free test doubles.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import api_model_id


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    latency_s: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def reasoning(self) -> str:
        return "\n".join(b.get("thinking", "") for b in self.content if b.get("type") == "thinking").strip()


class LLMClient(Protocol):
    def create(
        self,
        *,
        model: str,
        system: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse: ...


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(obj.__dict__)


class AnthropicClient:
    def __init__(self, *, max_retries: int = 4, timeout_s: float = 180.0):
        import anthropic  # deferred import: tests do not need it

        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)

    def create(
        self,
        *,
        model: str,
        system: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": api_model_id(model),
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
        params.update({k: v for k, v in kwargs.items() if v is not None})
        t0 = time.monotonic()
        resp = self._client.messages.create(**params)
        return LLMResponse(
            content=[_to_dict(b) for b in resp.content],
            stop_reason=str(resp.stop_reason or ""),
            usage=_to_dict(resp.usage) if resp.usage is not None else {},
            model=model,
            latency_s=time.monotonic() - t0,
        )


Script = LLMResponse | Callable[[dict[str, Any]], LLMResponse]


class ScriptedLLM:
    """Returns predefined responses; records every request (for tests)."""

    def __init__(self, script: list[Script] | None = None, default: Script | None = None):
        self.script = list(script or [])
        self.default = default
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> LLMResponse:
        self.requests.append(kwargs)
        item = self.script.pop(0) if self.script else self.default
        if item is None:
            raise RuntimeError("ScriptedLLM exhausted")
        resp = item(kwargs) if callable(item) else item
        if not resp.model:
            resp.model = kwargs.get("model", "")
        return resp


def text_response(text: str, usage: dict[str, Any] | None = None) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": text}], "end_turn", usage or {"input_tokens": 10, "output_tokens": 5})


def tool_response(calls: list[tuple[str, str, dict[str, Any]]], text: str = "",
                  usage: dict[str, Any] | None = None) -> LLMResponse:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for call_id, name, args in calls:
        content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
    return LLMResponse(content, "tool_use", usage or {"input_tokens": 100, "output_tokens": 20})

