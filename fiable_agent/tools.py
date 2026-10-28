"""Tool gateway: MCP (streamable-http) + safe local tools.

* chi-Bench MCP tool names use a dot (``review.submit_md_case_decision``,
  see repo/chi-bench/src/chi_bench/mcp/tool_naming.py). They are exposed to the
  LLM sanitized (``review__submit_md_case_decision``) and translated back.
* Local tools: Handbook search, file reads on allowed paths and writes ONLY
  under the case workspace. Verifier/solution paths are denied (Rules section 8:
  no access to hidden files).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .rag import NullRetriever, Retriever, format_hits

_API_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def to_api_name(name: str) -> str:
    api = _API_NAME_RE.sub("_", name.replace(".", "__"))
    if len(api) > 64:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        api = f"{api[:55]}_{digest}"
    return api


class ToolNameMap:
    def __init__(self) -> None:
        self._to_api: dict[str, str] = {}
        self._to_canonical: dict[str, str] = {}

    def register(self, canonical: str) -> str:
        api = to_api_name(canonical)
        if api in self._to_canonical and self._to_canonical[api] != canonical:
            api = to_api_name(canonical + "#" + str(len(self._to_api)))
        self._to_api[canonical] = api
        self._to_canonical[api] = canonical
        return api

    def api(self, canonical: str) -> str:
        return self._to_api.get(canonical) or self.register(canonical)

    def canonical(self, api: str) -> str:
        return self._to_canonical.get(api, api)


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolOutcome:
    ok: bool
    text: str
    data: Any = None


class ToolGateway(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome: ...


class MCPGateway:
    """MCP streamable-http client (``mcp`` SDK, present in the image venv)."""

    def __init__(self, url: str):
        self.url = url
        self._stack = None
        self._session = None

    async def __aenter__(self) -> "MCPGateway":
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = AsyncExitStack()
        read, write, *_ = await self._stack.enter_async_context(streamablehttp_client(self.url))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._session.list_tools()
        return [
            ToolSpec(t.name, t.description or "", dict(t.inputSchema or {"type": "object", "properties": {}}))
            for t in result.tools
        ]

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        try:
            result = await self._session.call_tool(name, args)
        except Exception as exc:  # transport or validation error
            return ToolOutcome(False, f"tool call failed: {type(exc).__name__}: {exc}")
        parts = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else json.dumps(getattr(block, "model_dump", dict)()))
        text = "\n".join(parts)
        if not text and getattr(result, "structuredContent", None) is not None:
            text = json.dumps(result.structuredContent)
        return ToolOutcome(not bool(getattr(result, "isError", False)), text)


Handler = Callable[[dict[str, Any]], tuple[bool, Any]]


class FakeGateway:
    """Test gateway: ``handlers[name](args) -> (ok, payload)``."""

    def __init__(self, specs: list[ToolSpec], handlers: dict[str, Handler] | None = None):
        self.specs = specs
        self.handlers = handlers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return list(self.specs)

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        self.calls.append((name, dict(args)))
        handler = self.handlers.get(name)
        if handler is None:
            return ToolOutcome(True, json.dumps({"ok": True, "tool": name}))
        ok, payload = handler(args)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return ToolOutcome(ok, text)


# --- local tools -------------------------------------------------------

_DENY_SEGMENTS = frozenset({"solution", "tests", "verifier", "fixtures", "judge"})
_DENY_SUBSTRINGS = ("expectations", "rubrics.json", "/opt/chi-bench/tasks", "/logs/verifier")


@dataclass
class LocalToolRoots:
    handbook: str = "/workspace/skills/managed-care-operations-handbook"
    readable: tuple[str, ...] = (
        "/workspace/skills",
        "/opt/healthverse-task-assets",
        "/opt/chi-bench-task-assets",
        "/logs/artifacts/workspaces",
        "/workspace/raw",
    )
    writable: tuple[str, ...] = ("/logs/artifacts/workspaces",)


class LocalTools:
    SEARCH = "handbook_search"
    READ = "read_file"
    LIST = "list_dir"
    WRITE = "write_workspace_file"

    def __init__(self, retriever: Retriever | None = None, roots: LocalToolRoots | None = None,
                 top_k: int = 4, max_read_chars: int = 30000):
        self.retriever = retriever or NullRetriever()
        self.roots = roots or LocalToolRoots()
        self.top_k = top_k
        self.max_read_chars = max_read_chars

    @property
    def names(self) -> set[str]:
        return {self.SEARCH, self.READ, self.LIST, self.WRITE}

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": self.SEARCH,
             "description": "Search the Managed-Care Operations Handbook (local index). Returns passages with "
                            "[document § section] citations. Use it for policy criteria and procedures.",
             "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                                "k": {"type": "integer"}},
                              "required": ["query"]}},
            {"name": self.READ,
             "description": "Read a text file under the handbook, the tool reference, raw inputs or the case "
                            "workspace. Verifier/solution/test files are not accessible.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"},
                                                                "offset": {"type": "integer"},
                                                                "limit": {"type": "integer"}},
                              "required": ["path"]}},
            {"name": self.LIST,
             "description": "List a directory under the allowed roots.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": self.WRITE,
             "description": "Write a UTF-8 text file under /logs/artifacts/workspaces/<case-id>/... "
                            "(working notes or required correspondence). Never fabricate source evidence.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"},
                                                                "content": {"type": "string"}},
                              "required": ["path", "content"]}},
        ]

    @staticmethod
    def _norm(path: str) -> str:
        return str(PurePosixPath(str(path).replace("\\", "/")))

    def _allowed(self, path: str, roots: tuple[str, ...]) -> bool:
        p = self._norm(path)
        parts = PurePosixPath(p).parts
        if ".." in parts:
            return False
        low = p.lower()
        if any(seg.lower() in _DENY_SEGMENTS for seg in parts) or any(s in low for s in _DENY_SUBSTRINGS):
            return False
        return any(p == r or p.startswith(r.rstrip("/") + "/") for r in map(self._norm, roots))

    def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        try:
            if name == self.SEARCH:
                if not self.retriever.available:
                    return ToolOutcome(True, "handbook index unavailable; read SKILL.md with read_file instead")
                hits = self.retriever.search(str(args.get("query", "")), int(args.get("k") or self.top_k))
                return ToolOutcome(True, format_hits(hits) or "no matches")
            if name == self.READ:
                path = str(args.get("path", ""))
                if not self._allowed(path, self.roots.readable):
                    return ToolOutcome(False, f"access denied: {path}")
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                offset = max(int(args.get("offset") or 0), 0)
                limit = min(int(args.get("limit") or self.max_read_chars), self.max_read_chars)
                chunk = text[offset:offset + limit]
                more = "" if offset + limit >= len(text) else f"\n[truncated; total {len(text)} chars]"
                return ToolOutcome(True, chunk + more)
            if name == self.LIST:
                path = str(args.get("path", ""))
                if not self._allowed(path, self.roots.readable):
                    return ToolOutcome(False, f"access denied: {path}")
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in Path(path).iterdir())
                return ToolOutcome(True, "\n".join(entries[:500]))
            if name == self.WRITE:
                path = str(args.get("path", ""))
                if not self._allowed(path, self.roots.writable):
                    return ToolOutcome(False, f"write denied (only under {self.roots.writable}): {path}")
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(args.get("content", "")), encoding="utf-8")
                return ToolOutcome(True, f"wrote {len(str(args.get('content', '')))} chars to {path}")
        except (OSError, ValueError) as exc:
            return ToolOutcome(False, f"{type(exc).__name__}: {exc}")
        return ToolOutcome(False, f"unknown local tool {name}")

