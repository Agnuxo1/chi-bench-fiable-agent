"""Install fiable_agent into a chi-bench checkout (local fork).

What it does (dry-run by default; pass ``--apply`` to write):
  1. Copies ``fiable_agent`` -> ``src/chi_bench/experiment/agents/fiable_agent``
  2. Copies ``harness/fiable_agent_harness.py`` and ``harness/fiable_cc_harness.py``
     -> ``src/chi_bench/experiment/agents/``
  3. Registers ``fiable-agent`` and ``fiable-cc`` in ``src/chi_bench/experiment/agents/registry.py``
  4. Adds a ``COPY`` of the package to the ``agent`` target of ``docker/Dockerfile``
     (that target only copies specific files; without this the runner never
     reaches the image)

Afterwards: ``uv run cb agent list`` and ``uv run cb docker build``.

Usage:
  python scripts/install_into_chi_bench.py --chi-bench repo/chi-bench          # dry-run
  python scripts/install_into_chi_bench.py --chi-bench repo/chi-bench --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

AGENTS = {
    "fiable-agent": "chi_bench.experiment.agents.fiable_agent_harness:FiableAgentHarness",
    "fiable-cc": "chi_bench.experiment.agents.fiable_cc_harness:FiableClaudeCodeHarness",
}
AGENT_NAME = "fiable-agent"
IMPORT_PATH = AGENTS[AGENT_NAME]
REGISTRY_LINE = f'    "{AGENT_NAME}": "{IMPORT_PATH}",\n'
HARNESS_FILES = ("fiable_agent_harness.py", "fiable_cc_harness.py")
REGISTRY_ANCHOR = "IN_TREE_AGENT_IMPORT_PATHS: dict[str, str] = {"
DOCKER_COPY = ("COPY src/chi_bench/experiment/agents/fiable_agent "
               "/workspace/src/chi_bench/experiment/agents/fiable_agent\n")
AGENT_STAGE = "FROM base AS agent"
AGENTS_DIR_SUFFIX = "/workspace/src/chi_bench/experiment/agents/"

HERE = Path(__file__).resolve().parent
AGENTE = HERE.parent


class PatchError(RuntimeError):
    pass


def patch_registry(text: str) -> str:
    missing = "".join(f'    "{name}": "{path}",\n' for name, path in AGENTS.items() if f'"{name}"' not in text)
    if not missing:
        return text
    idx = text.find(REGISTRY_ANCHOR)
    if idx < 0:
        raise PatchError("registry anchor not found (IN_TREE_AGENT_IMPORT_PATHS)")
    eol = text.index("\n", idx) + 1
    return text[:eol] + missing + text[eol:]


def patch_dockerfile(text: str) -> str:
    if DOCKER_COPY.strip() in text:
        return text
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == AGENT_STAGE)
    except StopIteration as exc:
        raise PatchError(f"'{AGENT_STAGE}' not found in Dockerfile") from exc
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("FROM "):
            break
        if lines[i].rstrip().endswith(AGENTS_DIR_SUFFIX):
            return "".join(lines[: i + 1] + [DOCKER_COPY] + lines[i + 1:])
    raise PatchError("agents COPY block not found in the agent stage")


def _copy_tree(src: Path, dst: Path, apply: bool) -> None:
    print(f"copy {src} -> {dst}")
    if apply:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def install(chi_bench: Path, apply: bool) -> list[str]:
    changes: list[str] = []
    agents_dir = chi_bench / "src" / "chi_bench" / "experiment" / "agents"
    registry = agents_dir / "registry.py"
    dockerfile = chi_bench / "docker" / "Dockerfile"
    for path in (agents_dir, registry, dockerfile):
        if not path.exists():
            raise PatchError(f"not a chi-bench checkout: missing {path}")

    _copy_tree(AGENTE / "fiable_agent", agents_dir / "fiable_agent", apply)
    changes.append("package")
    for harness_name in HARNESS_FILES:
        harness_src = AGENTE / "harness" / harness_name
        print(f"copy {harness_src} -> {agents_dir / harness_src.name}")
        if apply:
            shutil.copy2(harness_src, agents_dir / harness_src.name)
    changes.append("harness")

    for path, fn in ((registry, patch_registry), (dockerfile, patch_dockerfile)):
        old = path.read_text(encoding="utf-8")
        new = fn(old)
        if new != old:
            print(f"patch {path}")
            changes.append(path.name)
            if apply:
                path.write_text(new, encoding="utf-8")
        else:
            print(f"already patched: {path}")
    if not apply:
        print("dry-run: nothing written (use --apply)")
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chi-bench", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        install(args.chi_bench.resolve(), args.apply)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

