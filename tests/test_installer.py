"""The installer patches copies of the REAL files of repo/chi-bench (if present)."""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

AGENTE = Path(__file__).resolve().parents[1]
CHI = Path(os.environ.get("CHI_BENCH_DIR") or AGENTE.parent / "repo" / "chi-bench")

spec = importlib.util.spec_from_file_location("installer", AGENTE / "scripts" / "install_into_chi_bench.py")
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)

needs_repo = pytest.mark.skipif(not (CHI / "docker" / "Dockerfile").exists(), reason="repo/chi-bench not cloned")


@needs_repo
def test_patch_registry_idempotent():
    text = (CHI / "src/chi_bench/experiment/agents/registry.py").read_text(encoding="utf-8")
    once = inst.patch_registry(text)
    assert inst.IMPORT_PATH in once and inst.patch_registry(once) == once
    ns: dict = {}
    exec(compile(once, "registry.py", "exec"), ns)
    assert ns["IN_TREE_AGENT_IMPORT_PATHS"]["fiable-agent"] == inst.IMPORT_PATH
    assert "fiable-agent" in ns["KNOWN_AGENTS"]


@needs_repo
def test_patch_dockerfile_adds_copy_inside_agent_stage():
    text = (CHI / "docker/Dockerfile").read_text(encoding="utf-8")
    once = inst.patch_dockerfile(text)
    assert inst.patch_dockerfile(once) == once
    agent_stage = once.split("FROM base AS agent", 1)[1].split("\nFROM ", 1)[0]
    assert inst.DOCKER_COPY.strip() in agent_stage


def test_patch_errors_on_unknown_files():
    with pytest.raises(inst.PatchError):
        inst.patch_dockerfile("FROM python:3.12\n")
    with pytest.raises(inst.PatchError):
        inst.patch_registry("X = 1\n")


@needs_repo
def test_install_apply_on_temp_copy(tmp_path):
    fake = tmp_path / "chi-bench"
    (fake / "src/chi_bench/experiment/agents").mkdir(parents=True)
    (fake / "docker").mkdir()
    shutil.copy(CHI / "src/chi_bench/experiment/agents/registry.py", fake / "src/chi_bench/experiment/agents/")
    shutil.copy(CHI / "docker/Dockerfile", fake / "docker/")
    assert inst.install(fake, apply=False)  # dry-run: writes nothing
    assert not (fake / "src/chi_bench/experiment/agents/fiable_agent").exists()
    inst.install(fake, apply=True)
    pkg = fake / "src/chi_bench/experiment/agents/fiable_agent"
    assert (pkg / "runner.py").exists() and (pkg / "configs" / "default.yaml").exists()
    assert not list(pkg.rglob("__pycache__"))
    assert (fake / "src/chi_bench/experiment/agents/fiable_agent_harness.py").exists()
    assert "fiable-agent" in (fake / "src/chi_bench/experiment/agents/registry.py").read_text(encoding="utf-8")



def test_package_imports_when_nested_like_in_the_image(tmp_path, monkeypatch):
    """In the image the package lives at chi_bench.experiment.agents.fiable_agent."""
    import sys

    base = tmp_path / "src" / "chi_bench" / "experiment" / "agents"
    base.mkdir(parents=True)
    for d in (base.parents[1], base.parents[0], base):
        (d / "__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(AGENTE / "fiable_agent", base / "fiable_agent",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    for name in [m for m in sys.modules if m == "chi_bench" or m.startswith("chi_bench.")]:
        monkeypatch.delitem(sys.modules, name)
    import importlib

    runner = importlib.import_module("chi_bench.experiment.agents.fiable_agent.runner")
    args = runner.parse_args(["--instruction-file", "x", "--mcp-url", "http://localhost:8100/mcp"])
    assert args.mcp_url.endswith("/mcp")
    cfg = importlib.import_module("chi_bench.experiment.agents.fiable_agent.config").load_config()
    assert cfg.profile == "default"
