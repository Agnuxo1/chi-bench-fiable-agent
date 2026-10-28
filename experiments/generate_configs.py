"""Generate the flat ExperimentConfig files (repo/chi-bench/src/chi_bench/experiment/config.py)
for experiments E0-E7 of the experiment plan. Run from anywhere:

    python experiments/generate_configs.py

The YAML paths are relative to ``repo/chi-bench`` (the CWD of ``cb``).
"dev" subsets: chosen deterministically from PUBLIC METADATA
(``tasks.jsonl``: domain, expected status, stage), stratified by type.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent / "configs"

DOMAIN_DIR = {
    "pa": "data/prior_auth_provider/tasks",
    "um": "data/prior_auth_um/tasks",
    "cm": "data/care_management/tasks",
}
DEV = {
    # 2 submitted / 3 docs_needed / 1 closed
    "pa": ["pa_t008_t008_o002_p01_new_referral_provider", "pa_t011_t011_o003_p01_new_referral_provider",
           "pa_t012_t012_o001_p01_new_referral_provider", "pa_t013_t013_o002_p01_new_referral_provider",
           "pa_t015_t015_o002_p01_new_referral_provider", "pa_t016_t016_o001_p01_new_referral_provider"],
    # intake / triage / nurse / 2x MD / P2P
    "um": ["pa_t008_t008_o002_p01_mdreview_payer", "pa_t011_t011_o003_p01_intake_payer",
           "pa_t012_t012_o001_p01_triage_payer", "pa_t013_t013_o002_p01_nurse_review_payer",
           "pa_t016_t016_o001_p01_p2p_payer", "pa_t017_t017_o001_p01_mdreview_payer"],
    # 2 anxious / 2 hard_refuses / 1 low_coop / 1 tentative
    "cm": ["cm_afib_moderate_anxious_001", "cm_anorexia_hard_refuses_001", "cm_asthma_low_coop_001",
           "cm_ckd_moderate_anxious_001", "cm_complex_esrd_dm_hard_refuses_001",
           "cm_complex_parkinson_dep_moderate_tentative_001"],
}
SMOKE = {"pa": "pa_t014_t014_o001_p01_new_referral_provider",
         "um": "pa_t008_t008_o002_p01_mdreview_payer",
         "cm": "cm_afib_moderate_anxious_001"}

SONNET = "anthropic/claude-sonnet-5"
OPUS = "anthropic/claude-opus-5"
HAIKU = "anthropic/claude-haiku-4-5"


def config(name: str, domain: str, *, profile: str | None, model: str, agent: str = "fiable-agent",
           tasks: list[str] | None = None, single: str | None = None, n_attempts: int = 1,
           concurrency: int = 2) -> dict:
    cfg: dict = {
        "dataset": f"{DOMAIN_DIR[domain]}/{single}" if single else DOMAIN_DIR[domain],
        "data_root": "data",
        "agent": agent,
        "model": model,
        "environment": "docker",
        "env_file": ".env",
        "concurrency": concurrency,
        "n_attempts": n_attempts,
        "max_retries": 2,
        "timeout_multiplier": 2.0,
        "trials_dir": f"logs/experiments/{name}",
    }
    if tasks and not single:
        cfg["include_task_name"] = list(tasks)
    if profile and agent == "fiable-agent":
        cfg["agent_env"] = {"FIABLE_AGENT_CONFIG": profile}
    if profile and agent == "fiable-cc":
        cfg["agent_env"] = {"FIABLE_CC_ARM": profile}
    return cfg


def build() -> dict[str, dict]:
    out: dict[str, dict] = {}
    # E0 - smoke: official harness + our agent, 1 task per domain
    out["e0_smoke_claude_code_um"] = config("e0_smoke_claude_code_um", "um", profile=None, model=SONNET,
                                            agent="claude-code", single=SMOKE["um"], concurrency=1)
    for d in ("pa", "um", "cm"):
        out[f"e0_smoke_fiable_{d}"] = config(f"e0_smoke_fiable_{d}", d, profile="default", model=SONNET,
                                             single=SMOKE[d], concurrency=1)
    # E1/E2 - PA (provider + UM), dev subset
    for exp, profile in (("e1_baseline", "e1_baseline"), ("e2_rag", "e2_rag")):
        for d in ("pa", "um"):
            out[f"{exp}_{d}"] = config(f"{exp}_{d}", d, profile=profile, model=SONNET, tasks=DEV[d])
    # E3 - checklist, three domains
    for d in ("pa", "um", "cm"):
        out[f"e3_checklist_{d}"] = config(f"e3_checklist_{d}", d, profile="e3_checklist", model=SONNET, tasks=DEV[d])
    # E4 - critic; pass^3 with 3 attempts versus the E3 reference with 3 attempts
    for d in ("pa", "um", "cm"):
        out[f"e4_critic_{d}_x3"] = config(f"e4_critic_{d}_x3", d, profile="e4_critic", model=SONNET,
                                          tasks=DEV[d], n_attempts=3)
        out[f"e4_ref_e3_{d}_x3"] = config(f"e4_ref_e3_{d}_x3", d, profile="e3_checklist", model=SONNET,
                                          tasks=DEV[d], n_attempts=3)
    # E5 - model sweep (Sonnet = attempt 1 of E4)
    for tag, model in (("opus", OPUS), ("haiku", HAIKU)):
        for d in ("pa", "um", "cm"):
            out[f"e5_{tag}_{d}"] = config(f"e5_{tag}_{d}", d, profile="e4_critic", model=model, tasks=DEV[d])
    # E7 - heuristic critic (0 extra LLM calls) for the cost tie-breaker
    for d in ("pa", "um", "cm"):
        out[f"e7_heuristic_{d}"] = config(f"e7_heuristic_{d}", d, profile="e4_critic_heuristic", model=SONNET,
                                          tasks=DEV[d])
    # ABL - pre-registered ablation (docs/EVALUATION_PROTOCOL.md): official control + three guard arms,
    # 3 attempts each, same model, same per-task limits. abl1_* = 1 task/domain, abl3_* = 3 tasks/domain.
    # Care management is excluded unless a paid key powers its patient simulator.
    for k in (1, 3):
        for arm, (agent, profile) in ABLATION_ARMS.items():
            for d in ("pa", "um"):
                name = f"abl{k}_{arm}_{d}_x3"
                out[name] = config(name, d, profile=profile, model=SONNET, agent=agent, tasks=DEV[d][:k],
                                   n_attempts=3)
    return out


ABLATION_ARMS = {
    "c0_official": ("claude-code", None),       # control: official Claude Code harness, untouched
    "c1_plan": ("fiable-cc", "plan"),           # + operating principles and role plan
    "c2_checklist": ("fiable-cc", "checklist"),  # + contract-checklist hooks and closure gate
    "c3_critic": ("fiable-cc", "critic"),       # + three-judge critic panel (BenchClaw lineage)
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.yaml"):
        old.unlink()
    for name, cfg in build().items():
        header = f"# Generated by experiments/generate_configs.py - {name}\n# Run from repo/chi-bench\n"
        (OUT / f"{name}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"{len(list(OUT.glob('*.yaml')))} configs in {OUT}")


if __name__ == "__main__":
    main()

