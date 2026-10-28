# Evaluation protocol (pre-registered, 2026-10-28)

This protocol was fixed before any benchmark run. Results are added to `results/` without changing it; any
deviation is recorded in `results/DEVIATIONS.md` with its reason.

## Question

Does each reliability component improve the official χ-Bench score of the strongest reference agent, measured by the
official verifier, at equal model and equal per-task budget?

## Arms (cumulative)

| Arm | Harness | `FIABLE_CC_ARM` | Adds |
|---|---|---|---|
| C0 official (control) | `claude-code` (official) | – | nothing: reference agent as shipped |
| C1 plan | `fiable-cc` | `plan` | operating principles + role plan (system-prompt addendum) |
| C2 checklist | `fiable-cc` | `checklist` | contract-checklist hooks + closure gate |
| C3 critic | `fiable-cc` | `critic` | three-judge critic panel on terminal calls |

All arms: model `anthropic/claude-sonnet-5`, `timeout_multiplier: 2.0` (as in the official submission template),
`max_retries: 2` (infrastructure only), `n_attempts: 3`, concurrency 2, χ-Bench runtime `7998d113`, data
`chi-bench-v1.0.0`.

## Tasks

Development subsets are chosen deterministically from public task metadata (`experiments/generate_configs.py`,
`DEV`), stratified by workflow stage:

- `abl1_*`: 1 task per domain (minimum run);
- `abl3_*`: 3 tasks per domain (main run).

Domains: provider prior authorization (`pa`) and payer utilization management (`um`). Care management is added only
if an API key for the official patient simulator is available; otherwise it is reported as not evaluated.

## Metrics

From the official verifier's binary reward (`verifier/scorecard.json`), aggregated with the official
`scripts/aggregate.py`:

- primary: pass@1 (first attempt) and mean reward over the three attempts, with standard deviation across attempts;
- reliability: pass³ (all three attempts pass);
- tie-breakers: mean agent cost per trial (official price table) and mean tool calls;
- diagnostics: failed verifier checks per trial, guard blocks and forced releases (`/logs/agent/fiable/guard_log.jsonl`).

## Decision rule

A component is **accepted** when its arm's mean reward over the three attempts is at least the previous arm's and
its pass³ is not lower; ties are broken by cost, then tool calls. The frozen submission is the accepted arm with the
best mean reward. With the small development subsets, differences below one task are reported as inconclusive and
the simpler arm is preferred.

## Integrity

- Seeds are not controllable in the hosted models; variability is captured by three independent attempts.
- No task-specific rules, hard-coded answers, verifier-only files or hidden expectations are used.
- Negative results are reported with the same detail as positive ones.
