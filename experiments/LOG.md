# Experiment log

Protocol: [docs/EVALUATION_PROTOCOL.md](../docs/EVALUATION_PROTOCOL.md). Machine-readable copy: [log.csv](log.csv).
Negative results are recorded exactly like positive ones.

## T0 — guard integration on Linux (2026-10-28)

- **Hypothesis:** the `fiable-cc` hooks install into the official chi-bench checkout and behave as designed
  (block with exit 2 and a reason, record evidence, stay consistent under concurrent hook calls).
- **Design:** clean install on the pinned checkout (`7998d113`), `cb agent list`, unit tests, hook calls piped
  exactly as Claude Code invokes them, 8 concurrent PostToolUse writers.
- **Control:** stock checkout without the installer (no `fiable-cc` registered).
- **Result:** `fiable-cc` registered; 71/71 tests pass; a non-compliant MD decision is blocked with UM-02 and
  grounding reasons; 10/10 events recorded under concurrency.
- **Decision:** accepted.

## C0–C3 — pre-registered ablation

Pending: requires Handbook access (gated) to build the official runtime image.
