# Compliance notes

| Requirement (competition rules) | How this repository meets it |
|---|---|
| Handbook is gated, non-commercial and non-redistributable | No Handbook file, excerpt or derived passage is stored here. Retrieval runs at execution time on the locally mounted copy. Trajectories, which may quote it, are kept out of the repository. |
| No credentials in submissions, notebooks, logs or repositories | Credentials come only from environment variables. `fiable_agent/config.py` rejects YAML profiles with secret-like keys or values. `.gitignore` excludes `.env`. |
| Autonomous evaluated trials | The guard is automatic code; no human input is possible once a trial starts. |
| No access to hidden expectations, solutions or verifier-only files | The rules use only information visible to the agent (its own tool calls and their results). The system prompt forbids reading simulator source, tests or expectation files. |
| General workflow logic only | Rules encode role-level workflow contracts (e.g. "a finalizing MD decision needs a rationale"), never task-specific answers. |
| Open-source code under an OSI license that allows commercial use | Apache-2.0 for this repository; χ-Bench and Harbor are Apache-2.0. |
| Disclosure of external models, tools and data | Listed in the README (Credits) and in the technical report. |
