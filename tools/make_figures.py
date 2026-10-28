"""Generate the animated figures used in the README (assets/*.gif).

Run from the repository root:  python tools/make_figures.py
Every number drawn here is either read from this repository's code
(contract rules) or computed analytically and labelled as such.
No benchmark result is drawn by this script.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
PANEL = "#f1f0ec"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#a3a29c"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, BAD = "#1baf7a", "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def box(ax, x, y, w, h, text, color, active=False, fs=9.5, sub=None):
    face = color if active else PANEL
    edge = color
    txt = "white" if active else INK
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                       linewidth=2.2 if active else 1.2, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), text, ha="center", va="center",
            fontsize=fs, color=txt, weight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.035, sub, ha="center", va="center", fontsize=7.5,
                color=("white" if active else INK2), zorder=3)


def arrow(ax, a, b, active=False, color=INK2, style="-|>", conn="arc3"):
    ax.annotate("", xy=b, xytext=a, zorder=1,
                arrowprops=dict(arrowstyle=style, lw=2.4 if active else 1.2, connectionstyle=conn,
                                color=color if active else MUTED, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- 1. architecture
def architecture() -> None:
    fig = plt.figure(figsize=(10, 5.4), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])

    nodes = {
        "task": (0.03, 0.70, 0.15, 0.13, "Task instruction", "role, case, goal"),
        "role": (0.24, 0.70, 0.15, 0.13, "Role planner", "provider | payer | CM"),
        "llm": (0.45, 0.70, 0.17, 0.13, "LLM (ReAct)", "tool-use loop"),
        "rag": (0.45, 0.43, 0.17, 0.13, "Handbook RAG", "BM25, role sub-book"),
        "guard": (0.68, 0.70, 0.17, 0.13, "Action guard", "checklist + critics"),
        "sim": (0.68, 0.43, 0.17, 0.13, "χ-World via MCP", "official simulator"),
        "gate": (0.68, 0.16, 0.17, 0.13, "Closure gate", "terminal-state rules"),
        "done": (0.45, 0.16, 0.17, 0.13, "Terminal state", "letter / plan / bundle"),
        "ledger": (0.20, 0.43, 0.19, 0.13, "Cost & call ledger", "tie-break metrics"),
    }
    colors = {"task": BLUE, "role": BLUE, "llm": BLUE, "rag": AQUA, "guard": ORANGE,
              "sim": AQUA, "gate": ORANGE, "done": GOOD, "ledger": YELLOW}
    edges = [("task", "role"), ("role", "llm"), ("rag", "llm"), ("llm", "guard"),
             ("guard", "sim"), ("sim", "llm"), ("guard", "gate"), ("gate", "done"),
             ("llm", "ledger")]

    def center(k, side):
        x, y, w, h = nodes[k][:4]
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h),
                "b": (x + w / 2, y)}[side]

    anchors = {("task", "role"): ("r", "l"), ("role", "llm"): ("r", "l"), ("rag", "llm"): ("t", "b"),
               ("llm", "guard"): ("r", "l"), ("guard", "sim"): ("b", "t"), ("sim", "llm"): ("l", "b"),
               ("guard", "gate"): ("r", "r"), ("gate", "done"): ("l", "r"), ("llm", "ledger"): ("b", "r")}
    conns = {("guard", "gate"): "arc,angleA=0,angleB=0,armA=40,armB=40,rad=10",
             ("llm", "ledger"): "angle,angleA=-90,angleB=0,rad=8", ("sim", "llm"): "arc3,rad=0"}

    steps = [
        ({"task"}, set(), "1 · The task instruction names the case and the seat the agent works in."),
        ({"role"}, {("task", "role")}, "2 · The role planner injects an ordered, role-specific workflow plan."),
        ({"llm", "rag"}, {("role", "llm"), ("rag", "llm")}, "3 · The LLM grounds itself in the Managed-Care Handbook (lazy, role-scoped retrieval)."),
        ({"llm", "guard"}, {("llm", "guard")}, "4 · Every mutating or irreversible tool call is intercepted by the action guard."),
        ({"guard"}, set(), "5 · Contract checklist (deterministic) + 3-judge critic panel vote ALLOW or BLOCK."),
        ({"guard", "sim"}, {("guard", "sim")}, "6 · Allowed calls reach the official χ-World simulator through MCP."),
        ({"sim", "llm"}, {("sim", "llm")}, "7 · Observations flow back; blocked calls return a reason the agent must fix."),
        ({"gate"}, {("guard", "gate")}, "8 · Before finishing, the closure gate re-checks every terminal-state obligation."),
        ({"done", "ledger"}, {("gate", "done"), ("llm", "ledger")}, "9 · Terminal state reached; tokens, cost and tool calls are logged for tie-breaks."),
    ]
    per = 9
    frames = len(steps) * per + 6

    def draw(i):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        k = min(i // per, len(steps) - 1)
        act_nodes, act_edges, caption = steps[k]
        ax.text(0.03, 0.93, "fiable-agent · reliability by construction", fontsize=15, weight="bold")
        ax.text(0.03, 0.885, "Agent architecture inside the official χ-Bench harness", fontsize=10, color=INK2)
        for (a, b) in edges:
            sa, sb = anchors[(a, b)]
            pa, pb = center(a, sa), center(b, sb)
            if (a, b) == ("llm", "ledger"):
                pa = (pa[0] - 0.05, pa[1])
            if (a, b) in (("rag", "llm"), ("sim", "llm")):
                pb = (pb[0] + (0.03 if a == "sim" else -0.03), pb[1])
            arrow(ax, pa, pb, active=(a, b) in act_edges, color=colors[b], conn=conns.get((a, b), "arc3"))
        for key, (x, y, w, h, t, s) in nodes.items():
            box(ax, x, y, w, h, t, colors[key], active=key in act_nodes, sub=s)
        ax.add_patch(FancyBboxPatch((0.03, 0.025), 0.94, 0.08, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor="white", edgecolor=MUTED, lw=1))
        ax.text(0.05, 0.065, caption, fontsize=10.5, va="center", color=INK)

    anim = FuncAnimation(fig, draw, frames=frames, interval=110)
    anim.save(OUT / "architecture.gif", writer=PillowWriter(fps=6))
    plt.close(fig)


# ---------------------------------------------------------------- 2. guarded action
def guarded_action() -> None:
    fig = plt.figure(figsize=(10, 5.0), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    attempt1 = ['review.submit_md_case_decision(', '  case_id = <target case>,', '  decision = "deny",',
                '  signed_off = true,', '  rationale = "",            ', ')']
    attempt2 = ['review.submit_md_case_decision(', '  case_id = <target case>,', '  decision = "deny",',
                '  signed_off = true,', '  rationale = "<criterion + patient facts',
                '               + handbook section id>"', ')']
    checks = [("GLOBAL-01", "mutates only the target case"),
              ("UM-02", "MD decision signed off with rationale"),
              ("UM-02", "denial carries a denial rationale")]
    judges = ["contract", "grounding", "policy"]
    # timeline: 0-11 show attempt1, 12-29 checks, 30-41 judges, 42-53 BLOCK, 54-65 attempt2, 66-95 pass
    frames = 112

    def draw(i):
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.03, 0.92, "Guarded irreversible action (payer UM review)", fontsize=15, weight="bold")
        ax.text(0.03, 0.875, "Illustrative walk-through of rule UM-02 as implemented in fiable_agent/contracts.py",
                fontsize=9.5, color=INK2)
        second = i >= 60
        code = attempt2 if second else attempt1
        ax.add_patch(FancyBboxPatch((0.03, 0.30), 0.42, 0.50, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="#1a1a19", edgecolor="#1a1a19"))
        ax.text(0.05, 0.765, "proposed tool call · attempt %d" % (2 if second else 1), color="#c3c2b7", fontsize=9)
        for j, line in enumerate(code):
            c = "#f0efec"
            if "rationale" in line and not second:
                c = "#e66767"
            if second and ("<criterion" in line or "handbook" in line):
                c = "#4cc79a"
            ax.text(0.05, 0.71 - j * 0.055, line, family="DejaVu Sans Mono", fontsize=9.5, color=c)
        # checklist
        ax.text(0.50, 0.80, "Contract checklist (deterministic)", fontsize=11, weight="bold")
        local = i - (66 if second else 12)
        for j, (rid, desc) in enumerate(checks):
            y = 0.72 - j * 0.075
            shown = local >= j * 5
            ok = second or j != 2 and j != 1
            mark, col = ("✓", GOOD) if ok else ("✗", BAD)
            ax.text(0.50, y, rid, fontsize=9.5, weight="bold", color=INK if shown else MUTED)
            ax.text(0.60, y, desc, fontsize=9.5, color=INK2 if shown else MUTED)
            if shown:
                ax.text(0.955, y, mark, fontsize=13, color=col, weight="bold", ha="center")
        ax.text(0.50, 0.46, "Critic panel (majority vote)", fontsize=11, weight="bold")
        jl = i - (78 if second else 30)
        for j, name in enumerate(judges):
            shown = jl >= j * 4
            ok = second or name == "grounding"
            col = (GOOD if ok else BAD) if shown else MUTED
            box(ax, 0.505 + j * 0.155, 0.32, 0.125, 0.09, name, col, active=shown, fs=9,
                sub=("allow" if ok else "block") if shown else "…")
        # verdict
        if (not second and i >= 42) or (second and i >= 92):
            verdict = "ALLOW → sent to χ-World" if second else "BLOCK → reason returned to the agent"
            col = GOOD if second else BAD
            ax.add_patch(FancyBboxPatch((0.50, 0.16), 0.455, 0.09, boxstyle="round,pad=0.008,rounding_size=0.02",
                                        facecolor=col, edgecolor=col))
            ax.text(0.7275, 0.205, verdict, color="white", weight="bold", fontsize=11, ha="center", va="center")
        if not second and i >= 48:
            ax.text(0.03, 0.20, "Reason: \"UM-02 – a finalizing deny needs a denial rationale that\n"
                    "names the criterion and the patient-specific facts.\"", fontsize=9.5, color=INK2)
        if second:
            ax.text(0.03, 0.20, "The agent repairs the payload instead of repeating the blocked call.\n"
                    "Bounded: after N blocks the action is released, so no task times out.", fontsize=9.5,
                    color=INK2)
        ax.text(0.03, 0.05, "Each rule is traced to the official verifier's check IDs "
                "(e.g. md.signed_off, md.denial_rationale_present).", fontsize=9, color=INK2)

    anim = FuncAnimation(fig, draw, frames=frames, interval=110)
    anim.save(OUT / "guarded_action.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


# ---------------------------------------------------------------- 3. reliability math
def reliability_math() -> None:
    """pass^3 for a fixed mean pass@1 as per-task consistency changes (analytic)."""
    import numpy as np

    mean = 0.30
    n_tasks = 30
    fig = plt.figure(figsize=(10, 5.0), dpi=96)
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, bottom=0.17, top=0.78, wspace=0.28)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    lams = np.concatenate([np.linspace(0, 1, 40), np.ones(12)])

    def per_task(lam):
        # mix between "flaky" (every task p=mean) and "consistent" (a mean-fraction of tasks at p=1, rest 0)
        k = int(round(mean * n_tasks))
        consistent = np.array([1.0] * k + [0.0] * (n_tasks - k))
        return (1 - lam) * mean + lam * consistent

    def draw(i):
        lam = lams[i]
        p = per_task(lam)
        for a in (ax1, ax2):
            a.clear()
            a.spines[["top", "right"]].set_visible(False)
        fig.texts.clear()
        fig.text(0.07, 0.92, "Why consistency matters: same pass@1, very different pass³", fontsize=15, weight="bold")
        fig.text(0.07, 0.865, "Analytic illustration (not a measured result): 30 tasks, mean success 0.30, "
                 "three independent attempts", fontsize=9.5, color=INK2)
        ax1.bar(range(n_tasks), p, color=BLUE, width=0.8)
        ax1.set_ylim(0, 1.05); ax1.set_xlabel("task"); ax1.set_ylabel("per-task success probability p")
        ax1.set_xticks([])
        ax1.set_title("consistency λ = %.2f" % lam, fontsize=10, color=INK2, loc="left")
        pass1 = p.mean(); pass3 = (p ** 3).mean()
        bars = ax2.bar(["pass@1", "pass³"], [pass1, pass3], color=[BLUE, ORANGE], width=0.5)
        for b, v in zip(bars, [pass1, pass3]):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.3f" % v, ha="center", fontsize=11, weight="bold")
        ax2.set_ylim(0, 0.4); ax2.set_ylabel("score")
        ax2.set_title("pass³ = mean(p³)", fontsize=10, color=INK2, loc="left")
        fig.text(0.07, 0.05, "Removing avoidable, random failures (contract slips, premature finishes) "
                 "moves an agent to the right: pass³ rises toward pass@1.", fontsize=9.5, color=INK)

    anim = FuncAnimation(fig, draw, frames=len(lams), interval=120)
    anim.save(OUT / "reliability_pass3.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


# ---------------------------------------------------------------- 4. rule coverage (real counts)
def rule_coverage() -> None:
    sys.path.insert(0, str(ROOT))
    from fiable_agent.contracts import ALL_RULES

    dom = {"GLOBAL": "Global", "PROV": "Provider PA", "UM": "Payer UM", "CM": "Care Mgmt"}
    n_rules = {v: 0 for v in dom.values()}
    n_checks = {v: set() for v in dom.values()}
    for rule in ALL_RULES:
        d = dom[rule.rule_id.split("-")[0]]
        n_rules[d] += 1
        n_checks[d].update(rule.protects)
    names = list(dom.values())
    r = [n_rules[k] for k in names]
    c = [len(n_checks[k]) for k in names]
    frames = 30
    fig = plt.figure(figsize=(10, 4.6), dpi=96)
    ax = fig.add_axes([0.08, 0.16, 0.88, 0.62])

    def draw(i):
        t = min(1.0, (i + 1) / 20)
        ax.clear(); fig.texts.clear()
        ax.spines[["top", "right"]].set_visible(False)
        x = range(len(names))
        ax.bar([k - 0.19 for k in x], [v * t for v in r], width=0.36, color=BLUE, label="contract rules")
        ax.bar([k + 0.19 for k in x], [v * t for v in c], width=0.36, color=AQUA, label="verifier check IDs traced")
        if t >= 1:
            for k in x:
                ax.text(k - 0.19, r[k] + 0.3, str(r[k]), ha="center", fontsize=10, weight="bold")
                ax.text(k + 0.19, c[k] + 0.3, str(c[k]), ha="center", fontsize=10, weight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(names)
        ax.set_ylim(0, max(c) + 3); ax.set_ylabel("count")
        ax.legend(frameon=False, loc="upper left")
        fig.text(0.08, 0.91, "Contract checklist coverage per domain", fontsize=15, weight="bold")
        fig.text(0.08, 0.855, "Counted directly from fiable_agent/contracts.py (%d rules, %d distinct verifier "
                 "check IDs)" % (sum(r), sum(c)), fontsize=9.5, color=INK2)

    anim = FuncAnimation(fig, draw, frames=frames, interval=100)
    anim.save(OUT / "rule_coverage.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["architecture", "guarded_action", "reliability_math", "rule_coverage"]
    for name in which:
        globals()[name]()
        print("wrote", name)
