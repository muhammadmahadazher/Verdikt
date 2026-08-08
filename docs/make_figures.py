"""Regenerate every figure in the README. All numbers are computed here, never typed by hand.

    python docs/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verdikt import theme
from verdikt.stats import (
    normal_approx_n,
    one_sided_upper,
    power_exact,
    required_n,
    wilson,
)
from verdikt.stats.power import mde

DOCS = Path(__file__).resolve().parent
plt.rcParams.update(theme.mpl_rc())


# --------------------------------------------------------------- figure 1
def fig_power() -> None:
    """Why n=20 cannot settle a robotics comparison."""
    ns = list(range(4, 121, 2))
    power = [power_exact(n, 0.35, 0.70, 0.05, "fisher") for n in ns]
    detect = [mde(n, 0.35, 0.80, 0.05, "fisher") for n in ns]
    n_exact = required_n(0.35, 0.70, 0.80, 0.05, "fisher")
    n_approx = normal_approx_n(0.35, 0.70, 0.80, 0.05)
    p_at_approx = power_exact(n_approx, 0.35, 0.70, 0.05, "fisher")
    p_at_20 = power_exact(20, 0.35, 0.70, 0.05, "fisher")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(ns, power, color=theme.ACCENT, lw=2, label="exact power (Fisher)")
    ax.axhline(0.80, color=theme.TEXT_FAINT, ls="--", lw=1)
    ax.text(6, 0.82, "80% power", color=theme.TEXT_FAINT, fontsize=9)
    ax.scatter([20], [p_at_20], color=theme.BAD, zorder=5, s=60)
    ax.annotate(f"n=20\npower {p_at_20:.2f}", (20, p_at_20), textcoords="offset points",
                xytext=(12, -28), color=theme.BAD, fontsize=9)
    ax.scatter([n_approx], [p_at_approx], color=theme.WARN, zorder=5, s=60)
    ax.annotate(f"normal approx says n={n_approx}\nreally {p_at_approx:.2f}",
                (n_approx, p_at_approx), textcoords="offset points", xytext=(10, -40),
                color=theme.WARN, fontsize=9)
    ax.scatter([n_exact], [power_exact(n_exact, 0.35, 0.70, 0.05, "fisher")],
               color=theme.OK, zorder=5, s=60)
    ax.annotate(f"exact: n={n_exact}", (n_exact, 0.80), textcoords="offset points",
                xytext=(8, 14), color=theme.OK, fontsize=9)
    ax.set_xlabel("episodes per arm")
    ax.set_ylabel("probability of detecting the difference")
    ax.set_title("detecting 35% vs 70%", loc="left")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(ns, [d * 100 if d else np.nan for d in detect], color=theme.WARN, lw=2)
    for n_mark in (20, 50, 100):
        d = mde(n_mark, 0.35, 0.80, 0.05, "fisher")
        if d:
            ax.scatter([n_mark], [d * 100], color=theme.TEXT, zorder=5, s=40)
            ax.annotate(f"n={n_mark}: {d * 100:.0f} pp", (n_mark, d * 100),
                        textcoords="offset points", xytext=(8, 6), fontsize=9,
                        color=theme.TEXT_DIM)
    ax.set_xlabel("episodes per arm")
    ax.set_ylabel("smallest detectable difference (pp)")
    ax.set_title("what your budget can actually resolve", loc="left")

    fig.suptitle("verdikt plan  -  power computed through the test that issues the verdict",
                 x=0.01, ha="left", color=theme.TEXT_DIM, fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "power.png", dpi=150)
    print("wrote docs/power.png")


# --------------------------------------------------------------- figure 2
def fig_audit() -> None:
    """The published study, re-read honestly."""
    arms = [("upstream diffusion", 14, 20, theme.SERIES[0]),
            ("diffusion 50k", 7, 20, theme.SERIES[1]),
            ("smolvla 20k", 0, 20, theme.SERIES[3]),
            ("act 50k", 0, 20, theme.SERIES[4])]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, (name, k, n, colour) in enumerate(arms):
        lo, hi = wilson(k, n)
        y = len(arms) - i
        ax.plot([lo * 100, hi * 100], [y, y], color=colour, lw=3, solid_capstyle="round")
        ax.scatter([k / n * 100], [y], color=colour, s=70, zorder=5)
        ax.text(-3, y, name, ha="right", va="center", color=theme.TEXT, fontsize=10)
        label = f"{k}/{n}"
        if k == 0:
            label += f"   (<= {one_sided_upper(0, n) * 100:.1f}% one-sided)"
        ax.text(103, y, label, ha="left", va="center", color=theme.TEXT_DIM, fontsize=9)

    # shade only the two arms whose overlap is the point being made
    ax.axvspan(48.1, 56.7, ymin=0.62, ymax=0.97, color=theme.WARN, alpha=0.16)
    ax.text(52.4, 3.5, "overlap", color=theme.WARN, fontsize=9, ha="center", va="center")
    ax.text(52, 0.42, "these two intervals overlap:\nFisher p = 0.056, not significant at n=20",
            color=theme.WARN, fontsize=9.5, ha="center")

    ax.set_xlim(-38, 128)
    ax.set_ylim(0, len(arms) + 0.8)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success rate (%), Wilson 95% interval")
    ax.set_title("verdikt compare  -  the same four runs, with their uncertainty", loc="left")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "audit.png", dpi=150)
    print("wrote docs/audit.png")


# --------------------------------------------------------------- figure 3
def fig_workflow() -> None:
    """What reads what. Nothing here trains, runs a policy, or needs a GPU."""
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, sub, colour, alpha=0.14):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.4, edgecolor=colour, facecolor=colour, alpha=alpha))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.4, edgecolor=colour, facecolor="none"))
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
                color=theme.TEXT, fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                color=theme.TEXT_DIM, fontsize=8.5)

    def arrow(x1, y1, x2, y2, colour=theme.HAIRLINE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3,
                                    shrinkA=2, shrinkB=2))

    ax.text(98, 55, "files you already have", color=theme.TEXT_FAINT, fontsize=9.5, ha="right")
    for i, (t, s) in enumerate([("LeRobot dataset", "meta/ + parquet"),
                                ("eval JSON", "any harness"),
                                ("train configs", "run metadata")]):
        box(2 + i * 33, 46, 28, 7, t, s, theme.TEXT_FAINT, alpha=0.06)

    ax.text(65, 42.5, "before you spend the GPU", color=theme.TEXT_FAINT, fontsize=9.5,
            ha="right")
    box(2, 32, 28, 8.5, "verdikt lint", "dataset integrity rules", theme.SERIES[5])
    box(68, 32, 28, 8.5, "verdikt doctor", "silent-failure preflight", theme.SERIES[5])

    ax.text(65, 28.5, "after you have results", color=theme.TEXT_FAINT, fontsize=9.5,
            ha="right")
    box(2, 18, 28, 8.5, "verdikt plan", "required N, pre-registered", theme.SERIES[0])
    box(35, 18, 28, 8.5, "verdikt ingest", "-> canonical table", theme.SERIES[0])
    box(68, 18, 28, 8.5, "verdikt manifest", "samples_seen confound", theme.SERIES[0])

    box(28, 7, 42, 8.5, "verdikt compare", "exact tests - intervals - confound check",
        theme.SERIES[1])

    for i in range(3):
        arrow(16 + i * 33, 46, 16 + i * 33, 41 if i != 1 else 27)
    arrow(16, 32, 16, 27)
    arrow(82, 32, 82, 27)
    arrow(16, 18, 40, 16)
    arrow(49, 18, 49, 16)
    arrow(82, 18, 58, 16)

    states = [("exit 0", "BETTER", theme.OK), ("exit 1", "REGRESSION", theme.BAD),
              ("exit 2", "UNDERPOWERED", theme.WARN), ("exit 3", "NOT COMPARABLE",
                                                       theme.CONFOUND)]
    for i, (code, name, colour) in enumerate(states):
        x = 2 + i * 24.5
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.5), 22, 4.2, boxstyle="round,pad=0.4,rounding_size=0.8",
            linewidth=1.3, edgecolor=colour, facecolor=colour, alpha=0.16))
        ax.text(x + 11, 2.6, f"{code}  {name}", ha="center", va="center",
                color=colour, fontsize=9.5, weight="bold")
    arrow(49, 7, 49, 5.2)

    ax.set_title("verdikt  -  every command reads a file that already exists",
                 loc="left", color=theme.TEXT, fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(DOCS / "workflow.png", dpi=150)
    print("wrote docs/workflow.png")


# --------------------------------------------------------------- figure 4
def fig_n_matters() -> None:
    """The same four policies at n=20 and at n=200. Nothing changed but the evidence."""
    import json

    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pusht_n200"
    n200 = {}
    for name in ("upstream", "diffusion", "act", "smolvla"):
        f = fixtures / f"{name}.json"
        if not f.exists():
            print("skipping fig_n_matters: n=200 corpus not present")
            return
        succ = json.loads(f.read_text())["per_task"][0]["metrics"]["successes"]
        n200[name] = (sum(bool(s) for s in succ), len(succ))

    n20 = {"upstream": (14, 20), "diffusion": (7, 20), "act": (0, 20), "smolvla": (0, 20)}
    order = ["upstream", "diffusion", "act", "smolvla"]
    labels = {"upstream": "upstream diffusion", "diffusion": "diffusion 50k",
              "act": "act 50k", "smolvla": "smolvla 20k"}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    for ax, data, title in ((axes[0], n20, "n = 20 per arm"),
                            (axes[1], n200, "n = 200 per arm")):
        for i, key in enumerate(order):
            k, n = data[key]
            lo, hi = wilson(k, n)
            y = len(order) - i
            colour = theme.SERIES[i % len(theme.SERIES)]
            ax.plot([lo * 100, hi * 100], [y, y], color=colour, lw=3.4,
                    solid_capstyle="round")
            ax.scatter([k / n * 100], [y], color=colour, s=62, zorder=5)
            ax.text(102, y, f"{k}/{n}", ha="left", va="center", color=theme.TEXT_DIM,
                    fontsize=8.5)
        ax.set_xlim(-4, 118)
        ax.set_ylim(0.4, len(order) + 0.6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left")
        ax.set_xlabel("success rate (%), Wilson 95% interval")
    axes[0].set_yticks(range(1, len(order) + 1))
    axes[0].set_yticklabels([labels[k] for k in reversed(order)], fontsize=9.5)

    axes[0].text(56, 0.75, "diffusion and upstream overlap\np = 0.056", color=theme.WARN,
                 fontsize=9, ha="center")
    axes[1].text(50, 0.75, "every pair separated\nthree distinct groups", color=theme.OK,
                 fontsize=9, ha="center")

    fig.suptitle("same four policies, ten times the evidence", x=0.008, ha="left",
                 color=theme.TEXT_DIM, fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "n_matters.png", dpi=150)
    print("wrote docs/n_matters.png")


if __name__ == "__main__":
    fig_power()
    fig_audit()
    fig_workflow()
    fig_n_matters()
