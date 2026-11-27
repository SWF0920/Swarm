"""Matplotlib vector PDF figure generator for the Two-Tier Oversight Game experiments.

Generates data plots from exact analytical formulas and CSV outputs in data/,
using the unified three-role semantic palette:
  - Workers (W, colW):              #1964BA (Cobalt Blue family)
  - Frontier Orchestrator (F, colF):#9D174D (Royal Crimson-Plum)
  - Human Overseer (H, colH):       #0F765E (Forest Emerald-Teal)

By default writes generated plots to `plots_reproduced/` so existing standalone
publication PDFs in the root folder remain untouched unless `--overwrite-root`
is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_exact import (
    SwarmParams,
    eval_alarm_fatigue,
    eval_cap_noisy_priority,
    eval_policy2a,
    eval_policy2b,
    eval_priority_cost_control,
    eval_symmetric_roster_cost,
)
from train_ipg import run_exact_symmetric_ipg

# Canonical Three-Role Semantic Palette
COL_W = "#1964BA"     # Workers (W)
COL_F = "#9D174D"     # Frontier Orchestrator (F)
COL_H = "#0F765E"     # Human Overseer (H)
COL_WARN = "#CC520E"  # Policy 2a unreviewed worker harm absorbed

PALETTE_Q = {
    0.0: "#6BA1E6",
    0.1: "#1964BA",
    0.2: "#14498C",
    0.4: "#0D2F5E",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_crowding_vs_n(out_dir: Path, p: SwarmParams) -> None:
    n_grid = np.arange(2, 65)
    n_ipg = [2, 5, 10, 20, 40]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.8), sharex=True)

    for col, mode in enumerate(["cost", "cap"]):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        for q_w in (0.0, 0.1, 0.2, 0.4):
            if mode == "cost":
                res = [eval_symmetric_roster_cost(int(n), int(n), q_W=q_w, params=p) for n in n_grid]
            else:
                res = [eval_cap_noisy_priority(int(n), q_W=q_w, eps=1.0, params=p) for n in n_grid]
            ax_top.plot(n_grid, [r["harm"] for r in res], color=PALETTE_Q[q_w], lw=1.6, label=f"Policy 1 ($q_W={q_w}$)")
            ax_bot.plot(n_grid, [r["f_ovs"] for r in res], color=PALETTE_Q[q_w], lw=1.6)

        p2a = [eval_policy2a(int(n), p, model="cost") for n in n_grid]
        p2b = [eval_policy2b(int(n), p, model="cost") for n in n_grid]
        ax_top.plot(n_grid, [r["harm"] for r in p2a], color=COL_WARN, lw=1.8, label="Policy 2a: Orchestrator-Triage (harm absorbed)")
        ax_top.plot(n_grid, [r["harm"] for r in p2b], color=COL_F, lw=1.8, label="Policy 2b: Orchestrator-Triage ($d_F=0.7$ filter)")
        ax_bot.axhline(1.0, color=COL_WARN, lw=1.6)

        if mode == "cost":
            ipg_pts = [run_exact_symmetric_ipg(N=n, p_H0=0.5, q_W0=0.4, n_iters=20_000, params=p) for n in n_ipg]
            ax_top.scatter(n_ipg, [r["harm"] for r in ipg_pts], marker="d", color="#16202A", zorder=5, label="Policy 3: learned symmetric IPG")
            ax_bot.scatter(n_ipg, [r["f_ovs"] for r in ipg_pts], marker="d", color="#16202A", zorder=5)
            ax_top.axvline(41, color=COL_WARN, ls="--", lw=0.9)
            ax_top.set_title("Convex inbox congestion ($c_0=0.5, \\gamma=1.8$)", fontweight="bold")
        else:
            ax_top.axvline(13, color=COL_WARN, ls="--", lw=0.9)
            ax_top.set_title("Hard review cap ($K=2$, random triage)", fontweight="bold")

        ax_bot.set_xscale("log")
        ax_bot.set_xticks([2, 5, 10, 20, 40, 60])
        ax_bot.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_bot.set_xlabel("Swarm size $N$ (parallel workers, log scale)")
        ax_bot.set_ylim(-0.03, 1.05)

    axes[0, 0].set_ylabel("Total expected harm per episode")
    axes[1, 0].set_ylabel("$\\mathbb{P}(\\mathrm{human\\ oversees} \\mid A_F=1)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "fig_crowding_vs_N.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scaling_laws(out_dir: Path, p: SwarmParams) -> None:
    eps_grid = [0.01, 0.02, 0.05, 0.10, 0.25, 1.0]
    lam_grid = [0.01, 0.02, 0.05, 0.10, 0.20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)

    for q_w in (0.0, 0.1, 0.2):
        n_eps = []
        for eps in eps_grid:
            n_star = next(
                N for N in range(1, 5000)
                if eval_policy2a(N, p, model="cap")["harm"] < eval_cap_noisy_priority(N, q_W=q_w, eps=eps, params=p)["harm"] - 1e-12
            )
            n_eps.append(n_star)
        ax1.plot(eps_grid, n_eps, marker="o", ms=4, color=PALETTE_Q[q_w], lw=1.7, label=f"$q_W={q_w}$")

        n_lam = []
        for lam in lam_grid:
            n_star = next(
                N for N in range(1, 5000)
                if eval_policy2a(N, p, model="fatigue")["harm"] < eval_alarm_fatigue(N, q_W=q_w, lam=lam, params=p)["harm"] - 1e-12
            )
            n_lam.append(n_star)
        ax2.plot(lam_grid, n_lam, marker="o", ms=4, color=PALETTE_Q[q_w], lw=1.7, label=f"$q_W={q_w}$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axhline(100, color="gray", ls=":", lw=0.9)
    ax1.set_title("A. Noisy priority triage ($K=2$): $N^* \\sim x^*/(r_W\\varepsilon)$", fontweight="bold")
    ax1.set_xlabel("Priority misclassification probability $\\varepsilon$ (log scale)")
    ax1.set_ylabel("Crossover swarm size $N^*$ (Policy 2a beats Policy 1)")

    ax2.set_xscale("log")
    ax2.axhline(100, color="gray", ls=":", lw=0.9)
    ax2.set_title("B. Alarm-fatigue decay (flat cost): $N^* \\sim \\log(1/\\lambda)/(r_W\\lambda)$", fontweight="bold")
    ax2.set_xlabel("Per-ping accuracy decay rate $\\lambda$ (log scale)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "fig_scaling_laws.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_optimal_roster(out_dir: Path, p: SwarmParams) -> None:
    j_grid = np.arange(0, 81)
    p2a_val = eval_policy2a(80, p)["harm"]
    h_cap = [eval_cap_noisy_priority(80, j=int(j), q_W=0.0, eps=0.25, params=p)["harm"] for j in j_grid]
    h_fat = [eval_alarm_fatigue(80, j=int(j), q_W=0.0, lam=0.10, params=p)["harm"] for j in j_grid]

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.axhline(p2a_val, color="#475569", ls="--", lw=1.3, label=f"Policy 2a: workers silent ($j=0$, harm {p2a_val:.2f})")
    ax.plot(j_grid, h_cap, color=COL_H, lw=2.0, label="Noisy priority triage ($K=2, \\varepsilon=0.25$)")
    ax.plot(j_grid, h_fat, color=COL_WARN, lw=2.0, label="Alarm-fatigue accuracy decay ($\\lambda=0.10$)")

    j_cap = int(np.argmin(h_cap))
    j_fat = int(np.argmin(h_fat))
    ax.scatter([j_cap], [h_cap[j_cap]], color=COL_H, zorder=5)
    ax.scatter([j_fat], [h_fat[j_fat]], color=COL_WARN, zorder=5)
    ax.set_title("Fixed-size rotating roster vs. Policy 1 ($j=80$) and Policy 2a ($j=0$)", fontweight="bold")
    ax.set_xlabel("Permitted worker count $j \\in \\{0,\\dots,80\\}$ per episode ($N=80, q_W=0$)")
    ax.set_ylabel("Exact expected total harm per episode")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_optimal_roster.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_review_volume_and_controls(out_dir: Path, p: SwarmParams) -> None:
    n_grid = np.arange(1, 61)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for q_w in (0.0, 0.1, 0.2, 0.4):
        res = [eval_symmetric_roster_cost(int(n), int(n), q_W=q_w, params=p) for n in n_grid]
        ax1.plot(n_grid, [r["reviews"] for r in res], color=PALETTE_Q[q_w], lw=1.6, label=f"Policy 1 ($q_W={q_w}$)")
        ax2.plot(n_grid, [r["cost"] for r in res], color=PALETTE_Q[q_w], lw=1.6)
    p2a = eval_policy2a(10, p, model="cost")
    ax1.axhline(p2a["reviews"], color=COL_WARN, ls="--", lw=1.4, label="Policy 2a: workers silent")
    ax2.axhline(p2a["cost"], color=COL_WARN, ls="--", lw=1.4)
    ax1.set_title("A. Expected human reviews performed per episode", fontweight="bold")
    ax2.set_title("B. Expected review cost per episode", fontweight="bold")
    ax1.set_xlabel("Swarm size $N$ (parallel workers)")
    ax2.set_xlabel("Swarm size $N$ (parallel workers)")
    ax1.set_ylabel("Human reviews per episode")
    ax2.set_ylabel("Total review cost $\\mathbb{E}[c(M)\\cdot\\text{reviews}]$")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_dir / "fig_review_volume.pdf", bbox_inches="tight")
    plt.close(fig)

    # Noiseless priority controls
    n_c = np.arange(2, 65)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for q_w in (0.0, 0.1, 0.2, 0.4):
        r_cap = [eval_cap_noisy_priority(int(n), q_W=q_w, eps=0.0, params=p)["harm"] for n in n_c]
        r_cost = [eval_priority_cost_control(int(n), q_W=q_w, params=p)["harm"] for n in n_c]
        ax1.plot(n_c, r_cap, color=PALETTE_Q[q_w], lw=1.5)
        ax2.plot(n_c, r_cost, color=PALETTE_Q[q_w], lw=1.5)
    for ax in (ax1, ax2):
        ax.plot(n_c, [eval_policy2a(int(n), p)["harm"] for n in n_c], color=COL_WARN, lw=1.7)
        ax.plot(n_c, [eval_policy2b(int(n), p)["harm"] for n in n_c], color=COL_F, lw=1.7)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40, 60])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Swarm size $N$ (parallel workers, log scale)")
    ax1.set_title("A. Hard cap $K=2$, orchestrator-first triage (control)", fontweight="bold")
    ax2.set_title("B. Per-review cost $c_0 j^\\gamma$, priority order (control)", fontweight="bold")
    ax1.set_ylabel("Total expected harm per episode")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_controls_priority.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vector PDF figures from exact model.")
    parser.add_argument(
        "--overwrite-root",
        action="store_true",
        help="Write generated plots directly into the root directory instead of plots_reproduced/.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root if args.overwrite_root else (root / "plots_reproduced")
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    p = SwarmParams()
    plot_crowding_vs_n(out_dir, p)
    plot_scaling_laws(out_dir, p)
    plot_optimal_roster(out_dir, p)
    plot_review_volume_and_controls(out_dir, p)
    print(f"Saved vector PDF plots to {out_dir}")


if __name__ == "__main__":
    main()
