"""Generates CSV datasets and verifies all quantitative numbers reported in the manuscript."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from model_exact import (
    SwarmParams,
    eval_alarm_fatigue,
    eval_cap_noisy_priority,
    eval_policy2a,
    eval_policy2b,
    eval_priority_cost_control,
    eval_symmetric_roster_cost,
    optimal_symmetric_throttling_cost,
)
from train_ipg import run_exact_symmetric_ipg


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p = SwarmParams()

    # 1. Full threshold scan N in [1, 150] (Appendix C Table 3 & Figures 3, 7, 9)
    scan_rows = []
    for N in range(1, 151):
        p2a_cost = eval_policy2a(N, p, model="cost")
        p2b_cost = eval_policy2b(N, p, model="cost")
        for q_W in (0.0, 0.1, 0.2, 0.4):
            c_res = eval_symmetric_roster_cost(N=N, j=N, q_W=q_W, params=p)
            cap_rand = eval_cap_noisy_priority(N=N, q_W=q_W, eps=1.0, params=p)
            cap_prio = eval_cap_noisy_priority(N=N, q_W=q_W, eps=0.0, params=p)
            cost_prio = eval_priority_cost_control(N=N, q_W=q_W, params=p)
            scan_rows.append(
                {
                    "N": N,
                    "q_W": q_W,
                    "p2a_harm": round(p2a_cost["harm"], 6),
                    "p2a_loss": round(p2a_cost["loss"], 6),
                    "p2b_harm": round(p2b_cost["harm"], 6),
                    "cost_p1_harm": round(c_res["harm"], 6),
                    "cost_p1_cost": round(c_res["cost"], 6),
                    "cost_p1_loss": round(c_res["loss"], 6),
                    "cost_p1_reviews": round(c_res["reviews"], 6),
                    "cost_p1_f_ovs": round(c_res["f_ovs"], 6),
                    "cap_rand_p1_harm": round(cap_rand["harm"], 6),
                    "cap_rand_p1_f_ovs": round(cap_rand["f_ovs"], 6),
                    "cap_prio_p1_harm": round(cap_prio["harm"], 6),
                    "cost_prio_p1_harm": round(cost_prio["harm"], 6),
                }
            )

    with (data_dir / "baseline_scan_N1_150.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)

    # Verify Table 3 crossovers
    print("=== Table 3: Critical Swarm-Size Thresholds ===")
    for model_key, label in [("cost_p1_harm", "Convex cost"), ("cap_rand_p1_harm", "Hard cap K=2 random")]:
        crossovers = {}
        for q_W in (0.0, 0.1, 0.2, 0.4):
            sub = [r for r in scan_rows if r["q_W"] == q_W]
            n_star = next((r["N"] for r in sub if r["p2a_harm"] < r[model_key] - 1e-12), None)
            crossovers[q_W] = n_star
        print(f"  {label:22s}: N* = {crossovers}")

    # 2. Noisy priority scaling laws (Figure 4A)
    eps_grid = [0.01, 0.02, 0.05, 0.10, 0.25, 1.00]
    noisy_rows = []
    for q_W in (0.0, 0.1, 0.2):
        for eps in eps_grid:
            n_star = None
            for N in range(1, 5000):
                h_p1 = eval_cap_noisy_priority(N=N, q_W=q_W, eps=eps, params=p)["harm"]
                h_p2a = eval_policy2a(N, p, model="cap")["harm"]
                if h_p2a < h_p1 - 1e-12:
                    n_star = N
                    break
            noisy_rows.append({"q_W": q_W, "eps": eps, "N_star": n_star})
    with (data_dir / "scaling_laws_noisy_priority.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["q_W", "eps", "N_star"])
        writer.writeheader()
        writer.writerows(noisy_rows)

    # 3. Alarm-fatigue scaling laws (Figure 4B)
    lam_grid = [0.01, 0.02, 0.05, 0.10, 0.20]
    fat_rows = []
    for q_W in (0.0, 0.1, 0.2):
        for lam in lam_grid:
            n_star = None
            for N in range(1, 5000):
                h_p1 = eval_alarm_fatigue(N=N, q_W=q_W, lam=lam, params=p)["harm"]
                h_p2a = eval_policy2a(N, p, model="fatigue")["harm"]
                if h_p2a < h_p1 - 1e-12:
                    n_star = N
                    break
            fat_rows.append({"q_W": q_W, "lambda": lam, "N_star": n_star})
    with (data_dir / "scaling_laws_alarm_fatigue.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["q_W", "lambda", "N_star"])
        writer.writeheader()
        writer.writerows(fat_rows)

    # 4. Fixed-size rotating roster at N = 80 (Figure 5)
    roster_rows = []
    p2a_80 = eval_policy2a(80, p)["harm"]
    for j in range(81):
        h_cap = eval_cap_noisy_priority(N=80, j=j, q_W=0.0, eps=0.25, params=p)["harm"]
        h_fat = eval_alarm_fatigue(N=80, j=j, q_W=0.0, lam=0.10, params=p)["harm"]
        roster_rows.append(
            {
                "j": j,
                "p2a_harm": round(p2a_80, 6),
                "noisy_priority_harm": round(h_cap, 6),
                "alarm_fatigue_harm": round(h_fat, 6),
            }
        )
    with (data_dir / "optimal_roster_N80.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(roster_rows[0].keys()))
        writer.writeheader()
        writer.writerows(roster_rows)

    best_cap = min(roster_rows, key=lambda r: r["noisy_priority_harm"])
    best_fat = min(roster_rows, key=lambda r: r["alarm_fatigue_harm"])
    print(
        f"=== Figure 5 Roster Optima (N=80, q_W=0) ===\n"
        f"  Noisy priority (K=2, eps=0.25): j*={best_cap['j']} (harm {best_cap['noisy_priority_harm']:.2f} vs {roster_rows[-1]['noisy_priority_harm']:.2f})\n"
        f"  Alarm fatigue (lam=0.10):       j*={best_fat['j']} (harm {best_fat['alarm_fatigue_harm']:.2f} vs {roster_rows[-1]['alarm_fatigue_harm']:.2f})"
    )

    # 5. Cost-bottleneck roster j in {0..40} at N = 40, h_W = 1, q_W = 0 & gap vs optimal symmetric throttling
    cost_roster_rows = []
    for j in range(41):
        r40 = eval_symmetric_roster_cost(N=40, j=j, q_W=0.0, h_W=1.0, params=p)
        cost_roster_rows.append(
            {
                "j": j,
                "harm": round(r40["harm"], 6),
                "cost": round(r40["cost"], 6),
                "loss": round(r40["loss"], 6),
                "pings": round(r40["pings"], 6),
                "f_ovs": round(r40["f_ovs"], 6),
            }
        )
    with (data_dir / "cost_roster_vs_symmetric_N40.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost_roster_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cost_roster_rows)

    # 6. Table 1 / Figure 6 baseline + symmetric optimum + exact IPG + roster optimum comparison
    t1_rows = []
    print("=== Table 1 & Appendix B Roster vs. Symmetric Comparison ===")
    for N in (5, 10, 20, 40):
        p1 = eval_symmetric_roster_cost(N=N, j=N, q_W=0.4, params=p)
        p1_q0 = eval_symmetric_roster_cost(N=N, j=N, q_W=0.0, params=p)
        p2a = eval_policy2a(N=N, params=p, model="cost")
        sym = optimal_symmetric_throttling_cost(N=N, params=p)
        ipg = run_exact_symmetric_ipg(N=N, p_H0=0.5, q_W0=0.4, n_iters=40_000, params=p)
        best_roster = min(
            (
                {**eval_symmetric_roster_cost(N=N, j=j, q_W=0.0, h_W=1.0, params=p), "j_star": j}
                for j in range(N + 1)
            ),
            key=lambda d: d["loss"],
        )
        t1_rows.append(
            {
                "N": N,
                "p1_q04_harm": round(p1["harm"], 4),
                "p1_q04_loss": round(p1["loss"], 4),
                "p1_q00_harm": round(p1_q0["harm"], 4),
                "p1_q00_loss": round(p1_q0["loss"], 4),
                "p2a_harm": round(p2a["harm"], 4),
                "p2a_loss": round(p2a["loss"], 4),
                "sym_opt_h": round(sym["h_sym"], 4),
                "sym_opt_harm": round(sym["harm"], 4),
                "sym_opt_loss": round(sym["loss"], 4),
                "exact_ipg_h": round(ipg["h_learned"], 4),
                "exact_ipg_harm": round(ipg["harm"], 4),
                "exact_ipg_loss": round(ipg["loss"], 4),
                "roster_j_star": best_roster["j_star"],
                "roster_harm": round(best_roster["harm"], 4),
                "roster_loss": round(best_roster["loss"], 4),
            }
        )
        print(
            f"  N={N:2d} | P1(q=0.4)=({p1['harm']:.2f},{p1['loss']:.2f}) | P2a=({p2a['harm']:.2f},{p2a['loss']:.2f}) "
            f"| SymOpt=({sym['harm']:.3f},{sym['loss']:.3f}) | ExactIPG=({ipg['harm']:.3f},{ipg['loss']:.3f}) "
            f"| Roster(j*={best_roster['j_star']})=({best_roster['harm']:.3f},{best_roster['loss']:.3f})"
        )

    with (data_dir / "specialization_comparison_table1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t1_rows)


if __name__ == "__main__":
    main()
