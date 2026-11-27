"""Trust-trap bisection (exact symmetric IPG) and sampled 40-player IPG sweep across initial human oversight p_{H,0}."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model_exact import SwarmParams
from train_ipg import run_exact_symmetric_ipg, run_sampled_nplayer_ipg


def bisect_exact_trust_trap(params: SwarmParams = SwarmParams()) -> float:
    """Finds the exact symmetric IPG trust-trap threshold p_{H,0}^* at N=40, q_{W,0}=0.4."""
    lo, hi = 0.0010, 0.0050
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        res = run_exact_symmetric_ipg(N=40, p_H0=mid, q_W0=0.40, n_iters=80_000, params=params)
        if res["harm"] > 12.20:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact bisection and sampled N=40 trust-trap sweep.")
    parser.add_argument("--quick", action="store_true", help="Run a fast verification pass instead of full 30x30k sweep.")
    args = parser.parse_args()

    p = SwarmParams()
    p_crit = bisect_exact_trust_trap(p)
    print(f"[Exact Symmetric IPG] Bisected trust-trap boundary at N=40: p_{{H,0}}^* approx {p_crit:.4f}")

    if args.quick:
        res = run_sampled_nplayer_ipg(
            N=40, p_H0=0.50, q_W0=0.40, n_iters=3_000, batch_size=1_024, lr=0.50, seed=1,
            condition_on_frontier_cell=True, params=p
        )
        print(
            f"[Quick Sampled N=40 Check] active_askers={res['n_active_h']}, "
            f"harm={res['harm']:.2f}, loss={res['loss']:.2f}"
        )
        return

    out_path = Path(__file__).resolve().parent.parent / "data" / "trust_trap_sweep_N40.csv"
    rows = []
    for p_h0 in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50):
        exact_res = run_exact_symmetric_ipg(N=40, p_H0=p_h0, q_W0=0.40, n_iters=60_000, params=p)
        for seed in range(5):
            samp = run_sampled_nplayer_ipg(
                N=40,
                p_H0=p_h0,
                q_W0=0.40,
                n_iters=30_000,
                batch_size=1_024,
                lr=0.50,
                step_clip=0.25,
                jitter=0.05,
                seed=100 * int(p_h0 * 1000) + seed,
                condition_on_frontier_cell=True,
                params=p,
            )
            rows.append(
                {
                    "p_H0": p_h0,
                    "seed": seed,
                    "exact_harm": round(exact_res["harm"], 4),
                    "sampled_harm": round(float(samp["harm"]), 4),
                    "sampled_loss": round(float(samp["loss"]), 4),
                    "sampled_pings": round(float(samp["pings"]), 4),
                    "sampled_cost": round(float(samp["cost"]), 4),
                    "n_active_h": samp["n_active_h"],
                    "n_uncond": samp["n_uncond"],
                }
            )
            print(f"p_H0={p_h0:.3f} seed={seed}: harm={samp['harm']:.2f}, active={samp['n_active_h']}, uncond={samp['n_uncond']}")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
