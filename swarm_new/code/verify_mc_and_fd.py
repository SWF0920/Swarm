"""Monte Carlo verification and finite-difference gradient check for the Swarm Oversight model."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from model_exact import (
    SwarmParams,
    eval_alarm_fatigue,
    eval_cap_noisy_priority,
    eval_priority_cost_control,
    eval_symmetric_roster_cost,
)
from train_ipg import exact_symmetric_loss_and_grad


def mc_simulate_cell(
    N: int,
    q_W: float,
    model: str,
    n_episodes: int = 200_000,
    seed: int = 42,
    eps: float = 1.0,
    lam: float = 0.10,
    params: SwarmParams = SwarmParams(),
) -> Tuple[float, float]:
    """Simulates episodes via Monte Carlo and returns (sample_mean_harm, std_error)."""
    rng = np.random.default_rng(seed)
    f_risky = rng.random(n_episodes) < params.p_F
    f_ask = np.where(f_risky, rng.random(n_episodes) < params.h_F, rng.random(n_episodes) < params.q_F)

    w_risky = rng.random((n_episodes, N)) < params.p_W
    w_ask = np.where(w_risky, rng.random((n_episodes, N)) < params.h_W, rng.random((n_episodes, N)) < q_W)

    w_pings = w_ask.sum(axis=1)
    m_total = f_ask.astype(np.int32) + w_pings

    if model == "cost":
        bw = params.B_W(params.h_W, q_W)
        m_W = params.cutoff_m(bw)
        m_F = params.cutoff_m(params.B_F)
        f_overseen = f_ask & (m_total <= m_F)
        w_overseen = w_ask & (m_total[:, None] <= m_W)
        harm_f = np.where(f_risky & ~f_overseen, params.L_F, 0.0)
        harm_w = np.where(w_risky & ~w_overseen, params.L_W, 0.0).sum(axis=1)
        ep_harm = harm_f + harm_w
    elif model == "cap_noisy":
        # High priority indicator for each worker that asked
        w_high = w_ask & (rng.random((n_episodes, N)) < eps)
        t_high = w_high.sum(axis=1)
        ep_harm = np.zeros(n_episodes, dtype=np.float64)
        for ep in range(n_episodes):
            # Fast vectorized-per-episode slot allocation when K is small
            pass
        # Vectorized slot selection across episodes using random tie-breaking keys
        f_prio = np.where(f_ask, 2.0 + rng.random(n_episodes), -1.0)
        w_prio = np.where(w_ask, np.where(w_high, 2.0, 1.0) + rng.random((n_episodes, N)), -1.0)
        all_prio = np.column_stack([f_prio, w_prio])
        top_k_idx = np.argpartition(-all_prio, kth=min(params.K, N), axis=1)[:, : params.K]
        reviewed = np.zeros_like(all_prio, dtype=bool)
        rows = np.arange(n_episodes)[:, None]
        valid = all_prio[rows, top_k_idx] > 0.0
        for slot in range(params.K):
            col = top_k_idx[:, slot]
            mask = valid[:, slot]
            reviewed[mask, col[mask]] = True
        f_overseen = reviewed[:, 0]
        w_overseen = reviewed[:, 1:]
        harm_f = np.where(f_risky & ~f_overseen, params.L_F, 0.0)
        harm_w = np.where(w_risky & ~w_overseen, params.L_W, 0.0).sum(axis=1)
        ep_harm = harm_f + harm_w
    elif model == "fatigue":
        acc = np.where(m_total >= 1, np.exp(-lam * (m_total - 1)), 1.0)
        f_neutralized = f_ask & (rng.random(n_episodes) < acc)
        w_neutralized = w_ask & (rng.random((n_episodes, N)) < acc[:, None])
        harm_f = np.where(f_risky & ~f_neutralized, params.L_F, 0.0)
        harm_w = np.where(w_risky & ~w_neutralized, params.L_W, 0.0).sum(axis=1)
        ep_harm = harm_f + harm_w
    elif model == "priority_cost_control":
        bw = params.B_W(params.h_W, q_W)
        m_W = params.cutoff_m(bw)
        f_overseen = f_ask  # always slot 1
        # Workers randomly ordered in slots (1 + f_ask) .. (w_pings + f_ask)
        w_ranks = np.argsort(np.argsort(np.where(w_ask, rng.random((n_episodes, N)), 2.0), axis=1), axis=1) + 1
        w_slots = w_ranks + f_ask[:, None].astype(np.int32)
        w_overseen = w_ask & (w_slots <= m_W)
        harm_f = np.where(f_risky & ~f_overseen, params.L_F, 0.0)
        harm_w = np.where(w_risky & ~w_overseen, params.L_W, 0.0).sum(axis=1)
        ep_harm = harm_f + harm_w
    else:
        raise ValueError(f"Unknown model: {model}")

    mean = float(ep_harm.mean())
    se = float(ep_harm.std(ddof=1) / math.sqrt(n_episodes))
    return mean, se


def verify_finite_difference_gradients(params: SwarmParams = SwarmParams()) -> float:
    """Checks exact analytical gradients of Policy 3 against central finite differences."""
    rng = np.random.default_rng(12345)
    max_err = 0.0
    eps_fd = 1e-6
    for N in (2, 5, 10, 20):
        theta_h = float(rng.normal(0.2, 0.8))
        theta_q = float(rng.normal(-0.5, 0.8))
        psi_F = rng.normal(0.5, 0.7, size=N + 2)
        psi_W = rng.normal(-0.2, 0.7, size=N + 2)

        _, g_h, g_q, g_psiF, g_psiW = exact_symmetric_loss_and_grad(
            N, theta_h, theta_q, psi_F, psi_W, params=params
        )

        # Check theta_h and theta_q
        lp, *_ = exact_symmetric_loss_and_grad(N, theta_h + eps_fd, theta_q, psi_F, psi_W, params=params)
        lm, *_ = exact_symmetric_loss_and_grad(N, theta_h - eps_fd, theta_q, psi_F, psi_W, params=params)
        max_err = max(max_err, abs(g_h - (lp - lm) / (2.0 * eps_fd)))

        lp, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q + eps_fd, psi_F, psi_W, params=params)
        lm, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q - eps_fd, psi_F, psi_W, params=params)
        max_err = max(max_err, abs(g_q - (lp - lm) / (2.0 * eps_fd)))

        for m in range(1, N + 2):
            p_plus = psi_F.copy()
            p_minus = psi_F.copy()
            p_plus[m] += eps_fd
            p_minus[m] -= eps_fd
            lp, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q, p_plus, psi_W, params=params)
            lm, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q, p_minus, psi_W, params=params)
            max_err = max(max_err, abs(g_psiF[m] - (lp - lm) / (2.0 * eps_fd)))

            w_plus = psi_W.copy()
            w_minus = psi_W.copy()
            w_plus[m] += eps_fd
            w_minus[m] -= eps_fd
            lp, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q, psi_F, w_plus, params=params)
            lm, *_ = exact_symmetric_loss_and_grad(N, theta_h, theta_q, psi_F, w_minus, params=params)
            max_err = max(max_err, abs(g_psiW[m] - (lp - lm) / (2.0 * eps_fd)))

    return float(max_err)


if __name__ == "__main__":
    p = SwarmParams()
    max_fd = verify_finite_difference_gradients(p)
    print(f"[FD Check] Maximum exact-vs-central-FD gradient error: {max_fd:.3e}")

    max_z = 0.0
    checked = 0
    for N in (2, 5, 10, 20, 40):
        for q_W in (0.0, 0.2):
            exact_cost = eval_symmetric_roster_cost(N=N, j=N, q_W=q_W, params=p)["harm"]
            mc_mean, mc_se = mc_simulate_cell(N=N, q_W=q_W, model="cost", n_episodes=100_000, seed=100 + N + int(q_W * 10), params=p)
            z = abs(mc_mean - exact_cost) / mc_se if mc_se > 0 else 0.0
            max_z = max(max_z, z)
            checked += 1

            exact_cap = eval_cap_noisy_priority(N=N, q_W=q_W, eps=1.0, params=p)["harm"]
            mc_mean_c, mc_se_c = mc_simulate_cell(N=N, q_W=q_W, model="cap_noisy", eps=1.0, n_episodes=100_000, seed=200 + N + int(q_W * 10), params=p)
            zc = abs(mc_mean_c - exact_cap) / mc_se_c if mc_se_c > 0 else 0.0
            max_z = max(max_z, zc)
            checked += 1

    print(f"[MC Check] Verified {checked} representative cells (100k episodes/cell); max |z| = {max_z:.2f}")
