"""Policy 3 Independent Policy Gradient (IPG) learners: exact symmetric IPG and sampled N-player REINFORCE."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit, logit
from scipy.stats import binom

from model_exact import SwarmParams, eval_heterogeneous_workers_cost


def exact_symmetric_loss_and_grad(
    N: int,
    theta_h: float,
    theta_q: float,
    psi_F: np.ndarray,
    psi_W: np.ndarray,
    params: SwarmParams = SwarmParams(),
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Computes exact expected team loss E[Harm + Cost] and exact analytical gradients."""
    h = float(expit(theta_h))
    q = float(expit(theta_q))
    piF = expit(psi_F)
    piW = expit(psi_W)

    a = params.r_F
    rw = params.p_W * h + (1.0 - params.p_W) * q
    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F

    w_vals = np.arange(N + 1)
    pmf_N = binom.pmf(w_vals, N, rw)
    k_vals = np.arange(N)
    pmf_Nm1 = binom.pmf(k_vals, N - 1, rw)

    costs_m = np.array([params.cost_per_review(m) for m in range(N + 2)], dtype=np.float64)

    # Expected Frontier oversight and worker oversight
    f_ovs = float(np.sum(pmf_N * piF[w_vals + 1]))
    w_ovs_0 = float(np.sum(pmf_Nm1 * piW[k_vals + 1]))
    w_ovs_1 = float(np.sum(pmf_Nm1 * piW[k_vals + 2]))
    w_ovs = (1.0 - a) * w_ovs_0 + a * w_ovs_1

    harm = unreviewed_harm - params.p_F * params.h_F * params.L_F * f_ovs - N * params.p_W * h * params.L_W * w_ovs

    # Expected review cost:
    # A_F = 0, w >= 1: w * piW[w] * c(w)
    # A_F = 1, w >= 0: (piF[w+1] + w * piW[w+1]) * c(w+1)
    term_cost_0 = w_vals * piW[w_vals] * costs_m[w_vals]
    term_cost_0[0] = 0.0
    term_cost_1 = (piF[w_vals + 1] + w_vals * piW[w_vals + 1]) * costs_m[w_vals + 1]
    cost = float((1.0 - a) * np.sum(pmf_N * term_cost_0) + a * np.sum(pmf_N * term_cost_1))
    loss = harm + cost

    # Gradients w.r.t. psi_F and psi_W
    g_psiF = np.zeros_like(psi_F, dtype=np.float64)
    g_psiW = np.zeros_like(psi_W, dtype=np.float64)

    # For m = w + 1 (m in 1..N+1):
    # dLoss / dpiF[m] = pmf_N[m-1] * (a * c(m) - p_F * h_F * L_F * 1 ?)
    # Wait: p_F * h_F = a * B_F / L_F, so - p_F * h_F * L_F + a * c(m) = a * (c(m) - B_F)
    m_idx = np.arange(1, N + 2)
    dL_dpiF = pmf_N * (a * costs_m[m_idx] - params.p_F * params.h_F * params.L_F)
    g_psiF[m_idx] = dL_dpiF * piF[m_idx] * (1.0 - piF[m_idx])

    # Worker oversee probability piW[m] for m = 1..N+1:
    # Harm term: - N * p_W * h * L_W * [ (1-a)*pmf_Nm1[m-1] (if m<=N) + a*pmf_Nm1[m-2] (if m>=2) ]
    # Cost term: (1-a)*pmf_N[m]*m*c(m) (if m<=N) + a*pmf_N[m-1]*(m-1)*c(m) (if m>=2)
    for m in range(1, N + 2):
        dh_dpiW = 0.0
        dc_dpiW = 0.0
        if m <= N:
            dh_dpiW -= N * params.p_W * h * params.L_W * (1.0 - a) * pmf_Nm1[m - 1]
            dc_dpiW += (1.0 - a) * pmf_N[m] * m * costs_m[m]
        if m >= 2:
            dh_dpiW -= N * params.p_W * h * params.L_W * a * pmf_Nm1[m - 2]
            dc_dpiW += a * pmf_N[m - 1] * (m - 1) * costs_m[m]
        g_psiW[m] = (dh_dpiW + dc_dpiW) * piW[m] * (1.0 - piW[m])

    # Derivative of Binomial(n, p) expectation E[g(X)] w.r.t. p equals n * E[g(Y+1) - g(Y)] for Y ~ Binom(n-1, p)
    g_N_total = (
        -params.p_F * params.h_F * params.L_F * piF[w_vals + 1]
        + (1.0 - a) * term_cost_0
        + a * term_cost_1
    )
    d_gN_drw = float(N * np.sum(pmf_Nm1 * (g_N_total[1:] - g_N_total[:-1])))

    g_Nm1_wovs = (1.0 - a) * piW[k_vals + 1] + a * piW[k_vals + 2]
    if N >= 2:
        pmf_Nm2 = binom.pmf(np.arange(N - 1), N - 2, rw)
        d_wovs_drw = float((N - 1) * np.sum(pmf_Nm2 * (g_Nm1_wovs[1:] - g_Nm1_wovs[:-1])))
    else:
        d_wovs_drw = 0.0

    dL_drw = d_gN_drw - N * params.p_W * h * params.L_W * d_wovs_drw
    dL_dh = -N * params.p_W * params.L_W * w_ovs + dL_drw * params.p_W
    dL_dq = dL_drw * (1.0 - params.p_W)

    g_h = float(dL_dh * h * (1.0 - h))
    g_q = float(dL_dq * q * (1.0 - q))
    return float(loss), g_h, g_q, g_psiF, g_psiW


def run_exact_symmetric_ipg(
    N: int,
    p_H0: float = 0.50,
    h_W0: float = 0.90,
    q_W0: float = 0.40,
    n_iters: int = 40_000,
    lr: float = 0.50,
    step_clip: float = 1.0,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, float]:
    """Runs deterministic exact-gradient symmetric IPG under the cost bottleneck."""
    theta_h = float(logit(np.clip(h_W0, 1e-6, 1.0 - 1e-6)))
    theta_q = float(logit(np.clip(q_W0, 1e-6, 1.0 - 1e-6)))
    psi_init = float(logit(np.clip(p_H0, 1e-6, 1.0 - 1e-6)))
    psi_F = np.full(N + 2, psi_init, dtype=np.float64)
    psi_W = np.full(N + 2, psi_init, dtype=np.float64)

    for _ in range(n_iters):
        _, g_h, g_q, g_psiF, g_psiW = exact_symmetric_loss_and_grad(
            N, theta_h, theta_q, psi_F, psi_W, params=params
        )
        # Note: in N-player symmetric IPG, each individual worker sees 1/N of the symmetric worker gradient
        step_h = np.clip(-lr * (g_h / N), -step_clip, step_clip)
        step_q = np.clip(-lr * (g_q / N), -step_clip, step_clip)
        step_F = np.clip(-lr * g_psiF, -step_clip, step_clip)
        step_W = np.clip(-lr * g_psiW, -step_clip, step_clip)

        theta_h += float(step_h)
        theta_q += float(step_q)
        psi_F += step_F
        psi_W += step_W

    h_final = float(expit(theta_h))
    q_final = float(expit(theta_q))
    piF_final = expit(psi_F)
    piW_final = expit(psi_W)
    metrics = eval_heterogeneous_workers_cost(
        [h_final] * N,
        [q_final] * N,
        params=params,
        pi_H_F=piF_final,
        pi_H_W=piW_final,
    )
    metrics["h_learned"] = h_final
    metrics["q_learned"] = q_final
    return metrics


def run_sampled_nplayer_ipg(
    N: int,
    p_H0: float = 0.50,
    h_W0: float = 0.90,
    q_W0: float = 0.40,
    n_iters: int = 6_000,
    batch_size: int = 2_048,
    lr: float = 0.30,
    step_clip: float = 0.25,
    jitter: float = 0.05,
    seed: int = 0,
    condition_on_frontier_cell: bool = False,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, object]:
    """Runs decentralized N-player sampled REINFORCE IPG with independent per-worker logits."""
    rng = np.random.default_rng(seed)
    theta_h = logit(np.clip(h_W0, 1e-5, 1.0 - 1e-5)) + rng.normal(0.0, jitter, size=N)
    theta_q = logit(np.clip(q_W0, 1e-5, 1.0 - 1e-5)) + rng.normal(0.0, jitter, size=N)
    psi_init = float(logit(np.clip(p_H0, 1e-5, 1.0 - 1e-5)))
    psi_F = np.full(N + 2, psi_init, dtype=np.float64)
    psi_W = np.full(N + 2, psi_init, dtype=np.float64)

    costs_m = np.array([params.cost_per_review(m) for m in range(N + 2)], dtype=np.float64)
    ema_cell = np.zeros(4, dtype=np.float64)  # conditioned on (f_risky, f_ask) in {0,1}^2
    ema_init = False

    for _ in range(n_iters):
        h_vec = expit(theta_h)
        q_vec = expit(theta_q)
        piF = expit(psi_F)
        piW = expit(psi_W)

        f_risky = rng.random(batch_size) < params.p_F
        f_ask = np.where(f_risky, rng.random(batch_size) < params.h_F, rng.random(batch_size) < params.q_F)

        w_risky = rng.random((batch_size, N)) < params.p_W
        p_ask_w = np.where(w_risky, h_vec[None, :], q_vec[None, :])
        w_ask = rng.random((batch_size, N)) < p_ask_w

        m_tot = f_ask.astype(np.int32) + w_ask.sum(axis=1)
        f_ovs = f_ask & (rng.random(batch_size) < piF[m_tot])
        w_ovs = w_ask & (rng.random((batch_size, N)) < piW[m_tot][:, None])

        ep_harm = (
            np.where(f_risky & ~f_overseen_mask(f_ovs), params.L_F, 0.0)
            + np.where(w_risky & ~w_ovs, params.L_W, 0.0).sum(axis=1)
        )
        ep_reviews = f_ovs.astype(np.int32) + w_ovs.sum(axis=1)
        ep_cost = ep_reviews * costs_m[m_tot]
        reward = -(ep_harm + ep_cost)

        cell_id = f_risky.astype(np.int32) * 2 + f_ask.astype(np.int32)
        if not ema_init:
            for cid in range(4):
                mask = cell_id == cid
                if mask.any():
                    ema_cell[cid] = float(reward[mask].mean())
            ema_init = True
        else:
            for cid in range(4):
                mask = cell_id == cid
                if mask.any():
                    ema_cell[cid] = 0.9 * ema_cell[cid] + 0.1 * float(reward[mask].mean())

        adv = (reward - ema_cell[cell_id]) if condition_on_frontier_cell else (reward - reward.mean())

        # Worker REINFORCE score functions
        score_h = np.where(w_risky, w_ask.astype(np.float64) - h_vec[None, :], 0.0)
        score_q = np.where(~w_risky, w_ask.astype(np.float64) - q_vec[None, :], 0.0)
        grad_h = (adv[:, None] * score_h).mean(axis=0)
        grad_q = (adv[:, None] * score_q).mean(axis=0)

        theta_h += np.clip(lr * grad_h, -step_clip, step_clip)
        theta_q += np.clip(lr * grad_q, -step_clip, step_clip)

        # Human REINFORCE score functions per inbox size m
        for m in range(1, N + 2):
            mask_m = m_tot == m
            if not mask_m.any():
                continue
            mask_f = mask_m & f_ask
            if mask_f.any():
                gf = float((adv[mask_f] * (f_ovs[mask_f].astype(np.float64) - piF[m])).mean()) * (
                    mask_f.sum() / batch_size
                )
                psi_F[m] += float(np.clip(lr * gf, -step_clip, step_clip))
            w_ask_m = w_ask[mask_m]
            if w_ask_m.any():
                w_score_m = np.where(w_ask_m, w_ovs[mask_m].astype(np.float64) - piW[m], 0.0).sum(axis=1)
                gw = float((adv[mask_m] * w_score_m).mean()) * (mask_m.sum() / batch_size)
                psi_W[m] += float(np.clip(lr * gw, -step_clip, step_clip))

    h_final = expit(theta_h)
    q_final = expit(theta_q)
    exact_eval = eval_heterogeneous_workers_cost(
        h_final,
        q_final,
        params=params,
        pi_H_F=expit(psi_F),
        pi_H_W=expit(psi_W),
    )
    exact_eval["h_vec"] = h_final.tolist()
    exact_eval["q_vec"] = q_final.tolist()
    exact_eval["n_active_h"] = int((h_final > 0.5).sum())
    exact_eval["n_uncond"] = int(((h_final > 0.5) & (q_final > 0.5)).sum())
    return exact_eval


def f_overseen_mask(f_ovs: np.ndarray) -> np.ndarray:
    return f_ovs


if __name__ == "__main__":
    p = SwarmParams()
    for N in (5, 10, 20, 40):
        res = run_exact_symmetric_ipg(N=N, p_H0=0.50, q_W0=0.40, n_iters=40_000, params=p)
        print(
            f"[Exact Symmetric IPG] N={N:2d}: h={res['h_learned']:.3f}, q={res['q_learned']:.4f}, "
            f"pings={res['pings']:.2f}, F_ovs={res['f_ovs']:.3f}, harm={res['harm']:.2f}, loss={res['loss']:.2f}"
        )
