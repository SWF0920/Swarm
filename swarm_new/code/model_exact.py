"""Exact analytical evaluation of the Two-Tier Oversight Game.

Implements exact Binomial and Poisson-binomial expectations across all human
attention models (convex inbox congestion, hard review cap with noisy priority,
flat-cost alarm-fatigue decay, and noiseless priority controls) and deferral
policies (Policy 1 Direct-Ask, Policy 2a/2b Orchestrator-Triage, Fixed-Size
Rotating Roster, and symmetric/heterogeneous worker throttling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class SwarmParams:
    """Baseline calibration parameters for the two-tier Oversight Game."""

    p_W: float = 0.15
    L_W: float = 2.0
    p_F: float = 0.25
    L_F: float = 50.0
    h_W: float = 0.90
    h_F: float = 0.95
    q_F: float = 0.05
    c_0: float = 0.50
    gamma: float = 1.80
    K: int = 2
    d_F: float = 0.70

    @property
    def r_F(self) -> float:
        return self.p_F * self.h_F + (1.0 - self.p_F) * self.q_F

    @property
    def B_F(self) -> float:
        return (self.p_F * self.h_F / self.r_F) * self.L_F

    def r_W(self, h_W: Optional[float] = None, q_W: float = 0.0) -> float:
        hw = self.h_W if h_W is None else h_W
        return self.p_W * hw + (1.0 - self.p_W) * q_W

    def B_W(self, h_W: Optional[float] = None, q_W: float = 0.0) -> float:
        hw = self.h_W if h_W is None else h_W
        rw = self.r_W(hw, q_W)
        if rw <= 0.0:
            return 0.0
        return (self.p_W * hw / rw) * self.L_W

    def cost_per_review(self, m: int) -> float:
        if m <= 0:
            return 0.0
        return self.c_0 * (m ** self.gamma)

    def cutoff_m(self, benefit: float) -> int:
        if benefit < self.c_0:
            return 0
        return int(math.floor((benefit / self.c_0) ** (1.0 / self.gamma)))


def poisson_binomial_pmf(probs: Sequence[float]) -> np.ndarray:
    """Exact PMF of a sum of independent non-identical Bernoulli variables."""
    pmf = np.array([1.0], dtype=np.float64)
    for p in probs:
        nxt = np.zeros(len(pmf) + 1, dtype=np.float64)
        nxt[:-1] += pmf * (1.0 - p)
        nxt[1:] += pmf * p
        pmf = nxt
    return pmf


def eval_policy2a(N: int, params: SwarmParams = SwarmParams(), model: str = "cost") -> Dict[str, float]:
    """Exact expectations for Policy 2a (workers silent, Frontier defers directly)."""
    a = params.r_F
    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F
    # Under Policy 2a, M = A_F in {0, 1}. Since B_F > c_0 and K >= 1, F is always overseen when A_F = 1.
    f_ovs = 1.0
    harm = unreviewed_harm - params.p_F * params.h_F * params.L_F
    reviews = a
    if model in ("cost", "priority_cost_control"):
        cost = a * params.c_0 * (1.0 ** params.gamma)
    elif model == "fatigue":
        cost = a * params.c_0
    else:
        cost = 0.0
    return {
        "harm": harm,
        "cost": cost,
        "loss": harm + cost,
        "pings": a,
        "reviews": reviews,
        "f_ovs": f_ovs,
    }


def eval_policy2b(N: int, params: SwarmParams = SwarmParams(), model: str = "cost") -> Dict[str, float]:
    """Exact expectations for Policy 2b (workers filtered by Frontier d_F)."""
    base = eval_policy2a(N, params, model=model)
    saved_worker_harm = N * params.p_W * params.d_F * params.L_W
    harm = base["harm"] - saved_worker_harm
    return {
        **base,
        "harm": harm,
        "loss": harm + base["cost"],
    }


def eval_symmetric_roster_cost(
    N: int,
    j: int,
    q_W: float = 0.0,
    h_W: Optional[float] = None,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, float]:
    """Exact evaluation under convex inbox congestion for j active symmetric workers."""
    hw = params.h_W if h_W is None else h_W
    rw = params.r_W(hw, q_W)
    bw = params.B_W(hw, q_W)
    bf = params.B_F
    a = params.r_F

    m_W = params.cutoff_m(bw)
    m_F = params.cutoff_m(bf)

    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F
    w_vals = np.arange(j + 1)
    pmf_w = binom.pmf(w_vals, j, rw)

    # Frontier coverage: A_F = 1 => M = w + 1 <= m_F iff w <= m_F - 1
    f_ovs = float(pmf_w[w_vals <= m_F - 1].sum()) if m_F >= 1 else 0.0

    # Worker coverage for each of the j permitted workers:
    if j == 0 or rw <= 0.0 or m_W <= 0:
        w_ovs = 0.0
    else:
        w_other = np.arange(j)
        pmf_other = binom.pmf(w_other, j - 1, rw)
        cov_af0 = float(pmf_other[w_other + 1 <= m_W].sum())
        cov_af1 = float(pmf_other[w_other + 2 <= m_W].sum())
        w_ovs = (1.0 - a) * cov_af0 + a * cov_af1

    harm = unreviewed_harm - params.p_F * params.h_F * params.L_F * f_ovs - j * params.p_W * hw * params.L_W * w_ovs

    # Expected reviews and convex review cost
    reviews = 0.0
    cost = 0.0
    for w, pw in zip(w_vals, pmf_w):
        if pw == 0.0:
            continue
        # Case A_F = 0: inbox size m = w
        if w >= 1:
            m = int(w)
            n_rev = w if m <= m_W else 0
            reviews += (1.0 - a) * pw * n_rev
            cost += (1.0 - a) * pw * n_rev * params.cost_per_review(m)
        # Case A_F = 1: inbox size m = w + 1
        m = int(w + 1)
        rev_f = 1 if m <= m_F else 0
        rev_w = int(w) if m <= m_W else 0
        n_rev = rev_f + rev_w
        reviews += a * pw * n_rev
        cost += a * pw * n_rev * params.cost_per_review(m)

    return {
        "harm": float(harm),
        "cost": float(cost),
        "loss": float(harm + cost),
        "pings": float(a + j * rw),
        "reviews": float(reviews),
        "f_ovs": float(f_ovs),
        "w_ovs": float(w_ovs),
    }


def eval_heterogeneous_workers_cost(
    h_vec: Sequence[float],
    q_vec: Sequence[float],
    params: SwarmParams = SwarmParams(),
    pi_H_F: Optional[Sequence[float]] = None,
    pi_H_W: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Exact evaluation for N heterogeneous workers under convex inbox congestion.

    If pi_H_F and pi_H_W (length N + 2 indexed by m = 0..N+1) are omitted,
    uses the Bayes-optimal human oversee policy against pooled worker posterior B_W.
    """
    N = len(h_vec)
    h_arr = np.asarray(h_vec, dtype=np.float64)
    q_arr = np.asarray(q_vec, dtype=np.float64)
    r_arr = params.p_W * h_arr + (1.0 - params.p_W) * q_arr
    a = params.r_F

    if pi_H_F is None or pi_H_W is None:
        sum_r = float(r_arr.sum())
        pooled_bw = (float((params.p_W * h_arr * params.L_W).sum()) / sum_r) if sum_r > 1e-15 else 0.0
        piF = np.array([1.0 if (m >= 1 and params.B_F >= params.cost_per_review(m)) else 0.0 for m in range(N + 2)])
        piW = np.array([1.0 if (m >= 1 and pooled_bw >= params.cost_per_review(m)) else 0.0 for m in range(N + 2)])
    else:
        piF = np.asarray(pi_H_F, dtype=np.float64)
        piW = np.asarray(pi_H_W, dtype=np.float64)

    pmf_all = poisson_binomial_pmf(r_arr)
    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F

    # Frontier harm averted
    w_idx = np.arange(N + 1)
    f_ovs = float(np.sum(pmf_all * piF[w_idx + 1]))
    harm_averted_F = params.p_F * params.h_F * params.L_F * f_ovs

    # Worker harm averted: exact leave-one-out Poisson-binomial for each worker i
    harm_averted_W = 0.0
    for i in range(N):
        if h_arr[i] <= 0.0:
            continue
        others = np.delete(r_arr, i)
        pmf_minus = poisson_binomial_pmf(others)
        k_idx = np.arange(N)
        cov_i = float(
            (1.0 - a) * np.sum(pmf_minus * piW[k_idx + 1])
            + a * np.sum(pmf_minus * piW[k_idx + 2])
        )
        harm_averted_W += params.p_W * h_arr[i] * params.L_W * cov_i

    # Expected reviews and cost
    reviews = 0.0
    cost = 0.0
    for w in range(N + 1):
        pw = float(pmf_all[w])
        if pw == 0.0:
            continue
        if w >= 1:
            n_rev0 = w * piW[w]
            reviews += (1.0 - a) * pw * n_rev0
            cost += (1.0 - a) * pw * n_rev0 * params.cost_per_review(w)
        m1 = w + 1
        n_rev1 = piF[m1] + w * piW[m1]
        reviews += a * pw * n_rev1
        cost += a * pw * n_rev1 * params.cost_per_review(m1)

    harm = unreviewed_harm - harm_averted_F - harm_averted_W
    return {
        "harm": float(harm),
        "cost": float(cost),
        "loss": float(harm + cost),
        "pings": float(a + r_arr.sum()),
        "reviews": float(reviews),
        "f_ovs": float(f_ovs),
    }


def eval_cap_noisy_priority(
    N: int,
    j: Optional[int] = None,
    q_W: float = 0.0,
    eps: float = 1.0,
    K: Optional[int] = None,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, float]:
    """Exact evaluation under hard review cap K with worker priority-misclassification eps."""
    active_j = N if j is None else j
    cap_k = params.K if K is None else K
    rw = params.r_W(params.h_W, q_W)
    bw = params.B_W(params.h_W, q_W)
    bf = params.B_F
    a = params.r_F

    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F

    # Expected total reviews V_N = E[min(K, A_F + W_j)]
    w_vals = np.arange(active_j + 1)
    pmf_w = binom.pmf(w_vals, active_j, rw)
    v_af0 = float(np.sum(pmf_w * np.minimum(cap_k, w_vals)))
    v_af1 = float(np.sum(pmf_w * np.minimum(cap_k, w_vals + 1)))
    v_total = (1.0 - a) * v_af0 + a * v_af1

    # Conditional Frontier coverage rho_F = E[min(1, K / (1 + T))], T ~ Binomial(j, r_W * eps)
    t_vals = np.arange(active_j + 1)
    pmf_t = binom.pmf(t_vals, active_j, rw * eps)
    f_ovs = float(np.sum(pmf_t * np.minimum(1.0, cap_k / (1.0 + t_vals))))

    # Proposition A.3 exact pricing identity
    harm = unreviewed_harm - bw * v_total - a * (bf - bw) * f_ovs
    return {
        "harm": float(harm),
        "cost": 0.0,
        "loss": float(harm),
        "pings": float(a + active_j * rw),
        "reviews": float(v_total),
        "f_ovs": float(f_ovs),
    }


def eval_alarm_fatigue(
    N: int,
    j: Optional[int] = None,
    q_W: float = 0.0,
    lam: float = 0.10,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, float]:
    """Exact evaluation under flat-cost review with exponential inbox clutter fatigue xi(m)=exp(-lam*(m-1))."""
    active_j = N if j is None else j
    rw = params.r_W(params.h_W, q_W)
    a = params.r_F
    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F

    eta = 1.0 - math.exp(-lam)
    d = 1.0 - rw * eta
    A_term = params.p_F * params.h_F * params.L_F
    C_term = params.p_W * params.h_W * params.L_W

    if active_j == 0:
        f_eff = 1.0
        averted = A_term
    else:
        f_eff = d ** active_j
        averted = A_term * (d ** active_j) + C_term * active_j * (1.0 - a * eta) * (d ** (active_j - 1))

    harm = unreviewed_harm - averted
    reviews = a + active_j * rw
    cost = reviews * params.c_0
    return {
        "harm": float(harm),
        "cost": float(cost),
        "loss": float(harm + cost),
        "pings": float(reviews),
        "reviews": float(reviews),
        "f_ovs": float(f_eff),
    }


def eval_priority_cost_control(
    N: int,
    q_W: float = 0.0,
    params: SwarmParams = SwarmParams(),
) -> Dict[str, float]:
    """Noiseless priority-ordered control where review j in queue costs c_0 * j^gamma (lambda = 0)."""
    rw = params.r_W(params.h_W, q_W)
    bw = params.B_W(params.h_W, q_W)
    bf = params.B_F
    a = params.r_F
    unreviewed_harm = N * params.p_W * params.L_W + params.p_F * params.L_F

    m_W = params.cutoff_m(bw)
    m_F = params.cutoff_m(bf)
    assert m_F >= 1, "Frontier first-slot review must satisfy B_F >= c_0."

    w_vals = np.arange(N + 1)
    pmf_w = binom.pmf(w_vals, N, rw)

    harm_averted = params.p_F * params.h_F * params.L_F  # F is always reviewed in slot 1 when A_F = 1
    reviews = 0.0
    cost = 0.0

    slot_costs = np.array([params.cost_per_review(idx) for idx in range(N + 2)], dtype=np.float64)
    cum_slot_costs = np.cumsum(slot_costs)

    for w, pw in zip(w_vals, pmf_w):
        if pw == 0.0:
            continue
        # A_F = 0: up to min(w, m_W) workers reviewed in slots 1..n0
        n0 = min(int(w), m_W)
        harm_averted += (1.0 - a) * pw * n0 * bw
        reviews += (1.0 - a) * pw * n0
        if n0 >= 1:
            cost += (1.0 - a) * pw * cum_slot_costs[n0]

        # A_F = 1: F reviewed in slot 1, workers reviewed in slots 2..min(w+1, m_W)
        max_slot = min(int(w) + 1, m_W)
        n1_w = max(0, max_slot - 1)
        harm_averted += a * pw * n1_w * bw
        reviews += a * pw * (1 + n1_w)
        cost += a * pw * cum_slot_costs[1 + n1_w]

    harm = unreviewed_harm - harm_averted
    return {
        "harm": float(harm),
        "cost": float(cost),
        "loss": float(harm + cost),
        "pings": float(a + N * rw),
        "reviews": float(reviews),
        "f_ovs": 1.0,
    }


def optimal_symmetric_throttling_cost(
    N: int,
    params: SwarmParams = SwarmParams(),
    grid_step: float = 0.005,
) -> Dict[str, float]:
    """Grid-searches uniform worker risky ask rate h_sym in [0, 1] (with q_W = 0) minimizing team loss."""
    h_grid = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 6)
    best: Optional[Dict[str, float]] = None
    for h_val in h_grid:
        res = eval_symmetric_roster_cost(N=N, j=N, q_W=0.0, h_W=float(h_val), params=params)
        res["h_sym"] = float(h_val)
        if best is None or res["loss"] < best["loss"]:
            best = res
    assert best is not None
    return best
