"""Polynomial-time best-response oracle for GROW-CATALOGUE.

The expert set E(C; eps) of Balcan et al. (2015) has size exponential in the
number of targets n, and `algorithm.compute_extreme_points` builds it by
enumerating every (n-1)-subset of the defining hyperplanes.  That costs
C(H(n,K), n-1) linear solves with H(n,K) = n + K*n*(n-1)/2, which is already
7.3e14 subsets at n = 10, K = 4.

This module replaces the expert set by a *linear optimisation oracle* over the
lifted representation

    y[alpha, i] = 1{ b_alpha(p) = i },        w[alpha, i] = y[alpha, i] * p_i,

in which the defender's payoff against any type alpha is linear:

    U_d(b_alpha(p), p) = sum_i [ u_d_c[i] * w[alpha, i]
                                 + u_d_u[i] * (y[alpha, i] - w[alpha, i]) ].

The oracle solves

    max_{p in P}  sum_{alpha, i} ( a[alpha, i] * y[alpha, i]
                                   + b[alpha, i] * w[alpha, i] )

which, grouping by the best-response profile sigma : C -> N, is
max_sigma of a linear program over the cell P_sigma(C).  Two implementations
are provided:

  * `maximize_enum`  -- one LP per profile, n^|C| LPs.  Polynomial in n for a
    fixed catalogue size; used as the ground-truth reference.
  * `maximize_milp`  -- a single mixed-integer program with n*|C| binaries,
    in the style of the Bayesian-Stackelberg MILP of Paruchuri et al. (DOBSS).
    This is the one used in the experiments.

Both take a strictness margin `gamma > 0`: the returned p satisfies
b_alpha(p) = sigma(alpha) with slack at least gamma, so the answer does not
depend on the tie-breaking rule.  This plays the role of the eps-approximate
extreme points of Lemma 4.3 in Balcan et al.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# Big-M for the best-response constraints.  All attacker utilities lie in
# [-1, 1], so |max_j U(j, p) - U(i, p)| <= 2; 4 is a safe slack.
BIG_M = 4.0


@dataclass
class OracleInstance:
    """Everything the oracle needs for one call.

    Attacker and defender utilities are passed explicitly so that the same
    object serves both the full-information algorithm of Section 5 and the
    block-revelation algorithm of Section 10.
    """

    n: int
    u_d_c: np.ndarray                 # (n,)
    u_d_u: np.ndarray                 # (n,)
    att_u_c: np.ndarray               # (K, n)
    att_u_u: np.ndarray               # (K, n)
    budget: float = 1.0
    equality: bool = True             # sum(p) == budget, else sum(p) <= budget

    @property
    def K(self) -> int:
        return int(self.att_u_c.shape[0])

    def att_delta(self) -> np.ndarray:
        """D[alpha, i] = u_c[alpha, i] - u_u[alpha, i]; U = D * p_i + u_u."""
        return self.att_u_c - self.att_u_u

    def best_response(self, alpha: int, p: np.ndarray) -> int:
        return int(np.argmax(self.att_u_c[alpha] * p + self.att_u_u[alpha] * (1.0 - p)))

    def defender_payoff(self, target: int, p: np.ndarray) -> float:
        return float(self.u_d_c[target] * p[target] + self.u_d_u[target] * (1.0 - p[target]))

    def lifted_value(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Objective value at p, evaluated through the true best responses."""
        tot = 0.0
        for alpha in range(self.K):
            i = self.best_response(alpha, p)
            tot += float(a[alpha, i]) + float(b[alpha, i]) * float(p[i])
        return tot


def payoff_lift(inst: OracleInstance, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lifted coefficients of `sum_alpha weights[alpha] * U_d(b_alpha(p), p)`.

    Returns (a, b) with a[alpha, i] = weights[alpha] * u_d_u[i] and
    b[alpha, i] = weights[alpha] * (u_d_c[i] - u_d_u[i]).
    """
    weights = np.asarray(weights, dtype=float).reshape(inst.K, 1)
    a = weights * inst.u_d_u.reshape(1, -1)
    b = weights * (inst.u_d_c - inst.u_d_u).reshape(1, -1)
    return a, b


# ---------------------------------------------------------------------------
# Feasible region of the coverage polytope
# ---------------------------------------------------------------------------


def _budget_rows(inst: OracleInstance, n_vars: int) -> LinearConstraint:
    row = np.zeros((1, n_vars))
    row[0, : inst.n] = 1.0
    lb = inst.budget if inst.equality else -np.inf
    return LinearConstraint(row, lb, inst.budget)


# ---------------------------------------------------------------------------
# Reference implementation: one LP per best-response profile
# ---------------------------------------------------------------------------


def solve_cell(
    inst: OracleInstance,
    sigma: Sequence[int],
    a: np.ndarray,
    b: np.ndarray,
    gamma: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """Maximise the lifted objective over the single cell P_sigma(C).

    The cell is shrunk by the strictness margin gamma, so any returned p has
    b_alpha(p) = sigma(alpha) with slack >= gamma under *any* tie-breaking
    rule.  Returns None if the shrunk cell is empty.
    """
    n = inst.n
    D = inst.att_delta()

    c = np.zeros(n)
    const = 0.0
    for alpha, i in enumerate(sigma):
        c[i] -= float(b[alpha, i])          # linprog minimises
        const += float(a[alpha, i])

    rows, ub = [], []
    for alpha, i in enumerate(sigma):
        for j in range(n):
            if j == i:
                continue
            # U_alpha(i,p) - U_alpha(j,p) >= gamma
            r = np.zeros(n)
            r[i] -= D[alpha, i]
            r[j] += D[alpha, j]
            rows.append(r)
            ub.append(inst.att_u_u[alpha, i] - inst.att_u_u[alpha, j] - gamma)

    A_ub = np.array(rows) if rows else None
    b_ub = np.array(ub) if rows else None
    A_eq = np.ones((1, n)) if inst.equality else None
    b_eq = np.array([inst.budget]) if inst.equality else None
    if not inst.equality:
        extra = np.ones((1, n))
        A_ub = extra if A_ub is None else np.vstack([A_ub, extra])
        b_ub = np.array([inst.budget]) if b_ub is None else np.append(b_ub, inst.budget)

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=[(0.0, inst.budget)] * n, method="highs",
    )
    if not res.success:
        return None
    return np.clip(np.asarray(res.x), 0.0, inst.budget), float(-res.fun + const)


def _fallback(inst, a, b):
    p = np.full(inst.n, inst.budget / inst.n)
    sigma = tuple(inst.best_response(k, p) for k in range(inst.K))
    return p, inst.lifted_value(p, a, b), sigma


def maximize_enum(
    inst: OracleInstance,
    a: np.ndarray,
    b: np.ndarray,
    gamma: float = 1e-6,
) -> Tuple[np.ndarray, float, Tuple[int, ...]]:
    """Exact oracle by enumeration of the n^K best-response profiles.

    Returns (p, value, sigma).  Cost: n^K linear programs in n variables with
    O(K n) constraints, i.e. polynomial in n for a fixed catalogue.
    """
    n, K = inst.n, inst.K
    if K == 0:
        # No types yet: the objective is empty, any feasible point will do.
        return np.full(n, inst.budget / n), 0.0, ()

    best_p, best_val, best_sigma = None, -np.inf, None
    for sigma in itertools.product(range(n), repeat=K):
        out = solve_cell(inst, sigma, a, b, gamma)
        if out is None:
            continue
        p, val = out
        if val > best_val:
            best_val, best_p, best_sigma = val, p, sigma

    if best_p is None:                       # every cell was gamma-infeasible
        return _fallback(inst, a, b)
    return best_p, best_val, best_sigma


# ---------------------------------------------------------------------------
# Practical implementation: one MILP
# ---------------------------------------------------------------------------


def maximize_milp(
    inst: OracleInstance,
    a: np.ndarray,
    b: np.ndarray,
    gamma: float = 1e-6,
    time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float, Tuple[int, ...]]:
    """Exact oracle as a single MILP with n*K binaries.

    Variable layout:  x = [ p (n) | q (K*n, binary) | z (K*n) | v (K) ].
    q[alpha, i] = 1 iff type alpha best-responds by attacking i;
    z[alpha, i] = q[alpha, i] * p_i;  v[alpha] = max_j U_alpha(j, p).
    """
    n, K = inst.n, inst.K
    if K == 0:
        p = np.full(n, inst.budget / n)
        return p, 0.0, ()

    D = inst.att_delta()
    nP, nQ, nZ, nV = n, K * n, K * n, K
    N = nP + nQ + nZ + nV
    oP = 0
    oQ = nP
    oZ = nP + nQ
    oV = nP + nQ + nZ

    def qi(alpha, i):
        return oQ + alpha * n + i

    def zi(alpha, i):
        return oZ + alpha * n + i

    def vi(alpha):
        return oV + alpha

    # ---- objective (milp minimises) ----
    c = np.zeros(N)
    for alpha in range(K):
        for i in range(n):
            c[qi(alpha, i)] = -float(a[alpha, i])
            c[zi(alpha, i)] = -float(b[alpha, i])

    rows, lb, ub = [], [], []

    def add(row, lo, hi):
        rows.append(row)
        lb.append(lo)
        ub.append(hi)

    # budget
    r = np.zeros(N)
    r[oP:oP + n] = 1.0
    add(r, inst.budget if inst.equality else -np.inf, inst.budget)

    for alpha in range(K):
        # exactly one attacked target
        r = np.zeros(N)
        for i in range(n):
            r[qi(alpha, i)] = 1.0
        add(r, 1.0, 1.0)

        # Tight big-M per (alpha, i): the gap max_j U_alpha(j,p) - U_alpha(i,p)
        # is at most (max over j of the largest value U_alpha(j, .) can take)
        # minus (the smallest value U_alpha(i, .) can take).  This is typically
        # ~2x smaller than the generic bound and speeds the MILP up a lot.
        hi_u = np.maximum(inst.att_u_c[alpha], inst.att_u_u[alpha])
        lo_u = np.minimum(inst.att_u_c[alpha], inst.att_u_u[alpha])
        top = float(hi_u.max())

        for i in range(n):
            # v_alpha >= U_alpha(i, p) + gamma * (1 - q[alpha,i])
            #   <=>  v - D_i p_i + gamma q >= u_u_i + gamma
            r = np.zeros(N)
            r[vi(alpha)] = 1.0
            r[oP + i] -= D[alpha, i]
            r[qi(alpha, i)] += gamma
            add(r, float(inst.att_u_u[alpha, i]) + gamma, np.inf)

            # v_alpha <= U_alpha(i, p) + M * (1 - q[alpha,i])
            #   <=>  v - D_i p_i + M q <= u_u_i + M
            M = max(top - float(lo_u[i]), gamma) + 1e-9
            r = np.zeros(N)
            r[vi(alpha)] = 1.0
            r[oP + i] -= D[alpha, i]
            r[qi(alpha, i)] += M
            add(r, -np.inf, float(inst.att_u_u[alpha, i]) + M)

            # z = q * p_i.  Only the side the objective actually pushes on is
            # needed: with a positive coefficient the maximiser drives z up to
            # min(budget*q, p_i) = q*p_i, with a negative one it drives z down
            # to max(0, p_i - budget(1-q)) = q*p_i.  Both are exact.
            if float(b[alpha, i]) >= 0.0:
                r = np.zeros(N); r[zi(alpha, i)] = 1.0; r[qi(alpha, i)] = -inst.budget
                add(r, -np.inf, 0.0)                               # z <= budget * q
                r = np.zeros(N); r[zi(alpha, i)] = 1.0; r[oP + i] = -1.0
                add(r, -np.inf, 0.0)                               # z <= p_i
            else:
                r = np.zeros(N)
                r[zi(alpha, i)] = 1.0; r[oP + i] = -1.0; r[qi(alpha, i)] = -inst.budget
                add(r, -inst.budget, np.inf)                       # z >= p_i - budget(1-q)

    constraints = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))

    lo = np.zeros(N)
    hi = np.zeros(N)
    hi[oP:oP + n] = inst.budget
    hi[oQ:oQ + nQ] = 1.0
    hi[oZ:oZ + nZ] = inst.budget
    lo[oV:oV + nV] = -np.inf
    hi[oV:oV + nV] = np.inf

    integrality = np.zeros(N)
    integrality[oQ:oQ + nQ] = 1

    options = {"mip_rel_gap": 1e-7, "presolve": True}
    if time_limit:
        options["time_limit"] = time_limit
    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lo, hi),
        options=options,
    )
    if res.x is None:
        return _fallback(inst, a, b)

    x = res.x
    p = np.clip(x[oP:oP + n], 0.0, inst.budget)
    sigma = tuple(int(np.argmax(x[oQ + alpha * n: oQ + (alpha + 1) * n])) for alpha in range(K))

    # Polish: re-solve the single LP for the profile the MILP selected.  The
    # MILP may return a point on the boundary of its cell (its strictness
    # constraint holds only to solver tolerance), in which case the realised
    # best responses would disagree with sigma.  One extra LP fixes that and
    # makes the returned p strictly interior to the cell.
    out = solve_cell(inst, sigma, a, b, gamma)
    if out is not None:
        return out[0], out[1], sigma
    return p, inst.lifted_value(p, a, b), tuple(inst.best_response(k, p) for k in range(K))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class BestResponseOracle:
    """Linear-optimisation oracle over the lifted set S(C) = {Y(p) : p in P}."""

    def __init__(self, inst: OracleInstance, backend: str = "milp", gamma: float = 1e-7):
        if backend not in ("milp", "enum"):
            raise ValueError("backend must be 'milp' or 'enum'")
        self.inst = inst
        self.backend = backend
        self.gamma = gamma
        self.calls = 0

    def maximize(self, a: np.ndarray, b: np.ndarray):
        self.calls += 1
        fn = maximize_milp if self.backend == "milp" else maximize_enum
        return fn(self.inst, a, b, gamma=self.gamma)

    def maximize_weighted_payoff(self, weights: np.ndarray):
        """max_p sum_alpha weights[alpha] * U_d(b_alpha(p), p): the Bayesian
        Stackelberg problem against the type frequencies `weights`."""
        a, b = payoff_lift(self.inst, weights)
        return self.maximize(a, b)


def instance_from_game(game, catalogue: Sequence, equality: bool = True) -> OracleInstance:
    """Build an OracleInstance from `algorithm.SSGame` + a list of AttackerType."""
    if catalogue:
        u_c = np.array([alpha.u_c for alpha in catalogue], dtype=float)
        u_u = np.array([alpha.u_u for alpha in catalogue], dtype=float)
    else:
        u_c = np.zeros((0, game.n))
        u_u = np.zeros((0, game.n))
    return OracleInstance(
        n=game.n,
        u_d_c=np.asarray(game.u_d_c, dtype=float),
        u_d_u=np.asarray(game.u_d_u, dtype=float),
        att_u_c=u_c,
        att_u_u=u_u,
        equality=equality,
    )
