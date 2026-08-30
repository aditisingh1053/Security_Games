"""Oracle-efficient implementations of GROW-CATALOGUE and BLOCK-GROW-CATALOGUE.

Both algorithms in the report keep an expert set E(C; eps) whose size is
exponential in the number of targets n.  This module replaces every use of
that set by calls to the linear-optimisation oracle of `oracle.py`, so the
per-round cost is polynomial in n for a fixed catalogue size and the
experiments can be run at n = 10, 20 instead of n = 3.

  * `OracleGrowCatalogue`   -- full-information feedback (Algorithm 1 of the
                               report), Hedge over E(C) replaced by Follow the
                               Perturbed Leader over the lifted set S(C).
  * `BlockGrowCatalogue`    -- partial-information feedback with revelation
                               blocks of length B (Algorithm 2 of the report).

Both use at most one oracle call per round: exactly one on an
exploitation round, and none on the exploration rounds of
`BlockGrowCatalogue`, which play a fixed spanner strategy.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence

import numpy as np

from oracle import (
    BestResponseOracle,
    OracleInstance,
    instance_from_game,
    payoff_lift,
)


# ---------------------------------------------------------------------------
# Follow the Perturbed Leader over the lifted set S(C) = { Y(p) : p in P }
# ---------------------------------------------------------------------------
#
# The lifted coordinates are y[alpha,i] = 1{b_alpha(p) = i} and
# w[alpha,i] = y[alpha,i] * p_i, in which the defender's payoff against a
# weighted mixture of types is linear.  Kalai and Vempala's FPL therefore
# applies.  Its three problem constants on this decision set are
#   ||Y - Y'||_1 <= 4K,   |Y . s| <= 1,   ||s||_1 <= 3n,
# giving the anytime schedule eps_t = sqrt(4K / (3 n t)).  Balcan et al.
# explicitly sanction Follow the (Lazy) Leader as a black box for their
# Proposition 1, so this is an implementation change, not a new algorithm.


def _fpl_eps(K: int, n: int, t: int) -> float:
    D, R, A = 4.0 * max(K, 1), 1.0, 3.0 * n
    return float(min(1.0, np.sqrt(D / (R * A * max(t, 1)))))


def _perturbed_leader(oracle, weights, rng, t, n) -> np.ndarray:
    """argmax_p sum_alpha weights[alpha] U_d(b_alpha(p), p), FPL-perturbed."""
    K = len(weights)
    a, b = payoff_lift(oracle.inst, weights)
    eps = _fpl_eps(K, n, t)
    # FPL adds Uniform[0, 1/eps] to the cumulative *cost*; the cost is the
    # negated payoff, so the perturbation is subtracted here.
    a = a - rng.uniform(0.0, 1.0 / eps, size=(K, n))
    b = b - rng.uniform(0.0, 1.0 / eps, size=(K, n))
    p, _, _ = oracle.maximize(a, b)
    return p


# ---------------------------------------------------------------------------
# Full information: Algorithm 1 of the report
# ---------------------------------------------------------------------------


class OracleGrowCatalogue:
    """GROW-CATALOGUE with the Hedge-over-E(C) sub-routine replaced by FPL.

    The state inside an epoch is just the vector of observed type counts: the
    cumulative lifted loss after t rounds is exactly the lift of
    -sum_alpha count[alpha] U_d(b_alpha(p), p), so the unperturbed leader is
    the Bayesian-Stackelberg optimum against the empirical type frequencies.
    """

    def __init__(self, game, K_max: int, T: int, backend: str = "milp",
                 gamma: float = 1e-6, rng=None, equality: bool = True):
        self.game, self.n = game, game.n
        self.K_max, self.T = K_max, T
        self.backend, self.gamma, self.equality = backend, gamma, equality
        self.rng = rng or np.random.default_rng()
        self.catalogue: List = []
        self.counts = np.zeros(0)
        self.t_local = 0
        self.oracle: Optional[BestResponseOracle] = None
        self.default_p = np.full(self.n, 1.0 / self.n)
        self.oracle_calls = 0

    def _rebuild(self) -> None:
        inst = instance_from_game(self.game, self.catalogue, equality=self.equality)
        self.oracle = BestResponseOracle(inst, backend=self.backend, gamma=self.gamma)
        self.counts = np.zeros(len(self.catalogue))
        self.t_local = 0

    def play(self) -> np.ndarray:
        if self.oracle is None or not self.catalogue:
            return self.default_p
        self.oracle_calls += 1
        return _perturbed_leader(self.oracle, self.counts, self.rng,
                                 self.t_local, self.n)

    def observe(self, attacker) -> None:
        ids = [alpha.type_id for alpha in self.catalogue]
        if attacker.type_id not in ids:           # discovery round
            self.catalogue.append(attacker)
            self._rebuild()
            return
        self.counts[ids.index(attacker.type_id)] += 1.0
        self.t_local += 1


# ---------------------------------------------------------------------------
# Barycentric spanner for W = { I_(sigma = i) }  (Balcan et al., Section 6.3)
# ---------------------------------------------------------------------------


def _profile(inst: OracleInstance, p: np.ndarray) -> np.ndarray:
    return np.array([inst.best_response(alpha, p) for alpha in range(inst.K)])


def collect_W(inst: OracleInstance, rng, n_samples: int = 400):
    """Sample realisable vectors I_(sigma = i) together with a witness (p, i).

    Every w in W is, by definition, the indicator of the set of types that
    attack some target i at some coverage vector p, so sampling p and reading
    off its best-response profile produces elements of W with their witnesses
    for free.  Vertices of P are included because the extreme cells of the
    partition are the ones that separate the types best.
    """
    K, n = inst.K, inst.n
    cand, seen = [], set()

    def add(p):
        sig = _profile(inst, p)
        for i in range(n):
            w = (sig == i).astype(float)
            key = tuple(w.astype(int))
            if key not in seen:
                seen.add(key)
                cand.append((w, p.copy(), i))

    add(np.full(n, inst.budget / n))
    for i in range(n):                                   # vertices of P
        p = np.zeros(n); p[i] = inst.budget; add(p)
    for _ in range(n_samples):
        add(rng.dirichlet(np.ones(n)) * inst.budget)
    return cand


def barycentric_spanner(cand, K: int):
    """Greedy determinant maximisation over the sampled pool `cand`.

    Follows Awerbuch and Kleinberg (2008, Prop. 2.2): build a basis greedily,
    then repeatedly swap in a vector that increases |det|.  Two caveats.  The
    pool is a sample of W rather than all of W, so the coefficients lambda(w)
    are not guaranteed to lie in [-1,1] for w outside the pool; and the swap
    loop is capped at 4K sweeps rather than run to a local optimum.  Both
    affect only the *range* of the loss estimator, not its unbiasedness,
    which needs nothing more than an invertible Lambda.
    """
    vecs = [w for (w, _, _) in cand]
    if not vecs:
        return None
    basis_idx: List[int] = []
    M = np.zeros((K, K))
    for d in range(K):                       # build an initial basis greedily
        best, best_val = None, 1e-9
        for j, w in enumerate(vecs):
            if j in basis_idx:
                continue
            M[d] = w
            val = abs(np.linalg.det(M)) if d == K - 1 else _partial_vol(M[: d + 1])
            if val > best_val:
                best_val, best = val, j
        if best is None:
            return None
        M[d] = vecs[best]
        basis_idx.append(best)
    for _ in range(4 * K):                   # local improvement swaps
        improved = False
        for d in range(K):
            cur = abs(np.linalg.det(M))
            for j, w in enumerate(vecs):
                old = M[d].copy()
                M[d] = w
                if abs(np.linalg.det(M)) > cur * 1.0 + 1e-12:
                    basis_idx[d], cur, improved = j, abs(np.linalg.det(M)), True
                else:
                    M[d] = old
        if not improved:
            break
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    return M, [cand[j] for j in basis_idx]


def _partial_vol(rows: np.ndarray) -> float:
    G = rows @ rows.T
    return float(max(np.linalg.det(G), 0.0)) ** 0.5


# ---------------------------------------------------------------------------
# Partial information + block revelation: Algorithm 2 of the report
# ---------------------------------------------------------------------------


class BlockGrowCatalogue:
    """BLOCK-GROW-CATALOGUE.

    Feedback per round is the attacked target only.  At the end of every
    revelation block of length B an oracle returns the *set* of types that
    have attacked so far, with their utility vectors, and nothing else.

    Inside an epoch (a maximal run of revelation blocks over which the
    revealed set does not change) the algorithm is exactly Algorithm 1 of
    Balcan et al. (2015): the epoch is cut into estimation blocks; in each
    one, |C| rounds chosen uniformly at random are spent playing the spanner
    strategies p_b and recording whether target i_b was attacked; the
    resulting vector g_tau = Bm^{-1} phat_tau is an unbiased estimator of the
    per-round type frequencies in that estimation block, and the cumulative
    G = sum_tau g_tau is fed to the full-information learner (here FPL).

    When the revealed set grows at a checkpoint, the whole revelation block
    that produced the growth is discarded together with the estimator state,
    and the sub-routine restarts on the enlarged catalogue.
    """

    def __init__(self, game, K_max: int, T: int, B: int, backend: str = "milp",
                 gamma: float = 1e-6, rng=None, equality: bool = True,
                 est_block: Optional[int] = None):
        self.game, self.n = game, game.n
        self.K_max, self.T, self.B = K_max, T, B
        self.backend, self.gamma, self.equality = backend, gamma, equality
        self.rng = rng or np.random.default_rng()
        # Balcan et al. take Z = Theta(T^{2/3} n log^{1/3}(nk)) estimation
        # blocks, i.e. blocks of length Theta(T^{1/3} / n); it must be at
        # least |C| + 1 so that every spanner direction fits inside one.
        self.est_block_hint = est_block or max(2, int(round(T ** (1.0 / 3.0))))

        self.catalogue: List = []
        self.oracle: Optional[BestResponseOracle] = None
        self.G = np.zeros(0)                 # cumulative estimated type counts
        self.Bm = None                       # spanner matrix (K x K)
        self.spanner: List = []              # [(w, p_b, i_b)]
        self.default_p = np.full(self.n, 1.0 / self.n)

        self.t_local = 0                     # rounds since the last restart
        self._blk_pos = 0                    # position inside the estimation block
        self._blk_len = 0
        self._explore_slots: dict = {}       # slot -> spanner index
        self._phat = np.zeros(0)
        self._pending = None                 # (spanner index) if exploring now
        self.oracle_calls = 0
        self.explore_rounds = 0
        self.rank_failures = 0

    # -- catalogue management ------------------------------------------------
    def _rebuild(self) -> None:
        inst = instance_from_game(self.game, self.catalogue, equality=self.equality)
        self.oracle = BestResponseOracle(inst, backend=self.backend, gamma=self.gamma)
        K = len(self.catalogue)
        cand = collect_W(inst, self.rng)
        out = barycentric_spanner(cand, K)
        if out is None:
            self.rank_failures += 1
            self.Bm, self.spanner = np.eye(K), []
        else:
            self.Bm, self.spanner = out
        self.G = np.zeros(K)
        self.t_local = 0
        self._start_estimation_block()

    def _start_estimation_block(self) -> None:
        K = len(self.catalogue)
        self._blk_len = max(self.est_block_hint, K + 1)
        self._blk_pos = 0
        self._phat = np.zeros(K)
        if self.spanner:
            slots = self.rng.choice(self._blk_len, size=K, replace=False)
            self._explore_slots = {int(s): j for j, s in enumerate(slots)}
        else:
            self._explore_slots = {}

    def _close_estimation_block(self) -> None:
        if self.spanner and self._blk_len > 0:
            try:
                g = np.linalg.solve(self.Bm, self._phat)
            except np.linalg.LinAlgError:
                g = np.linalg.lstsq(self.Bm, self._phat, rcond=None)[0]
            # phat is a per-round frequency; multiply by the block length to
            # recover an estimate of the type *counts* in this block.
            self.G += g * self._blk_len
        self._start_estimation_block()

    # -- interaction ---------------------------------------------------------
    def play(self) -> np.ndarray:
        if self.oracle is None or not self.catalogue:
            self._pending = None
            return self.default_p
        j = self._explore_slots.get(self._blk_pos)
        if j is not None:
            self._pending = j
            self.explore_rounds += 1
            return self.spanner[j][1].copy()
        self._pending = None
        self.oracle_calls += 1
        return _perturbed_leader(self.oracle, self.G, self.rng,
                                 self.t_local, self.n)

    def observe_target(self, target: int) -> None:
        """Partial-information feedback: only the attacked target."""
        if self._pending is not None:
            self._phat[self._pending] = 1.0 if target == self.spanner[self._pending][2] else 0.0
            self._pending = None
        self.t_local += 1
        self._blk_pos += 1
        if self._blk_pos >= self._blk_len:
            self._close_estimation_block()

    def checkpoint(self, revealed: Sequence) -> bool:
        """End of a revelation block: `revealed` is the set of types seen so far.

        `revealed` arrives as a Python list only because that is convenient;
        it is read purely as a set.  The only thing done with it is a
        membership test by `type_id`, so no information carried by the order
        of the list -- in particular the order in which the types first
        attacked -- can reach the algorithm.  That matters because Problem 15
        grants the defender the set and nothing else.

        Returns True if the catalogue grew (so the block just finished was a
        discovery block and its estimator contribution has been discarded).
        """
        ids = {alpha.type_id for alpha in self.catalogue}
        new = [alpha for alpha in revealed if alpha.type_id not in ids]
        if not new:
            return False
        self.catalogue = list(self.catalogue) + list(new)
        self._rebuild()
        return True
