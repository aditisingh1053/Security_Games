"""Core data structures and the GROW-CATALOGUE algorithm.

This module implements the online learning algorithm proved in the report:
GROW-CATALOGUE for a repeated Stackelberg security game in which the set of
attacker types is unknown to the defender in advance but bounded by K_max.

Notation follows Balcan, Blum, Haghtalab, Procaccia (EC 2015).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Game primitives
# ----------------------------------------------------------------------------


@dataclass
class AttackerType:
    """An attacker type (utilities bounded in [-1, 1])."""
    u_c: np.ndarray       # (n,) covered-target utility for the attacker
    u_u: np.ndarray       # (n,) uncovered-target utility for the attacker
    type_id: int

    def utility_vec(self, p: np.ndarray) -> np.ndarray:
        return self.u_c * p + self.u_u * (1.0 - p)

    def best_response(self, p: np.ndarray) -> int:
        return int(np.argmax(self.utility_vec(p)))

    def __hash__(self):
        return hash(self.type_id)

    def __eq__(self, other):
        return isinstance(other, AttackerType) and self.type_id == other.type_id


@dataclass
class SSGame:
    """Single-resource Stackelberg security game with n targets.

    With R = 1 resource that may also be idle, the coverage polytope P is
    { p in R^n : p_i >= 0, sum(p) <= 1 }.
    """
    n: int
    u_d_c: np.ndarray     # (n,) defender payoff when target i is attacked AND covered
    u_d_u: np.ndarray     # (n,) defender payoff when attacked AND uncovered

    def defender_util_given_target(self, target: int, p: np.ndarray) -> float:
        return float(self.u_d_c[target] * p[target] + self.u_d_u[target] * (1.0 - p[target]))

    def payoff_against(self, attacker: AttackerType, p: np.ndarray) -> float:
        return self.defender_util_given_target(attacker.best_response(p), p)


# ----------------------------------------------------------------------------
# epsilon-approximate extreme point set cal E(C; epsilon)
# ----------------------------------------------------------------------------


def _projected_hyperplanes(n: int, catalogue: List[AttackerType]):
    """Return (a, b) pairs defining hyperplanes a . x = b in the (n-1)-dim
    projection of the coverage polytope. Simplex boundaries plus pairwise
    best-response boundaries for every type in the catalogue.
    """
    dim = n - 1
    hps = []
    for i in range(dim):
        a = np.zeros(dim); a[i] = 1.0
        hps.append((a, 0.0))
    hps.append((np.ones(dim), 1.0))  # sum x = 1 (i.e., p_{n-1} = 0)
    for alpha in catalogue:
        for i in range(n):
            for j in range(i + 1, n):
                Di = alpha.u_c[i] - alpha.u_u[i]
                Dj = alpha.u_c[j] - alpha.u_u[j]
                rhs_p = alpha.u_u[j] - alpha.u_u[i]
                coeff_p = np.zeros(n)
                coeff_p[i] = Di; coeff_p[j] = -Dj
                a = np.array([coeff_p[k] - coeff_p[n - 1] for k in range(dim)])
                rhs_x = rhs_p - coeff_p[n - 1]
                hps.append((a, rhs_x))
    return hps


def compute_extreme_points(
    game: SSGame, catalogue: List[AttackerType], tol: float = 1e-8
) -> List[np.ndarray]:
    """Enumerate vertices of the partition of P induced by the catalogue.

    For each (n-1)-subset of the defining hyperplanes, solve the linear
    system and keep the intersection point if it lies in P. Corresponds to
    the set cal E(C; epsilon) of Lemma 4.3 of Balcan et al. (2015) for a
    sufficiently small epsilon.
    """
    n = game.n
    dim = n - 1
    if dim == 0:
        return [np.array([1.0])]

    hps = _projected_hyperplanes(n, catalogue)
    uniques: List[np.ndarray] = []
    for subset in itertools.combinations(range(len(hps)), dim):
        A = np.array([hps[s][0] for s in subset])
        b = np.array([hps[s][1] for s in subset])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if np.any(x < -tol) or x.sum() > 1.0 + tol or (1.0 - x.sum()) < -tol:
            continue
        p = np.concatenate([x, [1.0 - x.sum()]])
        p = np.clip(p, 0.0, 1.0)
        s = p.sum()
        if s > 0:
            p = p / s
        if not any(np.linalg.norm(p - u, ord=np.inf) < 1e-6 for u in uniques):
            uniques.append(p)
    return uniques


# ----------------------------------------------------------------------------
# Anytime Hedge / Polynomial Weights sub-routine (Proposition 1 in the report)
# ----------------------------------------------------------------------------


class HedgeLearner:
    """Multiplicative weights with time-varying learning rate.

    The schedule eta_t = sqrt(log(N) / t) gives the anytime guarantee
    R_T' <= O(sqrt(T' log N)) for any T', which is exactly what the proof's
    Step 3 needs (epoch lengths are not known in advance).
    """

    def __init__(self, n_experts: int):
        self.n_experts = max(n_experts, 1)
        self.log_weights = np.zeros(self.n_experts)
        self.t = 0

    def distribution(self) -> np.ndarray:
        m = self.log_weights.max()
        w = np.exp(self.log_weights - m)
        s = w.sum()
        return (w / s) if s > 0 else np.ones(self.n_experts) / self.n_experts

    def update(self, losses: np.ndarray) -> None:
        assert losses.shape == (self.n_experts,)
        self.t += 1
        log_N = max(np.log(max(self.n_experts, 2)), 1e-3)
        eta = float(np.sqrt(log_N / self.t))
        self.log_weights -= eta * losses
        self.log_weights -= self.log_weights.max()


# ----------------------------------------------------------------------------
# GROW-CATALOGUE
# ----------------------------------------------------------------------------


class GrowCatalogue:
    """Algorithm 1 of the report.

    At every round:
      * if the current expert set is empty, play a default uniform coverage;
        otherwise sample from the Hedge distribution over cal E(C_t).
      * observe the attacker a_t.
      * if a_t is a new type, add it to the catalogue, recompute cal E(C_t)
        from scratch, and restart Hedge on the enlarged expert set.
      * otherwise do a standard Hedge update with the realized loss.
    """

    def __init__(
        self,
        game: SSGame,
        K_max: int,
        T: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.K_max = K_max
        self.T = T
        self.catalogue: List[AttackerType] = []
        self.experts: List[np.ndarray] = []
        self.learner: Optional[HedgeLearner] = None
        self.default_p = np.ones(game.n) / game.n
        self.rng = rng or np.random.default_rng()

    def play_distribution(self):
        """Return (distribution, experts) for deterministic payoff accounting."""
        if self.learner is None or len(self.experts) == 0:
            return np.array([1.0]), [self.default_p]
        return self.learner.distribution(), self.experts

    def play(self) -> np.ndarray:
        dist, experts = self.play_distribution()
        idx = int(self.rng.choice(len(experts), p=dist))
        return experts[idx]

    def observe(self, attacker: AttackerType) -> None:
        known_ids = {a.type_id for a in self.catalogue}
        if attacker.type_id not in known_ids:
            # Discovery round: add the type, rebuild cal E, restart Hedge
            self.catalogue.append(attacker)
            self.experts = compute_extreme_points(self.game, self.catalogue)
            self.learner = HedgeLearner(len(self.experts))
            return
        # Routine round: compute losses and update Hedge
        if not self.experts:
            return
        losses = np.zeros(len(self.experts))
        for k, p in enumerate(self.experts):
            losses[k] = -self.game.payoff_against(attacker, p)
        self.learner.update(losses)


# ----------------------------------------------------------------------------
# Helpers for experiments
# ----------------------------------------------------------------------------


def best_fixed_in_hindsight(
    game: SSGame, types: List[AttackerType], sequence: List[AttackerType]
):
    """Return (p*, cum_payoff): the best extreme point of the full partition
    against the realized attacker sequence. Used as the benchmark in regret
    plots (Equation 3 in the report).
    """
    experts = compute_extreme_points(game, types)
    best_p, best_val = experts[0], -np.inf
    for p in experts:
        v = sum(game.payoff_against(a, p) for a in sequence)
        if v > best_val:
            best_val, best_p = v, p
    return best_p, best_val


def expected_payoff(game: SSGame, algorithm: GrowCatalogue, attacker: AttackerType) -> float:
    """Deterministic expected payoff of the algorithm's current distribution.

    Used instead of a single sampled action so that regret curves do not have
    the Monte-Carlo noise of the algorithm's internal sampling.
    """
    dist, experts = algorithm.play_distribution()
    return float(sum(q * game.payoff_against(attacker, p) for q, p in zip(dist, experts)))
