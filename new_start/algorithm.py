"""GROW-CATALOGUE algorithm for repeated Stackelberg Security Games with unknown K.

Follows the model of Balcan, Blum, Haghtalab, Procaccia (EC 2015), extended so that
the set of attacker types is unknown a priori but has cardinality at most K_max.
The defender learns a new type's full utility upon first observing it (full info).

Key classes:
  - AttackerType: attacker utilities u^c, u^u
  - SSGame: defender utilities (single-resource SSG, P = n-simplex)
  - compute_extreme_points: enumerates epsilon-approximate extreme points of the
        partition of P induced by the best-response hyperplanes of a catalogue
  - HedgeLearner: anytime multiplicative weights subroutine
  - GrowCatalogue: Algorithm 1 of our report (both total-recompute and incremental)
  - KnownKOracle: Balcan et al. Theorem 5.1 baseline (knows K upfront)
  - UniformBaseline: plays uniform coverage (R/n, ..., R/n) always
"""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Game primitives
# ----------------------------------------------------------------------------


@dataclass
class AttackerType:
    """An attacker type, identified by a unique integer id.

    Utilities are bounded in [-1, 1] as in Balcan et al. (Sec 2).
    """

    u_c: np.ndarray  # (n,) utility when covered
    u_u: np.ndarray  # (n,) utility when uncovered
    type_id: int

    def utility_vec(self, p: np.ndarray) -> np.ndarray:
        return self.u_c * p + self.u_u * (1.0 - p)

    def best_response(self, p: np.ndarray) -> int:
        # consistent tie-breaking: smallest index
        utils = self.utility_vec(p)
        return int(np.argmax(utils))

    def __hash__(self):
        return hash(self.type_id)

    def __eq__(self, other):
        return isinstance(other, AttackerType) and self.type_id == other.type_id


@dataclass
class SSGame:
    """Single-resource Stackelberg Security Game.

    With R = 1 resource that must be placed on one of n targets, pure strategies are
    the n one-hot vectors and the mixed-strategy coverage space P is the n-simplex:
        P = {p in R^n : p_i >= 0, sum p_i = 1}.
    Defender utilities u_d^c(i), u_d^u(i) in [-1, 1].
    """

    n: int
    u_d_c: np.ndarray  # (n,)
    u_d_u: np.ndarray  # (n,)

    def defender_util_given_target(self, target: int, p: np.ndarray) -> float:
        return float(self.u_d_c[target] * p[target] + self.u_d_u[target] * (1.0 - p[target]))

    def payoff_against(self, attacker: AttackerType, p: np.ndarray) -> float:
        target = attacker.best_response(p)
        return self.defender_util_given_target(target, p)


# ----------------------------------------------------------------------------
# Extreme point computation
# ----------------------------------------------------------------------------


def _projected_hyperplanes(n: int, catalogue: List[AttackerType]):
    """Return list of (a, b) where a·x = b in (n-1)-dim space after projecting out p_{n-1}.

    The hyperplanes are:
      (i)   simplex boundary hyperplanes: x_i = 0 for i = 0, ..., n-2
                                          sum x = 1 (corresponding to p_{n-1} = 0)
      (ii)  best-response hyperplanes: for each alpha in catalogue and each pair i < j,
            U_alpha(i, p) = U_alpha(j, p)
    """
    dim = n - 1
    hps = []
    # x_i = 0
    for i in range(dim):
        a = np.zeros(dim)
        a[i] = 1.0
        hps.append((a, 0.0))
    # sum x = 1  <=> p_{n-1} = 0
    hps.append((np.ones(dim), 1.0))
    # best-response hyperplanes
    for alpha in catalogue:
        for i in range(n):
            for j in range(i + 1, n):
                Di = alpha.u_c[i] - alpha.u_u[i]
                Dj = alpha.u_c[j] - alpha.u_u[j]
                rhs_p = alpha.u_u[j] - alpha.u_u[i]
                # In n-dim p: Di * p_i - Dj * p_j = rhs_p
                coeff_p = np.zeros(n)
                coeff_p[i] = Di
                coeff_p[j] = -Dj
                # Project p_{n-1} = 1 - sum_{k<n-1} x_k
                a = np.zeros(dim)
                for k in range(dim):
                    a[k] = coeff_p[k] - coeff_p[n - 1]
                rhs_x = rhs_p - coeff_p[n - 1]
                hps.append((a, rhs_x))
    return hps


def compute_extreme_points(
    game: SSGame,
    catalogue: List[AttackerType],
    tol: float = 1e-8,
) -> List[np.ndarray]:
    """Enumerate all (n-1)-subsets of hyperplanes and solve for their intersection.

    Each intersection point that lies inside P is an extreme point of some region of
    the partition induced by the catalogue, and hence belongs to the set E(C; eps)
    of Definition 1 in the report.
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
        # Rank / invertibility check
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        # Feasibility in the projected simplex
        if np.any(x < -tol):
            continue
        if x.sum() > 1.0 + tol:
            continue
        p_last = 1.0 - x.sum()
        if p_last < -tol:
            continue
        p = np.concatenate([x, [p_last]])
        p = np.clip(p, 0.0, 1.0)
        s = p.sum()
        if s > 0:
            p = p / s
        # dedupe
        if not any(np.linalg.norm(p - u, ord=np.inf) < 1e-6 for u in uniques):
            uniques.append(p)

    return uniques


# ----------------------------------------------------------------------------
# Hedge (anytime)
# ----------------------------------------------------------------------------


class HedgeLearner:
    """Multiplicative-weights / Hedge with time-varying learning rate.

    eta_t = sqrt(log(N) / t) gives the anytime regret guarantee
        R_T' <= O(sqrt(T' log N))
    for any number T' of update steps, without knowing T' upfront.
    """

    def __init__(self, n_experts: int):
        self.n_experts = max(n_experts, 1)
        self.log_weights = np.zeros(self.n_experts)
        self.t = 0

    def distribution(self) -> np.ndarray:
        m = self.log_weights.max()
        w = np.exp(self.log_weights - m)
        s = w.sum()
        if s == 0:
            return np.ones(self.n_experts) / self.n_experts
        return w / s

    def update(self, losses: np.ndarray) -> None:
        """losses: (n_experts,), each in [-1, 1]."""
        assert losses.shape == (self.n_experts,), (losses.shape, self.n_experts)
        self.t += 1
        log_N = max(np.log(max(self.n_experts, 2)), 1e-3)
        eta = float(np.sqrt(log_N / self.t))
        self.log_weights -= eta * losses
        # stability: shift
        self.log_weights -= self.log_weights.max()


# ----------------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------------


class GrowCatalogue:
    """Algorithm 1 of the report: GROW-CATALOGUE for unknown K with K_max bound.

    `recompute` controls how extreme points are updated:
      - 'total':   recompute cE(C) from scratch whenever a new type is observed
      - 'incremental': same E but reuse Hedge weights where possible
      (Both variants give identical regret; only wall-clock differs.)
    """

    def __init__(
        self,
        game: SSGame,
        K_max: int,
        T: int,
        recompute: str = "total",
        rng: Optional[np.random.Generator] = None,
    ):
        assert recompute in ("total", "incremental")
        self.game = game
        self.K_max = K_max
        self.T = T
        self.recompute = recompute
        self.catalogue: List[AttackerType] = []
        self.experts: List[np.ndarray] = []
        self.learner: Optional[HedgeLearner] = None
        self.default_p = np.ones(game.n) / game.n
        self.rng = rng or np.random.default_rng()
        self._last_dist: Optional[np.ndarray] = None

    def play_distribution(self) -> (np.ndarray, List[np.ndarray]):
        """Return (distribution, experts) -- used for expected-payoff accounting."""
        if self.learner is None or len(self.experts) == 0:
            return np.array([1.0]), [self.default_p]
        return self.learner.distribution(), self.experts

    def play(self) -> np.ndarray:
        dist, exs = self.play_distribution()
        idx = int(self.rng.choice(len(exs), p=dist))
        return exs[idx]

    def _reset_learner_for(self, new_experts: List[np.ndarray]) -> None:
        if self.recompute == "total" or self.learner is None:
            self.experts = new_experts
            self.learner = HedgeLearner(len(new_experts))
            return
        # incremental: transfer weights of surviving experts
        old_weights = {}
        if self.learner is not None:
            dist = self.learner.distribution()
            for p_old, d_old in zip(self.experts, dist):
                old_weights[tuple(np.round(p_old, 6))] = np.log(max(d_old, 1e-12))
        self.experts = new_experts
        new_learner = HedgeLearner(len(new_experts))
        for k, p in enumerate(new_experts):
            key = tuple(np.round(p, 6))
            if key in old_weights:
                new_learner.log_weights[k] = old_weights[key]
        new_learner.log_weights -= new_learner.log_weights.max()
        # keep the Hedge clock running so eta_t continues to shrink
        new_learner.t = self.learner.t if self.learner is not None else 0
        self.learner = new_learner

    def observe(self, attacker: AttackerType) -> None:
        known_ids = {a.type_id for a in self.catalogue}
        if attacker.type_id not in known_ids:
            # Discovery round: add to catalogue and refresh expert set
            self.catalogue.append(attacker)
            new_experts = compute_extreme_points(self.game, self.catalogue)
            self._reset_learner_for(new_experts)
            return
        # Known type: compute losses and update Hedge
        if not self.experts:
            return
        losses = np.zeros(len(self.experts))
        for k, p in enumerate(self.experts):
            losses[k] = -self.game.payoff_against(attacker, p)
        self.learner.update(losses)


class KnownKOracle:
    """Balcan et al. 2015 Theorem 5.1 baseline: knows the full type set K upfront."""

    def __init__(
        self,
        game: SSGame,
        full_K: List[AttackerType],
        T: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.T = T
        self.experts = compute_extreme_points(game, full_K)
        self.learner = HedgeLearner(max(len(self.experts), 1))
        self.rng = rng or np.random.default_rng()

    def play_distribution(self):
        return self.learner.distribution(), self.experts

    def play(self) -> np.ndarray:
        dist = self.learner.distribution()
        idx = int(self.rng.choice(len(self.experts), p=dist))
        return self.experts[idx]

    def observe(self, attacker: AttackerType) -> None:
        losses = np.zeros(len(self.experts))
        for k, p in enumerate(self.experts):
            losses[k] = -self.game.payoff_against(attacker, p)
        self.learner.update(losses)


class UniformBaseline:
    """Plays uniform coverage p = (1/n, ..., 1/n) at every round."""

    def __init__(self, game: SSGame, T: int, rng: Optional[np.random.Generator] = None):
        self.game = game
        self.T = T
        self.p = np.ones(game.n) / game.n

    def play_distribution(self):
        return np.array([1.0]), [self.p]

    def play(self) -> np.ndarray:
        return self.p

    def observe(self, attacker: AttackerType) -> None:
        pass


# ----------------------------------------------------------------------------
# Utility: best fixed strategy in hindsight (benchmark)
# ----------------------------------------------------------------------------


def best_fixed_in_hindsight(
    game: SSGame, types: List[AttackerType], sequence: List[AttackerType]
) -> (np.ndarray, float):
    """Return (p*, cum_payoff) of the best extreme point against the realized sequence."""
    experts = compute_extreme_points(game, types)
    best_p = experts[0]
    best_val = -np.inf
    for p in experts:
        v = 0.0
        for a in sequence:
            v += game.payoff_against(a, p)
        if v > best_val:
            best_val = v
            best_p = p
    return best_p, best_val


def expected_payoff(game: SSGame, algorithm, attacker: AttackerType) -> float:
    """Expected payoff of `algorithm` against `attacker` under its current distribution.

    This is the deterministic conditional expectation -- useful to reduce variance
    in regret curves when reporting per-seed results.
    """
    dist, experts = algorithm.play_distribution()
    v = 0.0
    for q, p in zip(dist, experts):
        v += q * game.payoff_against(attacker, p)
    return v
