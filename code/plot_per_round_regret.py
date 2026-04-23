"""Generate Figure 1 of the report: rolling-window per-round regret vs
iteration for GROW-CATALOGUE under the staggered-uniform adversary.

Saves to figures/avg_regret_per_iter.{pdf,png}.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm import (
    AttackerType,
    GrowCatalogue,
    SSGame,
    best_fixed_in_hindsight,
    expected_payoff,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def make_game(n, rng):
    return SSGame(
        n=n,
        u_d_c=rng.uniform(0.1, 1.0, n),
        u_d_u=rng.uniform(-1.0, -0.1, n),
    )


def make_types(n, k, rng):
    return [
        AttackerType(
            u_c=rng.uniform(-1.0, -0.1, n),
            u_u=rng.uniform(0.1, 1.0, n),
            type_id=i,
        )
        for i in range(k)
    ]


def make_staggered_uniform_sequence(T, types, rng):
    """In phase i, attacker is drawn uniformly from {alpha_1, ..., alpha_i}."""
    K = len(types)
    L = T // K
    seq, reveal_rounds = [], []
    for i in range(1, K + 1):
        active = types[:i]
        reveal_rounds.append((i - 1) * L)
        phase_len = L if i < K else T - (i - 1) * L
        idxs = rng.integers(0, i, size=phase_len)
        seq.extend(active[j] for j in idxs)
    return seq, reveal_rounds


def run_one(seed, n, K, T):
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    seq, reveal_rounds = make_staggered_uniform_sequence(
        T, types, np.random.default_rng(seed + 7777)
    )
    p_star, _ = best_fixed_in_hindsight(game, types, seq)
    alg = GrowCatalogue(game, K_max=K, T=T, rng=np.random.default_rng(seed + 10000))
    per_round = np.zeros(T)
    for t, a in enumerate(seq):
        per_round[t] = game.payoff_against(a, p_star) - expected_payoff(game, alg, a)
        alg.observe(a)
    return per_round, reveal_rounds


def rolling_mean(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    n, K, T, n_seeds, window = 3, 4, 4000, 40, 75
    os.makedirs(OUT_DIR, exist_ok=True)

    all_per_round = np.zeros((n_seeds, T))
    reveal_rounds = None
    for s in range(n_seeds):
        pr, rev = run_one(s, n, K, T)
        all_per_round[s] = pr
        if reveal_rounds is None:
            reveal_rounds = rev

    mean_pr = all_per_round.mean(axis=0)
    std_pr = all_per_round.std(axis=0)
    rmean = rolling_mean(mean_pr, window)
    rstd = rolling_mean(std_pr, window) / np.sqrt(n_seeds)
    t_axis = np.arange(len(rmean)) + window

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, tau in enumerate(reveal_rounds):
        ax.axvline(tau + 1, color="#d62728", ls="--", lw=1.3, alpha=0.8,
                   label="Type-reveal round" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(t_axis, rmean, color="#1f77b4", lw=2.0,
            label=f"Rolling-mean per-round regret (window={window})")
    ax.fill_between(t_axis, rmean - rstd, rmean + rstd,
                    color="#1f77b4", alpha=0.25, label=rf"$\pm 1$ s.e. over {n_seeds} seeds")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Per-round regret")
    ax.set_title(
        f"GROW-CATALOGUE: average per-round regret vs iteration  "
        f"($n={n}$, $K_{{max}}={K}$, staggered-uniform adversary)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "avg_regret_per_iter.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "avg_regret_per_iter.png"), dpi=150)
    print(f"Saved to {OUT_DIR}/avg_regret_per_iter.{{pdf,png}}")


if __name__ == "__main__":
    main()
