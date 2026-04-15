"""Presentation-quality experiment for GROW-CATALOGUE.

Sequence design (staggered reveal + uniform mixing):
  Partition [0, T) into K_max phases of equal length L = T / K_max.
  In phase i (i = 1, ..., K_max) the attacker set is {alpha_1, ..., alpha_i} and
  each round the type is drawn UNIFORMLY AT RANDOM from that set.
  Thus:
    - at round t = 0 only alpha_1 can attack;
    - at round t = L type alpha_2 is introduced and from then on types
      {alpha_1, alpha_2} mix 50/50;
    - at round t = 2L type alpha_3 is introduced, 1/3 each, ... etc.
  This is more realistic than strict slow-reveal because old types keep
  arriving after the reveal, so the benchmark p* is not trivially dominated
  by a single specialist.

Plots (saved to figures/presentation/):
  1. avg_regret_per_iter.{pdf,png}
       rolling-window mean of per-round regret, with vertical markers at the
       rounds where a new type is revealed.
  2. global_avg_regret.{pdf,png}
       cumulative regret divided by t (global average regret so far), with
       vertical markers at the reveal rounds.
  3. occurrence_distribution.{pdf,png}
       bar chart of the frequency with which each attacker type appears
       across the whole run, averaged across seeds.
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


# ----------------------------------------------------------------------------
# Instance + sequence construction
# ----------------------------------------------------------------------------


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


def make_staggered_uniform_sequence(
    T: int, types, rng: np.random.Generator
):
    """Phase i uses types {alpha_1, ..., alpha_i} uniformly at random.

    Reveal rounds: i*L for i = 0, 1, ..., K-1, where L = T // K.
    """
    K = len(types)
    L = T // K
    seq = []
    reveal_rounds = []
    for i in range(1, K + 1):
        active = types[:i]  # types 1..i available in phase i
        phase_start = (i - 1) * L
        reveal_rounds.append(phase_start)
        phase_len = L if i < K else T - phase_start  # last phase absorbs remainder
        idxs = rng.integers(0, i, size=phase_len)
        for j in idxs:
            seq.append(active[j])
    return seq, reveal_rounds


# ----------------------------------------------------------------------------
# Single experimental run
# ----------------------------------------------------------------------------


def run_one(seed, n, K, T):
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    seq, reveal_rounds = make_staggered_uniform_sequence(
        T, types, np.random.default_rng(seed + 7777)
    )

    p_star, _ = best_fixed_in_hindsight(game, types, seq)

    alg = GrowCatalogue(
        game,
        K_max=K,
        T=T,
        recompute="total",
        rng=np.random.default_rng(seed + 10000),
    )

    per_round_regret = np.zeros(T)
    type_ids = np.zeros(T, dtype=int)
    for t, a in enumerate(seq):
        p_algo_exp = expected_payoff(game, alg, a)
        p_bench = game.payoff_against(a, p_star)
        per_round_regret[t] = p_bench - p_algo_exp
        type_ids[t] = a.type_id
        alg.observe(a)
    return per_round_regret, type_ids, reveal_rounds, K


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------


REVEAL_COLOR = "#d62728"
ALG_COLOR = "#1f77b4"
AVG_COLOR = "#2ca02c"


def _mark_reveals(ax, reveal_rounds):
    for i, tau in enumerate(reveal_rounds):
        lbl = "Type-reveal round" if i == 0 else None
        ax.axvline(tau, color=REVEAL_COLOR, ls="--", lw=1.3, alpha=0.85, label=lbl)
        ax.text(
            tau,
            0.97,
            f" reveal {i + 1}",
            color=REVEAL_COLOR,
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            rotation=90,
            va="top",
            ha="right",
        )


def rolling_mean(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------


def plot_per_iter_regret(
    mean_pr, std_pr, reveal_rounds, out_dir, window, n_seeds, n, K
):
    rmean = rolling_mean(mean_pr, window)
    rstd = rolling_mean(std_pr, window) / np.sqrt(n_seeds)
    t = np.arange(len(rmean)) + window

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _mark_reveals(ax, reveal_rounds)
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(
        t,
        rmean,
        color=ALG_COLOR,
        lw=2.0,
        label=f"Rolling-mean per-round regret (window={window})",
    )
    ax.fill_between(
        t,
        rmean - rstd,
        rmean + rstd,
        color=ALG_COLOR,
        alpha=0.25,
        label=rf"$\pm 1$ s.e. over {n_seeds} seeds",
    )
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Per-round regret")
    ax.set_title(
        f"GROW-CATALOGUE: average per-round regret vs iteration "
        f"($n={n}$, $K_{{max}}={K}$, staggered-uniform adversary)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "avg_regret_per_iter.pdf"))
    fig.savefig(os.path.join(out_dir, "avg_regret_per_iter.png"), dpi=150)
    plt.close(fig)


def plot_global_avg_regret(
    mean_pr, std_pr, reveal_rounds, out_dir, n_seeds, n, K
):
    T = len(mean_pr)
    t_axis = np.arange(1, T + 1)
    # global average regret = cumulative regret / t
    cum_mean = np.cumsum(mean_pr)
    cum_std = np.sqrt(np.cumsum(std_pr ** 2))  # rough s.d. of cumulative sum
    avg_mean = cum_mean / t_axis
    avg_stderr = (cum_std / t_axis) / np.sqrt(n_seeds)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _mark_reveals(ax, reveal_rounds)
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(
        t_axis,
        avg_mean,
        color=AVG_COLOR,
        lw=2.2,
        label="Global average regret  Regret$(t)/t$",
    )
    ax.fill_between(
        t_axis,
        avg_mean - avg_stderr,
        avg_mean + avg_stderr,
        color=AVG_COLOR,
        alpha=0.25,
        label=rf"$\pm 1$ s.e. over {n_seeds} seeds",
    )
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Global average regret so far")
    ax.set_title(
        f"Global average regret vs iteration "
        f"($n={n}$, $K_{{max}}={K}$, staggered-uniform adversary)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "global_avg_regret.pdf"))
    fig.savefig(os.path.join(out_dir, "global_avg_regret.png"), dpi=150)
    plt.close(fig)


def plot_occurrence_distribution(
    type_counts_mean, type_counts_std, out_dir, K, T, n_seeds
):
    """Bar chart of the empirical frequency of each attacker type across the run."""
    labels = [rf"$\alpha_{{{i + 1}}}$" for i in range(K)]
    freqs = type_counts_mean / T
    errs = type_counts_std / T
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(
        labels,
        freqs,
        yerr=errs,
        capsize=6,
        color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"][:K],
        edgecolor="black",
    )
    # Theoretical frequencies under staggered-uniform sequence
    theoretical = []
    for i in range(K):
        # type alpha_{i+1} is active in phases i+1, i+2, ..., K
        # in phase j (j > i), it appears with prob 1/j
        # each phase has length L = T/K
        p = sum(1.0 / j for j in range(i + 1, K + 1)) * (1.0 / K)
        theoretical.append(p)
    ax.plot(
        range(K),
        theoretical,
        "k_",
        markersize=30,
        markeredgewidth=2.5,
        label="Theoretical frequency",
    )
    ax.set_ylabel("Empirical frequency in sequence")
    ax.set_xlabel("Attacker type")
    ax.set_title(
        f"Occurrence distribution of attacker types  "
        f"($K_{{max}}={K}$, $T={T}$, averaged over {n_seeds} seeds)"
    )
    for b, f in zip(bars, freqs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.003,
            f"{f:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, max(freqs.max(), max(theoretical)) * 1.25)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "occurrence_distribution.pdf"))
    fig.savefig(os.path.join(out_dir, "occurrence_distribution.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    n = 3
    K = 4
    T = 4000
    n_seeds = 40
    window = 75

    out_dir = "figures/presentation"
    os.makedirs(out_dir, exist_ok=True)

    all_per_round = np.zeros((n_seeds, T))
    type_counts = np.zeros((n_seeds, K), dtype=int)
    reveal_rounds = None
    for s in range(n_seeds):
        pr, tids, rev, _ = run_one(seed=s, n=n, K=K, T=T)
        all_per_round[s] = pr
        for i in range(K):
            type_counts[s, i] = int(np.sum(tids == i))
        if reveal_rounds is None:
            reveal_rounds = rev

    mean_pr = all_per_round.mean(axis=0)
    std_pr = all_per_round.std(axis=0)

    print("Reveal rounds:", reveal_rounds)
    print(
        "Mean per-round regret at reveal rounds:",
        [f"{mean_pr[r]:+.3f}" for r in reveal_rounds],
    )
    print(f"Mean per-round regret at t=T-1: {mean_pr[-1]:+.4f}")
    print("Type counts (mean):", type_counts.mean(axis=0))
    print("Type counts (std):", type_counts.std(axis=0))

    plot_per_iter_regret(
        mean_pr, std_pr, reveal_rounds, out_dir, window, n_seeds, n, K
    )
    plot_global_avg_regret(
        mean_pr, std_pr, reveal_rounds, out_dir, n_seeds, n, K
    )
    plot_occurrence_distribution(
        type_counts.mean(axis=0).astype(float),
        type_counts.std(axis=0),
        out_dir,
        K,
        T,
        n_seeds,
    )
    print("\nSaved:")
    print("  figures/presentation/avg_regret_per_iter.{pdf,png}")
    print("  figures/presentation/global_avg_regret.{pdf,png}")
    print("  figures/presentation/occurrence_distribution.{pdf,png}")


if __name__ == "__main__":
    main()
