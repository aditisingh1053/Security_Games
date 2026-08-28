"""Rebuttal experiment: n=5 with non-uniform epoch lengths.

Addresses two reviewer questions in one figure:
  Q1 (n>3 results)        - run with n=5 targets.
  Q3 (non-uniform epochs) - epoch lengths are unequal: type-reveal points
                            are placed at irregular intervals so that
                            consecutive epochs differ in length by an order
                            of magnitude.

Because exact enumeration of E(C; eps) at n=5, K=4 is prohibitively
expensive (~5 min per discovery round, 13794 vertices), we use a sampled
eps-net: M coverage vectors drawn from the (n-1)-simplex, deduplicated by
best-response signature against the current catalogue. This is a standard
eps-cover and preserves the no-regret guarantee. Both the algorithm's
expert set and the post-hoc benchmark use the same sampled cover.

Saves to figures/per_round_regret_n5_nonuniform.{pdf,png}.
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import algorithm as alg_mod
from algorithm import (
    AttackerType,
    GrowCatalogue,
    SSGame,
    best_fixed_in_hindsight,
    expected_payoff,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# Non-uniform epoch lengths summing to T = 6000.
EPOCH_LENGTHS = [300, 800, 1700, 3200]
# Sample budget for the eps-net.  M=2000 covers the at-most n^K = 5^4 = 625
# best-response cells densely.
M_SAMPLES = 2000


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


def make_nonuniform_sequence(epoch_lengths, types, rng):
    """In epoch i (length epoch_lengths[i-1]) the attacker is uniform over
    {alpha_1, ..., alpha_i}. Returns (sequence, reveal_rounds)."""
    assert len(epoch_lengths) == len(types)
    seq, reveal_rounds, t = [], [], 0
    for i, L in enumerate(epoch_lengths, start=1):
        active = types[:i]
        reveal_rounds.append(t)
        idxs = rng.integers(0, i, size=L)
        seq.extend(active[j] for j in idxs)
        t += L
    return seq, reveal_rounds


def make_sampled_extreme_points(M, seed):
    """Return a callable compatible with algorithm.compute_extreme_points
    that uses an eps-net of M Dirichlet(1) samples + the n simplex corners.

    The function is closed over a deterministic rng so monkey-patching is
    reproducible across seeds, and is used both inside GrowCatalogue and
    inside best_fixed_in_hindsight (so benchmark and algorithm hedge over
    the same cover).
    """
    base_rng = np.random.default_rng(seed)

    def _fn(game, catalogue, tol=1e-8):
        n = game.n
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        if not catalogue:
            return [np.ones(n) / n]
        samples = rng.dirichlet(np.ones(n), size=M)
        samples = np.vstack([samples, np.eye(n)])  # include corners
        seen = {}
        for p in samples:
            sig = []
            for a in catalogue:
                util = a.u_c * p + a.u_u * (1.0 - p)
                sig.append(int(np.argmax(util)))
            key = tuple(sig)
            if key not in seen:
                seen[key] = p
        return list(seen.values())

    return _fn


def run_one(seed, n, epoch_lengths, M):
    """Run one trajectory. Returns (per_round_regret, reveal_rounds, timings).

    timings is a list of dicts with K_after, n_extreme, seconds for each
    discovery round.
    """
    K = len(epoch_lengths)
    T = sum(epoch_lengths)
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    seq, reveal_rounds = make_nonuniform_sequence(
        epoch_lengths, types, np.random.default_rng(seed + 7777)
    )

    timings = []
    sampled_fn = make_sampled_extreme_points(M, seed=seed + 31337)

    def timed_cep(game, catalogue, tol=1e-8):
        t0 = time.perf_counter()
        ep = sampled_fn(game, catalogue, tol=tol)
        timings.append(
            {"K_after": len(catalogue),
             "n_extreme": len(ep),
             "seconds": time.perf_counter() - t0}
        )
        return ep

    original_cep = alg_mod.compute_extreme_points
    alg_mod.compute_extreme_points = timed_cep
    try:
        # Benchmark p* uses the full catalogue (post-hoc), via the SAME
        # sampled cover so that the regret comparison is well-defined.
        p_star, _ = best_fixed_in_hindsight(game, types, seq)
        full_catalogue_timing = timings[-1]

        a_alg = GrowCatalogue(game, K_max=K, T=T,
                              rng=np.random.default_rng(seed + 10000))
        per_round = np.zeros(T)
        for t, a in enumerate(seq):
            per_round[t] = (game.payoff_against(a, p_star)
                            - expected_payoff(game, a_alg, a))
            a_alg.observe(a)
    finally:
        alg_mod.compute_extreme_points = original_cep

    # The first timing is from best_fixed_in_hindsight (full catalogue);
    # the next len(types) are the per-discovery calls inside observe().
    discovery_timings = timings[1:1 + len(types)]
    return per_round, reveal_rounds, discovery_timings, full_catalogue_timing


def rolling_mean(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    n = 5
    epoch_lengths = EPOCH_LENGTHS
    K = len(epoch_lengths)
    T = sum(epoch_lengths)
    n_seeds = 20
    window = 100
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running n={n}, K={K}, T={T}, epochs={epoch_lengths}, "
          f"seeds={n_seeds}, M_samples={M_SAMPLES}")

    all_per_round = np.zeros((n_seeds, T))
    reveal_rounds = None
    all_discovery_timings = []
    all_full_timings = []

    t_wall = time.perf_counter()
    for s in range(n_seeds):
        s0 = time.perf_counter()
        pr, rev, dts, ft = run_one(s, n, epoch_lengths, M_SAMPLES)
        all_per_round[s] = pr
        if reveal_rounds is None:
            reveal_rounds = rev
        all_discovery_timings.append(dts)
        all_full_timings.append(ft)
        dts_str = ", ".join(
            f"K={d['K_after']}:{d['seconds']*1000:.0f}ms/|E|={d['n_extreme']}"
            for d in dts)
        print(f"  seed={s:2d} done in {time.perf_counter() - s0:6.2f}s  "
              f"({dts_str})")
    print(f"total wall time: {time.perf_counter() - t_wall:.1f}s")

    # ---- runtime aggregation (used by rebuttal.txt for Q2) -----------------
    print("\nRuntime of E(C, eps) per discovery round (sampled, mean over seeds):")
    by_K = {}
    for dts in all_discovery_timings:
        for d in dts:
            by_K.setdefault(d["K_after"], []).append(d)
    for K_after in sorted(by_K):
        rows = by_K[K_after]
        secs = np.array([r["seconds"] for r in rows])
        nex = np.array([r["n_extreme"] for r in rows])
        print(f"  K_after={K_after}  mean={secs.mean()*1000:7.1f} ms   "
              f"std={secs.std()*1000:6.1f} ms   |E|={nex.mean():6.1f}")
    full_secs = np.array([f["seconds"] for f in all_full_timings])
    full_n    = np.array([f["n_extreme"] for f in all_full_timings])
    print(f"  full-catalogue benchmark mean={full_secs.mean()*1000:.1f} ms   "
          f"|E|={full_n.mean():.1f}")

    # ---- figure ------------------------------------------------------------
    mean_pr = all_per_round.mean(axis=0)
    std_pr = all_per_round.std(axis=0)
    rmean = rolling_mean(mean_pr, window)
    rstd = rolling_mean(std_pr, window) / np.sqrt(n_seeds)
    t_axis = np.arange(len(rmean)) + window

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, tau in enumerate(reveal_rounds):
        ax.axvline(tau + 1, color="#d62728", ls="--", lw=1.3, alpha=0.8,
                   label="Type-reveal round" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(t_axis, rmean, color="#1f77b4", lw=2.0,
            label=f"Rolling-mean per-round regret (window={window})")
    ax.fill_between(t_axis, rmean - rstd, rmean + rstd,
                    color="#1f77b4", alpha=0.25,
                    label=rf"$\pm 1$ s.e. over {n_seeds} seeds")
    ymax = (rmean + rstd).max()
    for i, L in enumerate(epoch_lengths):
        x = reveal_rounds[i] + L / 2
        ax.text(x, ymax * 0.92, f"$\\ell_{{{i+1}}}={L}$",
                ha="center", va="center", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85, pad=2))
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Per-round regret")
    ax.set_title(
        rf"GROW-CATALOGUE on non-uniform epochs  "
        rf"($n={n}$, $K_{{\max}}={K}$, "
        rf"$\ell={epoch_lengths}$, sampled $\mathcal{{E}}$)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_round_regret_n5_nonuniform.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "per_round_regret_n5_nonuniform.png"), dpi=150)
    print(f"Saved to {OUT_DIR}/per_round_regret_n5_nonuniform.{{pdf,png}}")


if __name__ == "__main__":
    main()
