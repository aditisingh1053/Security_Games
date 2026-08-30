"""Reproduce the Section 7 experiments at n = 10 and n = 20.

Figures 1 and 2 of the report were run at n = 3 because the expert set
E(C; eps) was built by exhaustive vertex enumeration.  This script reruns the
same two experiments with the oracle-efficient implementation of
`tractable.OracleGrowCatalogue`, which never materialises E(C; eps).

Produces `figures/tractable_regret.{pdf,png}`:

  (a) rolling-window per-round regret at n = 10 under the staggered-uniform
      adversary, showing the epoch structure of the proof of Theorem 6 (one
      spike per discovery round) -- this is Figure 1 at n = 10;
  (b) the same at n = 20;
  (c) cumulative regret at n = 10 under the uniform-random adversary, with the
      C sqrt(t) envelope -- this is Figure 2 at n = 10;
  (d) the same at n = 20.

Panels (a), (b) and panels (c), (d) use the two adversaries of Section 7 for
the same reason Figures 1 and 2 do.  The staggered-uniform adversary spreads
the discovery rounds across the horizon and so exhibits the epoch structure,
but it is non-stationary, so the best *fixed* strategy in hindsight is a weak
comparator and cumulative regret against it is not informative.  The
uniform-random adversary is stationary and gives the scaling picture.

Every benchmark p* below is itself a single oracle call: the best fixed
strategy in hindsight is the Bayesian-Stackelberg optimum against the
realised type frequencies, so even *evaluating* regret used to be exponential
in n and is now one MILP.

Run from `code/`:  python plot_tractable_regret.py [--fast]
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm import AttackerType, SSGame
from oracle import BestResponseOracle, instance_from_game
from tractable import OracleGrowCatalogue

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def make_game(n, rng):
    return SSGame(n=n, u_d_c=rng.uniform(0.1, 1.0, n), u_d_u=rng.uniform(-1.0, -0.1, n))


def make_types(n, K, rng):
    return [AttackerType(u_c=rng.uniform(-1.0, -0.1, n), u_u=rng.uniform(0.1, 1.0, n),
                         type_id=i) for i in range(K)]


def staggered_uniform(T, types, rng):
    """The staggered-uniform adversary of Section 7.1: phase i draws uniformly
    from the first i types, so discovery rounds fall at t = 0, T/K, 2T/K, ..."""
    K = len(types)
    L = T // K
    seq = []
    for i in range(1, K + 1):
        m = L if i < K else T - (i - 1) * L
        seq.extend(types[j] for j in rng.integers(0, i, size=m))
    return seq


def uniform_random(T, types, rng):
    """The uniform-random adversary of Section 7.2: a_t is drawn uniformly
    from all K types, independently across rounds."""
    K = len(types)
    return [types[int(rng.integers(0, K))] for _ in range(T)]


def best_fixed(game, types, seq):
    """p* of Equation (3), by one call to the Bayesian-Stackelberg oracle."""
    inst = instance_from_game(game, types)
    index = {a.type_id: k for k, a in enumerate(types)}
    counts = np.zeros(len(types))
    for a in seq:
        counts[index[a.type_id]] += 1.0
    p, _, _ = BestResponseOracle(inst, backend="milp").maximize_weighted_payoff(counts)
    return p


def run(args):
    seed, n, K, T, adversary = args
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    gen = staggered_uniform if adversary == "staggered" else uniform_random
    seq = gen(T, types, np.random.default_rng(seed + 7777))
    p_star = best_fixed(game, types, seq)
    alg = OracleGrowCatalogue(game, K_max=K, T=T, rng=np.random.default_rng(seed + 999))
    uni = np.full(n, 1.0 / n)

    reg = np.zeros(T)
    reg_uni = np.zeros(T)
    for t, a in enumerate(seq):
        p = alg.play()
        bench = game.payoff_against(a, p_star)
        reg[t] = bench - game.payoff_against(a, p)
        reg_uni[t] = bench - game.payoff_against(a, uni)
        alg.observe(a)
    return reg, reg_uni


def average(pool, seeds, n, K, T, adversary):
    t0 = time.perf_counter()
    out = pool.map(run, [(s, n, K, T, adversary) for s in seeds])
    for s, (r, u) in zip(seeds, out):
        print(f"    {adversary:9s} n={n:2d} seed {s}: cum regret {r.sum():9.2f}"
              f"   uniform {u.sum():9.2f}", flush=True)
    print(f"    ({time.perf_counter()-t0:.1f}s wall)", flush=True)
    return np.mean([o[0] for o in out], axis=0), np.mean([o[1] for o in out], axis=0)


def panel_per_round(ax, per_round, K, T, title, window):
    ker = np.ones(window) / window
    smooth = np.convolve(per_round, ker, mode="same")
    ts = np.arange(1, T + 1)
    for i in range(K):
        ax.axvline(i * (T // K) + 1, color="tab:red", ls="--", lw=0.8,
                   label="phase boundary" if i == 0 else None)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.plot(ts, smooth, color="tab:green", lw=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("round $t$", fontsize=8)
    ax.set_ylabel("per-round regret", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.25)


def panel_cumulative(ax, per_round, per_round_uni, title):
    T = len(per_round)
    ts = np.arange(1, T + 1)
    cum = np.cumsum(per_round)
    cum_uni = np.cumsum(per_round_uni)
    C = max(0.0, float(np.max(cum / np.sqrt(ts))))
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.plot(ts, cum_uni, color="tab:red", lw=1.1, ls=":", label="uniform baseline")
    ax.plot(ts, C * np.sqrt(ts), color="tab:orange", lw=1.1, ls="--",
            label=rf"$C\sqrt{{t}}$ envelope, $C={C:.2f}$")
    ax.plot(ts, cum, color="tab:green", lw=1.7, label=r"\textsc{Oracle-Grow-Cat.}"
            if plt.rcParams["text.usetex"] else "Oracle-Grow-Catalogue")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("round $t$", fontsize=8)
    ax.set_ylabel("cumulative regret", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from figures/tractable_regret_curves.npz")
    args = ap.parse_args()
    K = 4
    if args.fast:
        T, seeds, window = 200, [0, 1], 21
    else:
        T, seeds, window = 1500, [0, 1, 2, 3], 101

    if args.replot:
        d = np.load(os.path.join(OUT_DIR, "tractable_regret_curves.npz"))
        a_pr, a_uni, b_pr, b_uni = d["a_pr"], d["a_uni"], d["b_pr"], d["b_uni"]
        c_pr, c_uni, d_pr, d_uni = d["c_pr"], d["c_uni"], d["d_pr"], d["d_uni"]
        T, K = int(d["T"]), int(d["K"])
        window = 101 if T > 500 else 21
    else:
        with Pool(processes=min(4 * len(seeds), os.cpu_count())) as pool:
            print("(a) staggered-uniform, n = 10", flush=True)
            a_pr, a_uni = average(pool, seeds, 10, K, T, "staggered")
            print("(b) staggered-uniform, n = 20", flush=True)
            b_pr, b_uni = average(pool, seeds, 20, K, T, "staggered")
            print("(c) uniform-random, n = 10", flush=True)
            c_pr, c_uni = average(pool, seeds, 10, K, T, "uniform")
            print("(d) uniform-random, n = 20", flush=True)
            d_pr, d_uni = average(pool, seeds, 20, K, T, "uniform")
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(os.path.join(OUT_DIR, "tractable_regret_curves.npz"),
                 a_pr=a_pr, a_uni=a_uni, b_pr=b_pr, b_uni=b_uni,
                 c_pr=c_pr, c_uni=c_uni, d_pr=d_pr, d_uni=d_uni, T=T, K=K)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    panel_per_round(axes[0, 0], a_pr, K, T, r"(a) $n=10$, $K_{\max}=4$: per-round regret", window)
    panel_per_round(axes[0, 1], b_pr, K, T, r"(b) $n=20$, $K_{\max}=4$: per-round regret", window)
    panel_cumulative(axes[1, 0], c_pr, c_uni,
                     r"(c) $n=10$, uniform-random: cumulative regret")
    panel_cumulative(axes[1, 1], d_pr, d_uni,
                     r"(d) $n=20$, uniform-random: cumulative regret")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"tractable_regret.{ext}"), dpi=180)
    print("wrote figures/tractable_regret.{pdf,png}")

    for name, pr, un in (("stag. n=10", a_pr, a_uni), ("stag. n=20", b_pr, b_uni),
                         ("unif. n=10", c_pr, c_uni), ("unif. n=20", d_pr, d_uni)):
        c, u = pr.sum(), un.sum()
        print(f"  {name}: algorithm {c:9.2f}   uniform {u:9.2f}   "
              f"ratio {(u/c if c > 1e-6 else float('inf')):7.1f}x   "
              f"C = {np.max(np.cumsum(pr)/np.sqrt(np.arange(1,T+1))):.3f}")


if __name__ == "__main__":
    main()
