"""Experiments for BLOCK-GROW-CATALOGUE (Section 10).

Produces `figures/block_revelation.{pdf,png}`:

  (a) cumulative regret at n = 10 against the C t^{2/3} envelope of Theorem 28;
  (b) the same at n = 20;
  (c) final regret as a function of the revelation granularity B, whose
      worst-case growth is the 2 K' B additive term of Theorem 28;
  (d) full information (Theorem 6) against block revelation (Theorem 28) as
      T grows, on a log-log scale.

Panels (a)-(c) use the staggered-uniform adversary of Section 7, which
spreads the discovery blocks across the horizon and so exercises the epoch
decomposition of Definition 25.  Panel (d) uses the stationary uniform-random
adversary, because it compares two algorithms against the best *fixed*
strategy in hindsight and that comparator is only meaningful for a stationary
sequence.

Feedback is partial information throughout: the simulator hands the algorithm
the attacked target only, and calls `checkpoint` with the *set* of types that
have attacked so far at the end of every revelation block.

Run from `code/`:  python plot_block_revelation.py [--fast]
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
from tractable import BlockGrowCatalogue, OracleGrowCatalogue

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def make_instance(seed, n, K, T, adversary="staggered"):
    rng = np.random.default_rng(seed)
    game = SSGame(n=n, u_d_c=rng.uniform(0.1, 1.0, n), u_d_u=rng.uniform(-1.0, -0.1, n))
    types = [AttackerType(u_c=rng.uniform(-1.0, -0.1, n), u_u=rng.uniform(0.1, 1.0, n),
                          type_id=i) for i in range(K)]
    srng = np.random.default_rng(seed + 7777)
    seq = []
    if adversary == "staggered":
        L = T // K
        for i in range(1, K + 1):
            m = L if i < K else T - (i - 1) * L
            seq.extend(types[j] for j in srng.integers(0, i, size=m))
    else:
        seq = [types[int(srng.integers(0, K))] for _ in range(T)]
    inst = instance_from_game(game, types)
    counts = np.zeros(K)
    index = {a.type_id: k for k, a in enumerate(types)}
    for a in seq:
        counts[index[a.type_id]] += 1.0
    p_star, _, _ = BestResponseOracle(inst).maximize_weighted_payoff(counts)
    return game, types, seq, p_star


def run_block(args):
    seed, n, K, T, B, adversary = args
    game, types, seq, p_star = make_instance(seed, n, K, T, adversary)
    alg = BlockGrowCatalogue(game, K_max=K, T=T, B=B, rng=np.random.default_rng(seed + 31))
    uni = np.full(n, 1.0 / n)
    revealed, reg, reg_uni = [], np.zeros(T), np.zeros(T)
    for t, a in enumerate(seq):
        p = alg.play()
        i = a.best_response(p)                        # the attacked target
        bench = game.payoff_against(a, p_star)
        reg[t] = bench - game.defender_util_given_target(i, p)
        reg_uni[t] = bench - game.payoff_against(a, uni)
        if a not in revealed:
            revealed.append(a)
        alg.observe_target(i)                         # partial information
        if (t + 1) % B == 0 or t == T - 1:
            alg.checkpoint(list(revealed))            # the set, and nothing else
    return reg, reg_uni


def run_full(args):
    seed, n, K, T, adversary = args
    game, types, seq, p_star = make_instance(seed, n, K, T, adversary)
    alg = OracleGrowCatalogue(game, K_max=K, T=T, rng=np.random.default_rng(seed + 31))
    reg = np.zeros(T)
    for t, a in enumerate(seq):
        p = alg.play()
        reg[t] = game.payoff_against(a, p_star) - game.payoff_against(a, p)
        alg.observe(a)
    return reg


def average(pool, fn, tasks, tag=""):
    t0 = time.perf_counter()
    out = pool.map(fn, tasks)
    if isinstance(out[0], tuple):
        res = (np.mean([o[0] for o in out], axis=0), np.mean([o[1] for o in out], axis=0))
        tot = res[0].sum()
    else:
        res = np.mean(out, axis=0)
        tot = res.sum()
    print(f"    {tag}: regret {tot:9.2f}   ({time.perf_counter()-t0:6.1f}s wall)",
          flush=True)
    return res


def panel_cumulative(ax, per_round, per_round_uni, title):
    T = len(per_round)
    ts = np.arange(1, T + 1)
    cum, cum_uni = np.cumsum(per_round), np.cumsum(per_round_uni)
    C = max(0.0, float(np.max(cum / ts ** (2.0 / 3.0))))
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.plot(ts, cum_uni, color="tab:red", lw=1.1, ls=":", label="uniform baseline")
    ax.plot(ts, C * ts ** (2.0 / 3.0), color="tab:orange", lw=1.1, ls="--",
            label=rf"$C t^{{2/3}}$ envelope, $C={C:.2f}$")
    ax.plot(ts, cum, color="tab:blue", lw=1.7, label="Block-Grow-Catalogue")
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
                    help="redraw from figures/block_revelation_curves.npz")
    args = ap.parse_args()
    K, n_a, n_b = 4, 10, 20
    if args.fast:
        T, seeds, Bs, Ts = 300, [0, 1], [5, 20, 80], [100, 200, 300]
    else:
        T, seeds, Bs = 2000, [0, 1, 2, 3], [4, 8, 16, 32, 64, 128, 256]
        Ts = [250, 500, 1000, 2000]

    if args.replot:
        d = np.load(os.path.join(OUT_DIR, "block_revelation_curves.npz"))
        a_pr, a_uni, b_pr, b_uni = d["a_pr"], d["a_uni"], d["b_pr"], d["b_uni"]
        Bs, curve_B = list(d["Bs"]), list(d["curve_B"])
        Ts, full_curve, block_curve = list(d["Ts"]), list(d["full_curve"]), list(d["block_curve"])
        T, K, B0 = int(d["T"]), int(d["K"]), int(d["B0"])
    else:
        B0 = max(4, int(round(T ** (2.0 / 3.0) / 4)))
        with Pool(processes=min(4 * len(seeds), os.cpu_count())) as pool:
            print("(a) Block-Grow-Catalogue, n = 10", flush=True)
            a_pr, a_uni = average(pool, run_block,
                                  [(s_, n_a, K, T, B0, "staggered") for s_ in seeds],
                                  f"n=10 T={T} B={B0}")
            print("(b) Block-Grow-Catalogue, n = 20", flush=True)
            b_pr, b_uni = average(pool, run_block,
                                  [(s_, n_b, K, T, B0, "staggered") for s_ in seeds],
                                  f"n=20 T={T} B={B0}")

            print("(c) regret versus the revelation granularity B, n = 10", flush=True)
            curve_B = []
            for B in Bs:
                pr, _ = average(pool, run_block,
                                [(s_, n_a, K, T, B, "staggered") for s_ in seeds],
                                f"B={B}")
                curve_B.append(float(pr.sum()))

            print("(d) full information versus block revelation, n = 10", flush=True)
            full_curve, block_curve = [], []
            for Tk in Ts:
                Bk = max(4, int(round(Tk ** (2.0 / 3.0) / 4)))
                f = average(pool, run_full,
                            [(s_, n_a, K, Tk, "uniform") for s_ in seeds],
                            f"full  T={Tk}")
                pr, _ = average(pool, run_block,
                                [(s_, n_a, K, Tk, Bk, "uniform") for s_ in seeds],
                                f"block T={Tk} B={Bk}")
                full_curve.append(float(f.sum()))
                block_curve.append(float(pr.sum()))

        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(os.path.join(OUT_DIR, "block_revelation_curves.npz"),
                 a_pr=a_pr, a_uni=a_uni, b_pr=b_pr, b_uni=b_uni,
                 Bs=np.array(Bs), curve_B=np.array(curve_B), Ts=np.array(Ts),
                 full_curve=np.array(full_curve), block_curve=np.array(block_curve),
                 T=T, K=K, B0=B0)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    panel_cumulative(axes[0, 0], a_pr, a_uni,
                     rf"(a) $n=10$, $K_{{\max}}=4$, $B={B0}$")
    panel_cumulative(axes[0, 1], b_pr, b_uni,
                     rf"(b) $n=20$, $K_{{\max}}=4$, $B={B0}$")

    ax = axes[1, 0]
    Bs_arr, cb = np.array(Bs, dtype=float), np.array(curve_B)
    ax.plot(Bs_arr, cb, "o-", color="tab:blue", lw=1.6, ms=4, label="measured regret")
    slope_B, icpt_B = np.polyfit(Bs_arr, cb, 1)
    ax.plot(Bs_arr, icpt_B + slope_B * Bs_arr, color="tab:orange", ls="--", lw=1.1,
            label=rf"linear fit, slope ${slope_B:.2f}$")
    ax.plot(Bs_arr, cb[0] + 2.0 * K * (Bs_arr - Bs_arr[0]), color="0.45", ls=":",
            lw=1.1, label=rf"worst case, slope $2K'={2*K}$")
    ax.set_ylim(0.0, max(float(cb.max()), cb[0] + 2.0 * K * (Bs_arr[-1] - Bs_arr[0])) * 1.05)
    ax.set_title(r"(c) $n=10$: regret versus revelation granularity $B$", fontsize=9)
    ax.set_xlabel("$B$", fontsize=8); ax.set_ylabel("cumulative regret", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    Ts_arr = np.array(Ts, dtype=float)
    ax.loglog(Ts_arr, full_curve, "s-", color="tab:green", lw=1.6, ms=4,
              label="full information (Thm. 6)")
    ax.loglog(Ts_arr, block_curve, "o-", color="tab:blue", lw=1.6, ms=4,
              label="block revelation (Thm. 28)")
    ax.loglog(Ts_arr, full_curve[0] * (Ts_arr / Ts_arr[0]) ** 0.5, color="tab:green",
              ls="--", lw=0.9, label=r"slope $1/2$")
    ax.loglog(Ts_arr, block_curve[0] * (Ts_arr / Ts_arr[0]) ** (2.0 / 3.0), color="tab:blue",
              ls="--", lw=0.9, label=r"slope $2/3$")
    ax.set_title(r"(d) $n=10$: regret growth, $\sqrt{T}$ versus $T^{2/3}$", fontsize=9)
    ax.set_xlabel("$T$", fontsize=8); ax.set_ylabel("cumulative regret", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"block_revelation.{ext}"), dpi=180)
    print("wrote figures/block_revelation.{pdf,png}")

    def slope(xs, ys):
        return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    print(f"\nfitted log-log slopes: full {slope(Ts_arr, full_curve):.3f}"
          f"   block {slope(Ts_arr, block_curve):.3f}")
    print(f"regret-versus-B linear fit: slope {slope_B:.4f} "
          f"(worst case 2K' = {2*K}); measured range "
          f"{cb.min():.2f} .. {cb.max():.2f} over B in [{Bs[0]}, {Bs[-1]}]")


if __name__ == "__main__":
    main()
