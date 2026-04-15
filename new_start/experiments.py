"""Experiments for the unknown-K Stackelberg Security Game paper.

Runs:
  (A) Cumulative regret vs. T at fixed (n, K_max, arrival_pattern)
  (B) Cumulative regret vs. K_max at fixed (n, T)
  (C) Wall-clock time total-recompute vs. incremental

All results are saved to `results/*.npz` and plots to `figures/*.pdf`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm import (
    AttackerType,
    GrowCatalogue,
    HedgeLearner,
    KnownKOracle,
    SSGame,
    UniformBaseline,
    best_fixed_in_hindsight,
    compute_extreme_points,
    expected_payoff,
)


# ----------------------------------------------------------------------------
# Instance generation
# ----------------------------------------------------------------------------


def make_game(n: int, rng: np.random.Generator) -> SSGame:
    """Defender utilities: covering a target is (weakly) better than not."""
    u_d_u = rng.uniform(-1, -0.1, n)
    u_d_c = rng.uniform(0.1, 1.0, n)
    return SSGame(n=n, u_d_c=u_d_c, u_d_u=u_d_u)


def make_attacker_types(n: int, k: int, rng: np.random.Generator) -> List[AttackerType]:
    """Attacker types: attacker is hurt by covering, helped by uncovered."""
    types = []
    for i in range(k):
        u_u = rng.uniform(0.1, 1.0, n)  # uncovered is good for attacker
        u_c = rng.uniform(-1.0, -0.1, n)  # covered is bad for attacker
        types.append(AttackerType(u_c=u_c, u_u=u_u, type_id=i))
    return types


def make_sequence(
    T: int,
    types: List[AttackerType],
    pattern: str,
    rng: np.random.Generator,
) -> List[AttackerType]:
    k = len(types)
    if pattern == "uniform":
        idxs = rng.integers(0, k, size=T)
        return [types[i] for i in idxs]
    elif pattern == "slow_reveal":
        block = max(1, T // k)
        seq = []
        for a in types:
            seq.extend([a] * block)
        while len(seq) < T:
            seq.append(types[-1])
        return seq[:T]
    elif pattern == "adversarial_round_robin":
        return [types[t % k] for t in range(T)]
    else:
        raise ValueError(pattern)


# ----------------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------------


@dataclass
class RunResult:
    cum_regret_curve: np.ndarray  # shape (T,)
    final_regret: float
    wallclock: float


def run_single(
    game: SSGame,
    alg,
    sequence: List[AttackerType],
    p_star: np.ndarray,
) -> RunResult:
    T = len(sequence)
    algo_cum = np.zeros(T)
    bench_cum = np.zeros(T)
    cum_a = 0.0
    cum_b = 0.0

    start = time.perf_counter()
    for t, a in enumerate(sequence):
        # deterministic expected payoff of the algorithm's current distribution
        p_algo = expected_payoff(game, alg, a)
        p_bench = game.payoff_against(a, p_star)
        cum_a += p_algo
        cum_b += p_bench
        algo_cum[t] = cum_a
        bench_cum[t] = cum_b
        alg.observe(a)
    wall = time.perf_counter() - start

    cum_regret_curve = bench_cum - algo_cum
    return RunResult(cum_regret_curve=cum_regret_curve, final_regret=float(cum_regret_curve[-1]), wallclock=wall)


# ----------------------------------------------------------------------------
# Experiment A: regret vs T
# ----------------------------------------------------------------------------


def experiment_regret_vs_T(
    n: int,
    K_max: int,
    T_list: List[int],
    pattern: str,
    n_seeds: int,
    out_prefix: str,
) -> Dict:
    results: Dict[str, Dict[int, List[np.ndarray]]] = {
        name: {T: [] for T in T_list}
        for name in ["Grow-Total", "Grow-Incr", "Known-K", "Uniform"]
    }
    final_regrets = {name: {T: [] for T in T_list} for name in results}

    max_T = max(T_list)
    for seed in range(n_seeds):
        rng = np.random.default_rng(10 * seed + 1)
        game = make_game(n, rng)
        types = make_attacker_types(n, K_max, rng)
        full_seq = make_sequence(max_T, types, pattern, rng)
        # precompute the benchmark p* over ALL T in T_list: we use a single p*
        # per (game, seed) against the longest sequence; sub-sequences of the full
        # sequence then use the same p* (which is a valid "best single fixed" lower
        # bound for the prefix).
        p_star, _ = best_fixed_in_hindsight(game, types, full_seq)

        for T in T_list:
            seq = full_seq[:T]
            # separate algorithm instances per T so they do not share state
            alg_rng = np.random.default_rng(10 * seed + 2)
            algs = {
                "Grow-Total": GrowCatalogue(game, K_max, T, recompute="total", rng=alg_rng),
                "Grow-Incr":  GrowCatalogue(game, K_max, T, recompute="incremental", rng=alg_rng),
                "Known-K":    KnownKOracle(game, types, T, rng=alg_rng),
                "Uniform":    UniformBaseline(game, T),
            }
            for name, alg in algs.items():
                res = run_single(game, alg, seq, p_star)
                results[name][T].append(res.cum_regret_curve)
                final_regrets[name][T].append(res.final_regret)

    # Aggregate
    agg = {}
    for name, per_T in final_regrets.items():
        agg[name] = {T: (float(np.mean(vs)), float(np.std(vs))) for T, vs in per_T.items()}

    # Save
    os.makedirs("results", exist_ok=True)
    np.savez(
        f"results/{out_prefix}_regret_vs_T.npz",
        T_list=np.array(T_list),
        names=list(results.keys()),
        final_mean=np.array([[agg[n][T][0] for T in T_list] for n in results.keys()]),
        final_std=np.array([[agg[n][T][1] for T in T_list] for n in results.keys()]),
    )

    return agg


# ----------------------------------------------------------------------------
# Experiment B: regret vs K_max
# ----------------------------------------------------------------------------


def experiment_regret_vs_Kmax(
    n: int,
    K_list: List[int],
    T: int,
    pattern: str,
    n_seeds: int,
    out_prefix: str,
) -> Dict:
    agg: Dict[str, Dict[int, Tuple[float, float]]] = {
        name: {} for name in ["Grow-Total", "Grow-Incr", "Known-K", "Uniform"]
    }

    for K_max in K_list:
        per_name_runs = {name: [] for name in agg}
        for seed in range(n_seeds):
            rng = np.random.default_rng(100 * K_max + seed)
            game = make_game(n, rng)
            types = make_attacker_types(n, K_max, rng)
            seq = make_sequence(T, types, pattern, rng)
            p_star, _ = best_fixed_in_hindsight(game, types, seq)
            alg_rng = np.random.default_rng(100 * K_max + seed + 7)
            algs = {
                "Grow-Total": GrowCatalogue(game, K_max, T, recompute="total", rng=alg_rng),
                "Grow-Incr":  GrowCatalogue(game, K_max, T, recompute="incremental", rng=alg_rng),
                "Known-K":    KnownKOracle(game, types, T, rng=alg_rng),
                "Uniform":    UniformBaseline(game, T),
            }
            for name, alg in algs.items():
                res = run_single(game, alg, seq, p_star)
                per_name_runs[name].append(res.final_regret)
        for name, vs in per_name_runs.items():
            agg[name][K_max] = (float(np.mean(vs)), float(np.std(vs)))

    np.savez(
        f"results/{out_prefix}_regret_vs_Kmax.npz",
        K_list=np.array(K_list),
        names=list(agg.keys()),
        final_mean=np.array([[agg[n][K][0] for K in K_list] for n in agg.keys()]),
        final_std=np.array([[agg[n][K][1] for K in K_list] for n in agg.keys()]),
    )
    return agg


# ----------------------------------------------------------------------------
# Experiment C: wall-clock total vs incremental
# ----------------------------------------------------------------------------


def experiment_wallclock(
    n: int,
    K_max: int,
    T: int,
    n_seeds: int,
    pattern: str,
    out_prefix: str,
) -> Dict:
    times_total = []
    times_incr = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1234)
        game = make_game(n, rng)
        types = make_attacker_types(n, K_max, rng)
        seq = make_sequence(T, types, pattern, rng)
        p_star, _ = best_fixed_in_hindsight(game, types, seq)
        alg_rng = np.random.default_rng(seed + 5678)
        gt = GrowCatalogue(game, K_max, T, recompute="total", rng=alg_rng)
        gi = GrowCatalogue(game, K_max, T, recompute="incremental", rng=alg_rng)
        r_t = run_single(game, gt, seq, p_star)
        r_i = run_single(game, gi, seq, p_star)
        times_total.append(r_t.wallclock)
        times_incr.append(r_i.wallclock)

    np.savez(
        f"results/{out_prefix}_wallclock.npz",
        times_total=np.array(times_total),
        times_incr=np.array(times_incr),
    )
    return {
        "total_mean": float(np.mean(times_total)),
        "total_std": float(np.std(times_total)),
        "incr_mean": float(np.mean(times_incr)),
        "incr_std": float(np.std(times_incr)),
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


COLORS = {
    "Grow-Total": "#1f77b4",
    "Grow-Incr": "#2ca02c",
    "Known-K": "#ff7f0e",
    "Uniform": "#d62728",
}
MARKERS = {"Grow-Total": "o", "Grow-Incr": "s", "Known-K": "D", "Uniform": "^"}


def plot_regret_vs_T(agg, T_list, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["Known-K", "Grow-Total", "Grow-Incr", "Uniform"]:
        means = np.array([agg[name][T][0] for T in T_list])
        stds = np.array([agg[name][T][1] for T in T_list])
        ax.errorbar(
            T_list, means, yerr=stds, label=name,
            color=COLORS[name], marker=MARKERS[name], capsize=3, lw=1.5,
        )
    # theoretical sqrt(T) curve
    T_arr = np.array(T_list, dtype=float)
    k_known = agg["Known-K"][T_list[-1]][0]
    if k_known > 0:
        C = k_known / np.sqrt(T_arr[-1])
        ax.plot(T_list, C * np.sqrt(T_arr), "k--", lw=1, label=r"$\propto \sqrt{T}$")
    ax.set_xlabel("T (rounds)")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_regret_vs_Kmax(agg, K_list, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["Known-K", "Grow-Total", "Grow-Incr", "Uniform"]:
        means = np.array([agg[name][K][0] for K in K_list])
        stds = np.array([agg[name][K][1] for K in K_list])
        ax.errorbar(
            K_list, means, yerr=stds, label=name,
            color=COLORS[name], marker=MARKERS[name], capsize=3, lw=1.5,
        )
    # theoretical sqrt(K) curve normalized to Known-K at the largest K
    K_arr = np.array(K_list, dtype=float)
    k_known = agg["Known-K"][K_list[-1]][0]
    if k_known > 0:
        C = k_known / np.sqrt(K_arr[-1])
        ax.plot(K_list, C * np.sqrt(K_arr), "k--", lw=1, label=r"$\propto \sqrt{K_{\max}}$")
    ax.set_xlabel(r"$K_{\max}$")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_wallclock(stats, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Grow-Total", "Grow-Incr"]
    means = [stats["total_mean"], stats["incr_mean"]]
    stds = [stats["total_std"], stats["incr_std"]]
    ax.bar(labels, means, yerr=stds, capsize=6,
           color=[COLORS["Grow-Total"], COLORS["Grow-Incr"]])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -------- Experiment A: regret vs T (n=3, K_max=3, uniform arrivals) --------
    print("[A] Regret vs T  (n=3, K_max=3, uniform arrivals)")
    T_list = [100, 200, 500, 1000, 2000, 4000]
    aggA = experiment_regret_vs_T(
        n=3, K_max=3, T_list=T_list,
        pattern="uniform", n_seeds=15, out_prefix="A",
    )
    plot_regret_vs_T(
        aggA, T_list,
        out_path="figures/regret_vs_T.pdf",
        title="Regret vs T  (n=3, K_max=3, uniform adversary)",
    )
    print("  done.")

    # -------- Experiment A2: slow-reveal sequence for sanity --------
    print("[A2] Regret vs T  (n=3, K_max=3, slow-reveal adversary)")
    aggA2 = experiment_regret_vs_T(
        n=3, K_max=3, T_list=T_list,
        pattern="slow_reveal", n_seeds=15, out_prefix="A2",
    )
    plot_regret_vs_T(
        aggA2, T_list,
        out_path="figures/regret_vs_T_slow_reveal.pdf",
        title="Regret vs T  (n=3, K_max=3, slow-reveal adversary)",
    )
    print("  done.")

    # -------- Experiment B: regret vs K_max (n=3, T=2000) --------
    print("[B] Regret vs K_max  (n=3, T=2000)")
    K_list = [2, 3, 4, 5]
    aggB = experiment_regret_vs_Kmax(
        n=3, K_list=K_list, T=2000,
        pattern="uniform", n_seeds=15, out_prefix="B",
    )
    plot_regret_vs_Kmax(
        aggB, K_list,
        out_path="figures/regret_vs_kmax.pdf",
        title=r"Regret vs $K_{\max}$  (n=3, T=2000)",
    )
    print("  done.")

    # -------- Experiment C: wall-clock total vs incremental --------
    print("[C] Wall-clock Grow-Total vs Grow-Incr  (n=3, K_max=4, T=2000)")
    statsC = experiment_wallclock(
        n=3, K_max=4, T=2000, n_seeds=10,
        pattern="uniform", out_prefix="C",
    )
    plot_wallclock(
        statsC,
        out_path="figures/wall_total_vs_incr.pdf",
        title=r"Wall-clock time: Total vs Incremental ($n=3, K_{\max}=4, T=2000$)",
    )
    print("  done.")

    # -------- Summary JSON --------
    summary = {
        "A_regret_vs_T_uniform": aggA,
        "A2_regret_vs_T_slow_reveal": aggA2,
        "B_regret_vs_Kmax": aggB,
        "C_wallclock": statsC,
    }
    with open("results/summary.json", "w") as f:
        def _convert(o):
            if isinstance(o, dict):
                return {str(k): _convert(v) for k, v in o.items()}
            if isinstance(o, tuple):
                return list(o)
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o
        json.dump(_convert(summary), f, indent=2)
    print("\nAll experiments complete. Results in results/, plots in figures/.")


if __name__ == "__main__":
    main()
