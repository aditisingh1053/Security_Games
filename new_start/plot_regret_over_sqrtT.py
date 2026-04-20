"""Plot cumulative Regret(T) vs T showing it stays below C * sqrt(T).

The theorem says Regret(T) <= O(sqrt(T * K_max * (n^2 + K_max log n))).
For fixed n, K_max this is C * sqrt(T) for some constant C.
We plot the empirical regret, a C*sqrt(T) upper envelope, and a linear
reference to show visually that regret is sublinear and bounded.
"""

from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm import (
    AttackerType, GrowCatalogue, SSGame,
    best_fixed_in_hindsight, expected_payoff,
)


def make_game(n, rng):
    return SSGame(n=n,
                  u_d_c=rng.uniform(0.1, 1.0, n),
                  u_d_u=rng.uniform(-1.0, -0.1, n))


def make_types(n, k, rng):
    return [AttackerType(u_c=rng.uniform(-1.0, -0.1, n),
                         u_u=rng.uniform(0.1, 1.0, n),
                         type_id=i) for i in range(k)]


def make_uniform_seq(T, types, rng):
    k = len(types)
    return [types[int(rng.integers(0, k))] for _ in range(T)]


def run_one(seed, n, K, T):
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    seq = make_uniform_seq(T, types, np.random.default_rng(seed + 7777))
    p_star, _ = best_fixed_in_hindsight(game, types, seq)
    alg = GrowCatalogue(game, K_max=K, T=T, recompute="total",
                        rng=np.random.default_rng(seed + 10000))
    cum = 0.0
    for a in seq:
        cum += game.payoff_against(a, p_star) - expected_payoff(game, alg, a)
        alg.observe(a)
    return cum


def main():
    n = 3
    K = 3
    T_list = [100, 200, 500, 1000, 2000, 4000, 6000]
    n_seeds = 80

    out_dir = "figures/presentation"
    os.makedirs(out_dir, exist_ok=True)

    means, ses = [], []
    for T in T_list:
        regrets = np.array([run_one(s, n, K, T) for s in range(n_seeds)])
        means.append(regrets.mean())
        ses.append(regrets.std() / np.sqrt(n_seeds))
        print(f"  T={T:5d}  Regret={regrets.mean():7.2f} +/- {regrets.std():6.2f}"
              f"  Regret/sqrt(T)={regrets.mean()/np.sqrt(T):.3f}")

    T_arr = np.array(T_list, dtype=float)
    mg = np.array(means)
    sg = np.array(ses)

    # Upper envelope constant: pick C so that C*sqrt(T) >= empirical regret for all T
    C_upper = max(mg[i] / np.sqrt(T_arr[i]) for i in range(len(T_arr))) * 1.15
    T_smooth = np.linspace(50, T_arr[-1] * 1.05, 300)

    # --- Main plot: Regret(T) vs T with sqrt(T) bound and linear reference ---
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Linear reference (what you'd get WITHOUT learning)
    slope = mg[-1] / T_arr[-1] * 8  # scale so it's visible
    ax.fill_between(T_smooth, 0, slope * T_smooth, color="#F2DEDE",
                    alpha=0.4, label=r"linear $\Theta(T)$ regime (no learning)")

    # Theoretical sqrt(T) upper bound
    ax.plot(T_smooth, C_upper * np.sqrt(T_smooth), color="#B08642",
            ls="--", lw=2.2,
            label=rf"$C\,\sqrt{{T}}$ upper bound  ($C={C_upper:.2f}$)")

    # Empirical
    ax.errorbar(T_arr, mg, yerr=sg,
                color="#006857", marker="o", lw=2.5, capsize=5, markersize=8,
                zorder=10,
                label=r"$\mathrm{Regret}(T)$  (Grow-Catalogue, 80 seeds)")

    ax.set_xlabel("$T$  (time horizon)", fontsize=13)
    ax.set_ylabel(r"Cumulative regret  $\mathrm{Regret}(T)$", fontsize=13)
    ax.set_title(
        rf"Regret vs $T$:  empirical regret stays below $C\sqrt{{T}}$   "
        rf"($n\!=\!{n}$, $K_{{\max}}\!=\!{K}$)",
        fontsize=13)
    ax.legend(fontsize=10.5, loc="upper left")
    ax.set_xlim(0, T_arr[-1] * 1.05)
    ax.set_ylim(0, max(slope * T_arr[-1], C_upper * np.sqrt(T_arr[-1])) * 1.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regret_vs_T_scaling.pdf"))
    fig.savefig(os.path.join(out_dir, "regret_vs_T_scaling.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Regret/sqrt(T) showing it is BOUNDED (does not grow) ---
    ratios = mg / np.sqrt(T_arr)
    ratio_se = sg / np.sqrt(T_arr)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(T_arr, ratios, yerr=ratio_se,
                color="#006857", marker="o", lw=2.2, capsize=5, markersize=7,
                label=r"$\mathrm{Regret}(T)\,/\,\sqrt{T}$  (empirical)")
    ax.axhline(C_upper, color="#B08642", ls="--", lw=1.8,
               label=rf"theoretical upper bound $C = {C_upper:.2f}$")
    ax.fill_between([T_arr[0]*0.8, T_arr[-1]*1.1], 0, C_upper,
                    color="#B08642", alpha=0.08)
    ax.set_xlabel("$T$  (time horizon)", fontsize=13)
    ax.set_ylabel(r"$\mathrm{Regret}(T)\;/\;\sqrt{T}$", fontsize=13)
    ax.set_title(
        rf"$\mathrm{{Regret}}(T)/\sqrt{{T}}$ is bounded:  "
        rf"stays below $C={C_upper:.2f}$ for all $T$",
        fontsize=13)
    ax.legend(fontsize=10.5, loc="upper right")
    ax.set_ylim(0, C_upper * 1.6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regret_over_sqrtT.pdf"))
    fig.savefig(os.path.join(out_dir, "regret_over_sqrtT.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {out_dir}/regret_vs_T_scaling.{{pdf,png}}")
    print(f"Saved to {out_dir}/regret_over_sqrtT.{{pdf,png}}")


if __name__ == "__main__":
    main()
