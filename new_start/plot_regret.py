"""Single plot: average per-round regret vs iteration for GROW-CATALOGUE,
with clearly visible spikes at rounds where a new attacker type is discovered.

We plot a *rolling-window average* of the per-round regret
    r_t  :=  U_d(b_{a_t}(p*), p*) - E[U_d(b_{a_t}(p_t), p_t)]
averaged over many random games. This makes the learning curve between
discoveries visible AND the spike at each discovery clearly visible.
"""

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


def make_slow_reveal(T, types):
    k = len(types)
    block = T // k
    seq = []
    for a in types:
        seq.extend([a] * block)
    while len(seq) < T:
        seq.append(types[-1])
    return seq[:T]


def run_one(seed, n, K, T):
    rng = np.random.default_rng(seed)
    game = make_game(n, rng)
    types = make_types(n, K, rng)
    seq = make_slow_reveal(T, types)

    p_star, _ = best_fixed_in_hindsight(game, types, seq)

    alg = GrowCatalogue(game, K_max=K, T=T, recompute="total",
                        rng=np.random.default_rng(seed + 10000))

    per_round = np.zeros(T)
    discovery_rounds = []
    seen = set()
    for t, a in enumerate(seq):
        p_algo_exp = expected_payoff(game, alg, a)
        p_bench = game.payoff_against(a, p_star)
        per_round[t] = p_bench - p_algo_exp
        if a.type_id not in seen:
            seen.add(a.type_id)
            discovery_rounds.append(t)
        alg.observe(a)
    return per_round, discovery_rounds


def rolling_mean(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    n = 3
    K = 4          # K_max = 4 types  ->  clear spikes at 4 discovery rounds
    T = 4000
    n_seeds = 40
    window = 50    # rolling window for visibility of local regret behaviour

    all_per_round = np.zeros((n_seeds, T))
    disc_all = None
    for s in range(n_seeds):
        pr, disc = run_one(seed=s, n=n, K=K, T=T)
        all_per_round[s] = pr
        if disc_all is None:
            disc_all = disc  # deterministic given slow-reveal

    mean_pr = all_per_round.mean(axis=0)   # shape (T,)
    std_pr = all_per_round.std(axis=0)

    rmean = rolling_mean(mean_pr, window)
    rstd = rolling_mean(std_pr, window) / np.sqrt(n_seeds)  # SE of rolling avg
    r_t = np.arange(len(rmean)) + window  # x-axis: right edge of window

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # Mark discoveries first (so lines go behind the regret curve)
    for i, tau in enumerate(disc_all):
        ax.axvline(tau + 1, color="#d62728", ls="--", lw=1.3, alpha=0.8,
                   label="New attacker type discovered" if i == 0 else None)
        ax.text(tau + 1, 0.92, f" discovery #{i+1}",
                color="#d62728", transform=ax.get_xaxis_transform(),
                fontsize=9, rotation=90, va="top", ha="right")

    ax.axhline(0, color="black", lw=0.6)
    ax.plot(r_t, rmean, color="#1f77b4", lw=2.0,
            label=f"Rolling-mean per-round regret (window={window})")
    ax.fill_between(r_t, rmean - rstd, rmean + rstd,
                    color="#1f77b4", alpha=0.25, label=r"$\pm 1$ s.e. over 40 seeds")

    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"Per-round regret  $U_d(b_{a_t}(p^\ast),p^\ast) - \mathbb{E}[U_d(b_{a_t}(p_t),p_t)]$")
    ax.set_title(f"GROW-CATALOGUE: average regret vs iteration  "
                 f"($n={n}$, $K_{{max}}={K}$, slow-reveal adversary)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig("figures/avg_regret_vs_iter.pdf")
    fig.savefig("figures/avg_regret_vs_iter.png", dpi=150)

    print(f"Discovery rounds (slow-reveal): {disc_all}")
    print(f"Mean per-round regret at discovery rounds (instantaneous):")
    for tau in disc_all:
        print(f"  t={tau:4d}  mean regret={mean_pr[tau]:+.3f}")
    print(f"Mean per-round regret at final step: {mean_pr[-1]:+.5f}")
    print("Saved figures/avg_regret_vs_iter.pdf and .png")


if __name__ == "__main__":
    main()
