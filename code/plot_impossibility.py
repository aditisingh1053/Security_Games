"""The lower bounds of Sections 7 and 8, in simulation.

Plays the three-target gadget game G_3 of Section 7.2 for real -- attacker
types are built from their utility vectors, best responses are computed from
those utilities, and the defender's payoff is read off the attacked target --
against the *optimal* defender for that game, namely the one that keeps the
version space of indices consistent with the observed targets and queries its
midpoint.  The expectation over the hidden type is computed exactly by
averaging over all M candidates rather than by sampling.

Produces `figures/impossibility.{pdf,png}`:

  (a) regret versus T when the universe has resolution M = 2^T (the infinite
      universe of Theorem 12): regret is T - 2, i.e. linear;
  (b) Proposition 14: with two attacker types on a two-element universe the
      regret is Theta(sqrt(T)), against the flat curve for one type;
  (c) Theorem 22: on the nested-interval instance, regret grows as K(B - 2)
      in the revelation length B, so batching really does cost Theta(K B).

Run from `code/`:  python plot_impossibility.py
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---- the gadget game G_3 --------------------------------------------------
# n = 3, one resource; u_d^c(i) = u_d^u(i) = -1 for i in {1,3} and 0 for i = 2,
# so U_d(1,p) = U_d(3,p) = -1 and U_d(2,p) = 0.
U_D = np.array([-1.0, 0.0, -1.0])


def attacker_utils(r1: float, r2: float):
    """Type alpha_{r1,r2} of Section 7.2: U(1,p) = r1 - p1, U(2,p) = 0,
    U(3,p) = (1 - r2) - p3.  All utilities lie in [-1, 1]."""
    u_c = np.array([r1 - 1.0, 0.0, -r2])
    u_u = np.array([r1, 0.0, 1.0 - r2])
    return u_c, u_u


def best_response(p, r1, r2):
    u_c, u_u = attacker_utils(r1, r2)
    return int(np.argmax(u_c * p + u_u * (1.0 - p)))   # ties -> lowest index


def defender_payoff(p, r1, r2):
    return float(U_D[best_response(p, r1, r2)])


def run_optimal_defender(M: int, T: int, j_true: int) -> int:
    """Misses incurred by midpoint version-space search against type j_true."""
    lo, hi, misses = 1, M, 0
    r1t, r2t = (j_true - 1) / M, j_true / M
    for _ in range(T):
        k = lo + (hi - lo) // 2                        # midpoint candidate
        p1 = k / M                                     # a point of I_k
        p = np.array([p1, 0.0, 1.0 - p1])
        obs = best_response(p, r1t, r2t)
        if obs == 1:                                   # target 2 attacked: hit
            continue
        misses += 1
        if obs == 0:                                   # target 1: I_true is right of p
            lo = k + 1
        else:                                          # target 3: I_true is left of p
            hi = k - 1
        if lo > hi:                                    # cannot happen
            lo, hi = 1, M
    return misses


def expected_regret(M: int, T: int) -> float:
    """Exact expectation over the uniform hidden index (benchmark payoff is 0)."""
    return float(np.mean([run_optimal_defender(M, T, j) for j in range(1, M + 1)]))


def closed_form(M: int, T: int) -> float:
    return T - (2.0 ** (T + 1) - T - 2.0) / M


def observe_exact(p1_num: int, r1_num: int, r2_num: int) -> int:
    """The observation of Lemma 10, in exact integer arithmetic.

    With p = (p1, 0, 1 - p1) and the type alpha_{r1,r2}, Lemma 10 gives
    b(p) = 2 iff p1 > r1 and 1 - p1 >= 1 - r2, i.e. iff r1 < p1 <= r2, and
    Section 7.2 gives b(p) = 1 for p1 <= r1 and b(p) = 3 for p1 > r2.  All three
    quantities are multiples of a common power of two here, so the comparisons
    are done on integer numerators and the nesting can be iterated to any
    depth without floating-point error.  `_check_observe_exact` verifies the
    identity against the utility-based best response at shallow depths.
    """
    if p1_num <= r1_num:
        return 0                                   # target 1
    if p1_num <= r2_num:
        return 1                                   # target 2 -- a hit
    return 2                                       # target 3


def _check_observe_exact(depth: int = 20, trials: int = 4000) -> float:
    rng = np.random.default_rng(0)
    den = 2 ** depth
    worst = 0
    for _ in range(trials):
        r1n = int(rng.integers(0, den)); r2n = r1n + int(rng.integers(1, den - r1n))
        p1n = int(rng.integers(0, den + 1))
        p1 = p1n / den
        got = best_response(np.array([p1, 0.0, 1.0 - p1]), r1n / den, r2n / den)
        worst = max(worst, int(got != observe_exact(p1n, r1n, r2n)))
    return worst


def run_nested(K: int, B: int, T: int, rng) -> int:
    """Theorem 22's instance: K nested types, one per revelation block.

    I_1 = (0,1]; given I_{j-1}, the interval I_j is a uniformly random dyadic
    sub-interval of I_{j-1} at depth B.  Block j is attacked by alpha^{(j)},
    whose interval the defender learns only at the checkpoint that closes the
    block.  A point of the innermost interval hits every type, so the
    benchmark is 0 and regret again counts misses.  All endpoints are
    multiples of 2^{-KB} and are stored as integer numerators.
    """
    den_bits = K * B
    lo, hi = 0, 1 << den_bits                     # currently revealed interval
    misses = 0
    for _ in range(1, K + 1):
        width = (hi - lo) >> B
        k_true = 1 + int(rng.integers(0, 1 << min(B, 62)))
        if B > 62:                                # not used, kept for safety
            raise ValueError("B too large")
        r1, r2 = lo + (k_true - 1) * width, lo + k_true * width
        klo, khi = 1, 1 << B                      # version space inside (lo, hi]
        for _ in range(min(B, max(T // K, 0))):
            k = klo + (khi - klo) // 2
            obs = observe_exact(lo + k * width, r1, r2)
            if obs == 1:
                continue
            misses += 1
            if obs == 0:
                klo = k + 1
            else:
                khi = k - 1
        lo, hi = r1, r2                           # the checkpoint reveals the type
    return misses


def _walk_abs_mean(T: int) -> float:
    """E|S_T| for a T-step simple random walk, exactly, in floating point."""
    from math import comb
    # S_T = 2X - T with X ~ Binomial(T, 1/2)
    return float(sum(comb(T, x) * abs(2 * x - T) for x in range(T + 1)) / 2 ** T)


def sqrt_floor(T: int) -> float:
    """The bound of Proposition 14: (1/2) sqrt(floor(T/2))."""
    return 0.5 * math.sqrt(T // 2)


def sqrt_exact(T: int) -> float:
    """(1/2) E|N_1 - N_2|: the smallest expected regret any algorithm can have
    on the two-type instance of Proposition 14, since no algorithm hits with
    probability more than 1/2 on any round."""
    return 0.5 * _walk_abs_mean(T)


def run_two_type(T: int, rng, reps: int) -> float:
    """Measured regret of follow-the-leader on the instance of Proposition 14.

    The two types are alpha_{0,1/2} and alpha_{1/2,1}; the defender plays the
    midpoint of the interval of whichever type it has seen more often so far,
    which is the best it can do given the past.  Regret = max(N1, N2) - hits,
    because a fixed strategy hits at most one of the two types (Lemma 10).
    """
    tot = 0.0
    for _ in range(reps):
        seq = rng.integers(0, 2, size=T)
        n = [0, 0]
        hits = 0
        for a in seq:
            guess = 0 if n[0] >= n[1] else 1
            # midpoint of I_guess; by Lemma 10 it hits iff the attacker is that type
            hits += int(guess == a)
            n[a] += 1
        tot += max(n) - hits
    return tot / reps


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- (a) infinite universe: M = 2^T --------------------------------
    Ts = list(range(1, 15))
    reg_inf = [expected_regret(2 ** T, T) for T in Ts]
    pred_inf = [closed_form(2 ** T, T) for T in Ts]
    err = max(abs(a - b) for a, b in zip(reg_inf, pred_inf))
    print(f"(a) max |simulated - T + (2^(T+1)-T-2)/2^T| = {err:.2e}")

    # ---- (c) the block-revelation lower bound -------------------------
    Bs = [4, 8, 16, 32]
    Ks = [1, 2, 3, 4]
    print(f"    exact-observation check against the utility-based best response: "
          f"{_check_observe_exact()} mismatches")
    rng = np.random.default_rng(0)
    nested = {K: [float(np.mean([run_nested(K, B, K * B, rng) for _ in range(400)]))
                  for B in Bs] for K in Ks}
    # ---- the sqrt(T) floor at Kmax = 2 (Proposition 14) -----------------
    Ts_c = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    ftl = [run_two_type(T, rng, 20000) for T in Ts_c]
    exact_c = [sqrt_exact(T) for T in Ts_c]
    floor_c = [sqrt_floor(T) for T in Ts_c]
    one_type = [expected_regret(2, min(T, 20)) for T in Ts_c]
    print("\n    T   Kmax=2 measured   (1/2)E|S_T|   floor   Kmax=1")
    for T, a, b_, c_, d_ in zip(Ts_c, ftl, exact_c, floor_c, one_type):
        print(f"{T:>5} {a:>15.3f} {b_:>13.3f} {c_:>7.3f} {d_:>8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    cols = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]

    ax = axes[0]
    ax.plot(Ts, reg_inf, "o-", color="tab:red", lw=1.7, ms=4,
            label="optimal defender, $M=2^{T}$")
    ax.plot(Ts, [T - 2 for T in Ts], color="black", ls="--", lw=1.0,
            label=r"$T-2$ (Theorem 12)")
    ax.plot(Ts, Ts, color="0.6", ls=":", lw=1.0, label=r"$T$ (all rounds lost)")
    ax.set_title(r"(a) infinite universe: $\mathbb{E}[\mathrm{Regret}] = T-2+o(1)$",
                 fontsize=9)
    ax.set_xlabel("horizon $T$", fontsize=8)
    ax.set_ylabel("expected regret", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(Ts_c, ftl, "o", color="tab:red", ms=5,
            label=r"$|K|=2$: measured (follow the leader)")
    ax.plot(Ts_c, exact_c, "-", color="tab:red", lw=1.4,
            label=r"$\frac{1}{2}\mathbb{E}|N_1-N_2|$ (optimal)")
    ax.plot(Ts_c, floor_c, "--", color="black", lw=1.2,
            label=r"$\frac{1}{2}\sqrt{\lfloor T/2\rfloor}$ (Prop. 14)")
    ax.plot(Ts_c, one_type, ":", color="tab:blue", lw=1.6,
            label=r"$|K|=1$ on the same universe")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(r"(b) two types already cost $\sqrt{T}$, at $|\mathfrak{U}|=2$",
                 fontsize=9)
    ax.set_xlabel("horizon $T$", fontsize=8)
    ax.set_ylabel("expected regret", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25)

    ax = axes[2]
    Bs_arr = np.array(Bs, dtype=float)
    for K, c in zip(Ks, cols):
        ax.plot(Bs_arr, K * (Bs_arr - 2), color="0.35", ls="--", lw=1.6,
                zorder=1, label=r"$|K|(B-2)$" if K == Ks[0] else None)
        ax.plot(Bs_arr, nested[K], "o", color=c, ms=5, zorder=3,
                label=rf"measured, $|K|={K}$")
    ax.set_title(r"(c) scheduled revelation: regret $\geq |K|(B-2)$ (Thm. 22)",
                 fontsize=9)
    ax.set_xlabel("revelation length $B$", fontsize=8)
    ax.set_ylabel("expected regret", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"impossibility.{ext}"), dpi=180)
    print("wrote figures/impossibility.{pdf,png}")

    print("\n  T   M=2^T   simulated   T-2")
    for T, r in zip(Ts, reg_inf):
        print(f"{T:>3} {2**T:>8}  {r:10.5f} {T-2:>6}")
    print("\n  |K|    B   scheduled   K(B-2)")
    for K in Ks:
        for B, v in zip(Bs, nested[K]):
            print(f"{K:>5} {B:>4} {v:>11.3f} {K*(B-2):>8}")


if __name__ == "__main__":
    main()
