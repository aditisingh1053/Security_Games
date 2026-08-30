"""Runtime of the best-response oracle versus exhaustive vertex enumeration.

Produces the scaling table quoted in Appendix C of the report: for each
(n, K) we time

  * `algorithm.compute_extreme_points` -- the exhaustive enumeration of
    E(C; eps) used by Algorithm 1 (measured where feasible, otherwise the
    analytic number of (n-1)-subsets it would have to solve);
  * `oracle.maximize_enum`             -- one LP per best-response profile;
  * `oracle.maximize_milp`             -- the single MILP.

Run from the `code/` directory:  python bench_oracle.py
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from algorithm import AttackerType, SSGame, compute_extreme_points
from oracle import instance_from_game, maximize_enum, maximize_milp, payoff_lift

ENUM_VERTEX_BUDGET = 3e6      # (n-1)-subsets we are willing to actually run
ENUM_LP_BUDGET = 2e5          # profiles we are willing to actually run


def random_instance(n: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    game = SSGame(
        n=n,
        u_d_c=rng.uniform(0.1, 1.0, n),
        u_d_u=rng.uniform(-1.0, -0.1, n),
    )
    types = [
        AttackerType(
            u_c=rng.uniform(-1.0, -0.1, n),
            u_u=rng.uniform(0.1, 1.0, n),
            type_id=i,
        )
        for i in range(K)
    ]
    return game, types, rng


def subset_count(n: int, K: int) -> float:
    """Number of (n-1)-subsets `compute_extreme_points` enumerates."""
    H = (n - 1) + 1 + K * n * (n - 1) // 2
    return float(math.comb(H, n - 1)) if n >= 2 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 5, 10, 20, 30, 40])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    header = (
        f"{'n':>4} {'K':>3} {'subsets C(H,n-1)':>18} {'enumerate E':>14} "
        f"{'profile LPs':>12} {'MILP (ms)':>10} {'agree':>6}"
    )
    print(header)
    print("-" * len(header))

    for n in args.ns:
        for K in args.ks:
            game, types, rng = random_instance(n, K, seed=1000 + 13 * n + K)
            inst = instance_from_game(game, types)

            # --- exhaustive vertex enumeration (Algorithm 1's expert set) ---
            subs = subset_count(n, K)
            if subs <= ENUM_VERTEX_BUDGET:
                t0 = time.perf_counter()
                E = compute_extreme_points(game, types)
                t_enum = f"{(time.perf_counter() - t0) * 1e3:.0f} ms |E|={len(E)}"
            else:
                t_enum = "infeasible"

            # --- oracle: profile enumeration and MILP -----------------------
            times_milp, agree = [], None
            t_lp = "infeasible"
            for rep in range(args.reps):
                a, b = payoff_lift(inst, rng.uniform(0.0, 1.0, K))
                a = a + rng.uniform(0.0, 3.0, (K, n))
                b = b + rng.uniform(-1.0, 3.0, (K, n))

                t0 = time.perf_counter()
                _, v_milp, _ = maximize_milp(inst, a, b)
                times_milp.append(time.perf_counter() - t0)

                if rep == 0 and float(n) ** K <= ENUM_LP_BUDGET:
                    t0 = time.perf_counter()
                    _, v_lp, _ = maximize_enum(inst, a, b)
                    t_lp = f"{(time.perf_counter() - t0) * 1e3:.0f} ms"
                    agree = abs(v_lp - v_milp) <= 1e-6

            tag = "-" if agree is None else ("yes" if agree else "NO")
            print(
                f"{n:>4} {K:>3} {subs:>18.3g} {t_enum:>14} "
                f"{t_lp:>12} {np.median(times_milp) * 1e3:>10.1f} {tag:>6}"
            )


if __name__ == "__main__":
    main()
