"""Numerical checks of the non-asymptotic claims in Sections 6-9.

Covered: Step 7 of Section 6, every claim of Sections 7 and 8 that is not
purely asymptotic, and
the oracle and estimator claims of Appendix B.  NOT covered, because they
are measurements rather than claims about the algebra: the fitted exponents,
envelope constants and timings of Section 9 and Appendices B-C, printed by
`bench_oracle.py`, `plot_tractable_regret.py`, `plot_block_revelation.py` and
`plot_impossibility.py` when those are rerun.

Theorem, lemma and corollary numbers below are those of
`project_report_extended.pdf`.

Each check prints PASS/FAIL and the worst discrepancy it saw.  Nothing here
is used to produce a figure; the point is that the algebra in the proofs can
be re-run.

Run from `code/`:  python verify_theory.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import math
from functools import lru_cache

import numpy as np

from algorithm import AttackerType, SSGame, compute_extreme_points
from oracle import instance_from_game, maximize_enum, maximize_milp, payoff_lift
from plot_impossibility import (
    U_D,
    attacker_utils,
    best_response,
    closed_form,
    expected_regret,
    observe_exact,
    run_nested,
    run_two_type,
    sqrt_exact,
    sqrt_floor,
)
from tractable import barycentric_spanner, collect_W

FAILS = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
def check_lemma10(trials=20000, seed=0):
    """Section 7.2: utilities in [-1,1] and the 1/2/3 pattern on the segment."""
    rng = np.random.default_rng(seed)
    bad_range = bad_pattern = 0
    for _ in range(trials):
        r1, r2 = sorted(rng.uniform(0, 1, 2))
        u_c, u_u = attacker_utils(r1, r2)
        if np.max(np.abs(u_c)) > 1 + 1e-12 or np.max(np.abs(u_u)) > 1 + 1e-12:
            bad_range += 1
        p1 = rng.uniform()
        p = np.array([p1, 0.0, 1 - p1])
        want = 0 if p1 <= r1 else (1 if p1 <= r2 else 2)
        if best_response(p, r1, r2) != want:
            bad_pattern += 1
    report("Section 7.2 (realizable threshold types)",
           bad_range == 0 and bad_pattern == 0,
           f"range violations {bad_range}, pattern mismatches {bad_pattern}")


def check_lemma11(trials=50000, seed=1):
    """Lemma 10: hit iff p1 > r1 and p3 >= 1-r2, and at most one hit."""
    rng = np.random.default_rng(seed)
    bad_iff = 0
    worst_hits = 0
    for _ in range(trials):
        r1, r2 = sorted(rng.uniform(0, 1, 2))
        p = rng.dirichlet(np.ones(4))[:3]           # p1+p2+p3 <= 1
        hit = best_response(p, r1, r2) == 1
        if hit != (p[0] > r1 and p[2] >= 1 - r2):
            bad_iff += 1
    for M in (8, 32, 128):
        for _ in range(trials // 10):
            p = rng.dirichlet(np.ones(4))[:3]
            hits = sum(best_response(p, (j - 1) / M, j / M) == 1 for j in range(1, M + 1))
            worst_hits = max(worst_hits, hits)
    report("Lemma 10 (hit condition, at most one hit)",
           bad_iff == 0 and worst_hits <= 1,
           f"iff mismatches {bad_iff}, max simultaneous hits {worst_hits}")


def check_benchmarks_tie_free(M=64, seed=7):
    """The benchmarks of Theorems 12 and 22 must hit under ANY tie rule.

    Theorem 12 uses the midpoint of I_j and Theorem 22 the midpoint of the
    innermost nested interval; at both, target 2 is the strict maximizer, so
    the value of the benchmark does not depend on how ties are broken.  The
    endpoint of I_j, which sits exactly on the tie U(3, p) = U(2, p) = 0,
    would not have this property -- that is checked here too.
    """
    def br(p, r1, r2, rule):
        u = np.array([r1 - p[0], 0.0, (1 - r2) - p[2]])
        ties = [i for i in range(3) if u[i] >= u.max() - 1e-15]
        return ties[0] if rule == "low" else ties[-1]

    mid_ok = end_ok = True
    for j in range(1, M + 1):
        r1, r2 = (j - 1) / M, j / M
        mu = (2 * j - 1) / (2 * M)
        pm = np.array([mu, 0.0, 1 - mu])
        pe = np.array([r2, 0.0, 1 - r2])
        for rule in ("low", "high"):
            mid_ok &= br(pm, r1, r2, rule) == 1
            end_ok &= br(pe, r1, r2, rule) == 1
    # Theorem 22: midpoint of the innermost of a nested chain hits every type
    rng = np.random.default_rng(seed)
    nested_ok = True
    for _ in range(2000):
        lo, hi = 0.0, 1.0
        chain = []
        for _ in range(4):
            k = int(rng.integers(1, 9)); w = (hi - lo) / 8
            lo, hi = lo + (k - 1) * w, lo + k * w
            chain.append((lo, hi))
        mu = 0.5 * (lo + hi)
        p = np.array([mu, 0.0, 1 - mu])
        for (r1, r2) in chain:
            for rule in ("low", "high"):
                nested_ok &= br(p, r1, r2, rule) == 1
    report("Theorems 12, 22 (benchmark hits under any tie rule)",
           mid_ok and nested_ok and not end_ok,
           f"midpoint {mid_ok}, nested midpoint {nested_ok}, "
           f"endpoint tie-free {end_ok} (expected False)")


def check_lemma12_and_thm13(Tmax=10):
    """Lemma 11 / Theorem 12: the counting bound is exactly the optimum.

    F(m, s) is the largest total number of hits, summed over all m candidates,
    that any deterministic strategy can force in s rounds when the consistent
    set has size m.  The optimal defender's expected regret is T - F(M,T)/M.
    """
    @lru_cache(maxsize=None)
    def F(m, s):
        if m == 0 or s == 0:
            return 0
        return max(s + F(k - 1, s - 1) + F(m - k, s - 1) for k in range(1, m + 1))

    worst = 0.0
    for T in range(1, Tmax + 1):
        M = 2 ** T
        dp = T - F(M, T) / M
        worst = max(worst, abs(dp - closed_form(M, T)))
    report("Lemma 11 / Theorem 12 (counting bound is exactly attained)",
           worst < 1e-12, f"max |DP optimum - (T - (2^(T+1)-T-2)/M)| = {worst:.2e}")


def check_simulation(Tmax=14):
    """Theorem 12 in the actual game: simulated regret equals T-2+(T+2)/2^T."""
    worst = max(abs(expected_regret(2 ** T, T) - closed_form(2 ** T, T))
                for T in range(1, Tmax + 1))
    report("Theorem 12 (simulated game matches the bound)",
           worst < 1e-12, f"max discrepancy {worst:.2e}")


def check_observe_exact(depth=24, trials=20000, seed=2):
    """The integer observation map used by the nested experiment is the true
    best response."""
    rng = np.random.default_rng(seed)
    den = 2 ** depth
    bad = 0
    for _ in range(trials):
        r1n = int(rng.integers(0, den))
        r2n = r1n + int(rng.integers(1, den - r1n))
        p1n = int(rng.integers(0, den + 1))
        p1 = p1n / den
        if best_response(np.array([p1, 0.0, 1 - p1]), r1n / den, r2n / den) != \
                observe_exact(p1n, r1n, r2n):
            bad += 1
    report("exact observation map", bad == 0, f"mismatches {bad}")


def check_holder(trials=200000, seed=3):
    """The Holder step of Theorem 18, as the proof actually uses it.

    Holder alone gives sum L_j^{2/3} <= m^{1/3} (sum L_j)^{2/3}, which is a
    theorem and cannot fail.  The step the proof takes is the chain that
    follows from it *together with* the two facts Lemma 17 supplies,
    m <= K' and sum_j L_j <= T, namely

        sum_j L_j^{2/3} <= K'^{1/3} T^{2/3},

    and the claim in the text that equality needs all epochs equal.  Epoch
    lengths are therefore drawn subject to those two constraints, including
    the degenerate cases m = 0 and L_j = 0.
    """
    rng = np.random.default_rng(seed)
    bad = bad_equal = 0
    worst_slack = np.inf
    for _ in range(trials):
        Kp = int(rng.integers(1, 30))
        T = int(rng.integers(1, 20000))
        m = int(rng.integers(0, Kp + 1))                  # m <= K', m = 0 allowed
        if m == 0:
            L = np.zeros(0)
        else:
            cuts = np.sort(rng.integers(0, T + 1, size=m - 1))
            L = np.diff(np.concatenate(([0], cuts, [T]))).astype(float)
            L = L[rng.permutation(m)] if m > 1 else L
        lhs = float(np.sum(L ** (2 / 3)))
        rhs = Kp ** (1 / 3) * float(T) ** (2 / 3)
        if lhs > rhs + 1e-6:
            bad += 1
        # equality case: m = K' epochs all of length T/K'
        eq_lhs = Kp * (T / Kp) ** (2 / 3)
        if abs(eq_lhs - rhs) > 1e-6 * rhs:
            bad_equal += 1
        if rhs > 0:
            worst_slack = min(worst_slack, (rhs - lhs) / rhs)
    report("Theorem 18 (the Holder step, under Lemma 17's constraints)",
           bad == 0 and bad_equal == 0,
           f"violations {bad}, equality-case mismatches {bad_equal}, "
           f"tightest relative slack {worst_slack:.2e}")


def check_Bstar(trials=100000, seed=4):
    """Corollary 21: B* is where batching stops being free, and the call count
    it licenses is optimal.

    Asserting 2*K*B* == lead would be a tautology, since B* is *defined* as
    lead/(2K).  What is checked here is the content of the corollary: the
    additive term is dominated for every B <= B* and dominates for B > B*;
    T/B* has the closed form of Equation (47); and the necessity bound
    N >= K T / (R + 2K) of Equation (48), evaluated at R = lead, agrees with
    that sufficiency count to within an absolute constant -- which is the
    optimality claim.
    """
    rng = np.random.default_rng(seed)
    bad_dom = bad_calls = bad_lb = 0
    worst_ratio = 0.0
    for _ in range(trials):
        n = int(rng.integers(2, 200))
        K = int(rng.integers(2, 200))
        T = float(10 ** rng.uniform(2, 9))
        lead = n * K ** (4 / 3) * T ** (2 / 3) * math.log(n * K) ** (1 / 3)
        Bstar = 0.5 * lead / K
        for B in (0.5 * Bstar, 0.99 * Bstar):          # dominated
            if 2 * K * B > lead + 1e-9:
                bad_dom += 1
        for B in (1.01 * Bstar, 2 * Bstar):            # dominating
            if 2 * K * B <= lead:
                bad_dom += 1
        claimed = T ** (1 / 3) / (n * K ** (1 / 3) * math.log(n * K) ** (1 / 3))
        if abs(T / Bstar - 2 * claimed) > 1e-6 * max(1.0, 2 * claimed):
            bad_calls += 1
        # Equation (48) rearranged: N_suff / N_nec = 2 (1 + 2K/lead) exactly.
        # Checking the identity catches an algebra slip in the corollary's
        # proof; checking the numeric band confirms it is an absolute constant.
        ratio = (T / Bstar) / (K * T / (lead + 2 * K))
        worst_ratio = max(worst_ratio, ratio)
        if abs(ratio - 2 * (1 + 2 * K / lead)) > 1e-9 * ratio or not (2.0 - 1e-9 <= ratio <= 3.0):
            bad_lb += 1
    report("Corollary 21 (B* is the crossover; call count optimal)",
           bad_dom == 0 and bad_calls == 0 and bad_lb == 0,
           f"dominance flips {bad_dom}, call-count form {bad_calls}, "
           f"sufficient/necessary call ratio = 2(1+2K/lead) <= {worst_ratio:.4f} "
           f"({bad_lb} violations)")


def check_sqrtT(seed=12):
    """Proposition 14: two types on a two-element universe cost Theta(sqrt T).

    (a) The floor (1/2) sqrt(floor(T/2)) is below the exact optimal expected
    regret (1/2) E|N_1 - N_2| for every T in a wide range -- this is the
    binomial estimate used in the proof.
    (b) Follow the leader, simulated in the real game, attains that optimum,
    which is the tightness claim after the proof.
    (c) The combinatorial identity binom(2v, v) 4^{-v} >= 1/(2 sqrt(v)).
    """
    from math import comb, sqrt
    floor_ok = all(sqrt_floor(T) <= sqrt_exact(T) + 1e-12 for T in range(2, 600))
    binom_ok = all(comb(2 * v, v) / 4.0 ** v >= 1 / (2 * sqrt(v)) - 1e-12
                   for v in range(1, 400))
    rng = np.random.default_rng(seed)
    worst = 0.0
    for T in (16, 64, 256, 1024):
        got = run_two_type(T, rng, 20000)
        worst = max(worst, abs(got - sqrt_exact(T)) / sqrt_exact(T))
    report("Proposition 14 (two types cost sqrt(T), and the floor is valid)",
           floor_ok and binom_ok and worst < 0.05,
           f"floor <= optimum {floor_ok}, binomial bound {binom_ok}, "
           f"max relative gap of follow-the-leader to the optimum {worst:.3f}")


def check_oracle(trials=60, seed=5):
    """Appendix B: MILP == profile enumeration == brute force over E(C;eps)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(3, 6))
        K = int(rng.integers(1, 4))
        game = SSGame(n=n, u_d_c=rng.uniform(0.1, 1, n), u_d_u=rng.uniform(-1, -0.1, n))
        types = [AttackerType(u_c=rng.uniform(-1, -0.1, n), u_u=rng.uniform(0.1, 1, n),
                              type_id=i) for i in range(K)]
        inst = instance_from_game(game, types)
        w = rng.uniform(0, 1, K)
        a, b = payoff_lift(inst, w)
        pe, ve, _ = maximize_enum(inst, a, b)
        pm, vm, _ = maximize_milp(inst, a, b)
        vE = max(sum(w[k] * game.payoff_against(al, p) for k, al in enumerate(types))
                 for p in compute_extreme_points(game, types))
        worst = max(worst, abs(ve - vm), abs(ve - inst.lifted_value(pe, a, b)),
                    abs(vm - inst.lifted_value(pm, a, b)), vE - ve)
    report("Appendix B (oracle back-ends agree with brute force)", worst < 1e-5,
           f"worst discrepancy {worst:.2e}")


def check_estimator(reps=40000, seed=6):
    """Lemma 24(ii): |B_tau| g_tau is unbiased for the block's type counts."""
    rng = np.random.default_rng(seed)
    n, K, L = 6, 3, 40
    game = SSGame(n=n, u_d_c=rng.uniform(0.1, 1, n), u_d_u=rng.uniform(-1, -0.1, n))
    types = [AttackerType(u_c=rng.uniform(-1, -0.1, n), u_u=rng.uniform(0.1, 1, n),
                          type_id=i) for i in range(K)]
    inst = instance_from_game(game, types)
    Lam, span = barycentric_spanner(collect_W(inst, rng), K)
    m = rng.multinomial(L, rng.dirichlet(np.ones(K)))
    block = np.repeat(np.arange(K), m)
    rng.shuffle(block)
    acc = np.zeros(K)
    for _ in range(reps):
        phat = np.zeros(K)
        slots = rng.choice(L, size=K, replace=False)
        for j, (_, p_b, i_b) in enumerate(span):
            phat[j] = 1.0 if types[block[slots[j]]].best_response(p_b) == i_b else 0.0
        acc += np.linalg.solve(Lam, phat)
    est = acc / reps * L
    err = float(np.abs(est - m).max())
    # three standard errors of the mean of a Bernoulli-driven estimate
    tol = 3 * L * math.sqrt(K) / math.sqrt(reps) + 0.05
    report("Lemma 24(ii) (frequency estimator is unbiased)", err < tol,
           f"max |estimate - truth| = {err:.3f} on counts {m} (tolerance {tol:.3f})")


def check_step7():
    """Section 6, Step 7: log Nmax = O(n^2 K log(nK)), and Step 10's substitution.

    Guards the whole chain, including the trap that sank the first draft: the
    sqrt(K) of the Cauchy-Schwarz step (Step 9) makes the regret scale with K,
    not sqrt(K), so Balcan's Theorem 5.1 form with k -> Kmax is NOT an upper
    bound on ours.
    """
    lg = lambda z: math.log(z, 2)
    grid = [(n, K) for n in range(2, 41)
                   for K in (1, 2, 3, 5, 10, 100, 10**3, 10**6)]

    # (i) each inequality of the displayed chain, term by term
    line1 = lambda n, K: n * lg(2**n + K * n * n) + K * lg(n)
    line2 = lambda n, K: n * (n + 1 + lg(K * n * n)) + K * lg(n)
    line3 = lambda n, K: n * n + n * lg(n * K) + K * lg(n)
    line4 = lambda n, K: n * n * K * lg(n * K)
    bad12 = [(n, K) for n, K in grid if line1(n, K) > line2(n, K) + 1e-9]
    # lines 2 -> 3 -> 4 hold up to absolute constants; record the constants
    c23 = max(line2(n, K) / line3(n, K) for n, K in grid)
    c34 = max(line3(n, K) / line4(n, K) for n, K in grid)
    c14 = max(line1(n, K) / line4(n, K) for n, K in grid)
    report("Section 6, Step 7 (log Nmax chain)",
           not bad12 and c23 <= 3 and c34 <= 3,
           f"exact<=line2 violations {len(bad12)}, "
           f"line2/line3 <= {c23:.3f}, line3/line4 <= {c34:.3f}, "
           f"exact/final <= {c14:.3f}")

    # (ii) Step 10: sqrt(K * T * logNmax) must reproduce n*K*sqrt(T log nK)
    T = 10**6
    worst = max(abs(math.sqrt(K * T * line4(n, K)) - n * K * math.sqrt(T * lg(n * K)))
                / (n * K * math.sqrt(T * lg(n * K))) for n, K in grid)
    report("Section 6, Step 10 (substitution gives n*K*sqrt(T log nK))",
           worst < 1e-12, f"max relative error {worst:.2e}")

    # (iii) the guard: K to the FIRST power understates our bound by exactly K.
    # (our regret)^2 = K * T * logNmax; Balcan Thm 5.1 with k -> K is T n^2 K log(nK).
    ours = lambda n, K: K * T * line4(n, K)
    balcan_k1 = lambda n, K: T * n * n * K * lg(n * K)
    rel = max(abs(ours(n, K) / balcan_k1(n, K) - K) / K for n, K in grid)
    viol = [(n, K) for n, K in grid if ours(n, K) > balcan_k1(n, K) * (1 + 1e-9)]
    kge2 = sum(1 for _, K in grid if K >= 2)
    report("Section 6 (Cauchy-Schwarz sqrt(K) is real: the k^1 form is not an upper bound)",
           rel < 1e-12 and len(viol) == kge2,
           f"(ours / k^1 form) = Kmax exactly, max relative dev {rel:.2e}; "
           f"strict on {len(viol)}/{kge2} points with Kmax >= 2, equality at Kmax = 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    print("Section 6 (the expert-set bound)")
    check_step7()
    print("Section 7 (the gadget and the lower bounds)")
    check_lemma10(2000 if q else 20000)
    check_lemma11(5000 if q else 50000)
    check_benchmarks_tie_free()
    check_lemma12_and_thm13(7 if q else 10)
    check_simulation(8 if q else 14)
    check_observe_exact(trials=2000 if q else 20000)
    check_sqrtT()
    print("Section 8 (the block-revelation bound)")
    check_holder(20000 if q else 200000)
    check_Bstar(10000 if q else 100000)
    print("Appendix B (the implementation)")
    check_oracle(10 if q else 60)
    check_estimator(4000 if q else 40000)
    print()
    print("all checks passed" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
