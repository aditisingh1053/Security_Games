#!/usr/bin/env python3
"""
SEAD Game Analysis: Phase 1 & 2

Phase 1: Nash equilibrium via CFR+ on the SEAD NFGSS
Phase 2: Stackelberg equilibrium via grid-search LP
Comparison: Comprehensive visualizations

Usage:
    python run_analysis.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sead_game import SEADGame, SEADConfig
from cfr_solver import CFRPlusSolver
from stackelberg_solver import StackelbergSolver
from visualizations import (
    plot_convergence, plot_strategy_comparison, plot_value_comparison,
    plot_resource_sensitivity, plot_strategy_evolution, plot_payoff_heatmap,
    plot_dashboard, plot_strategy_profile, plot_config_comparison_table,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results -> {os.path.abspath(OUTPUT_DIR)}")


def run_cfr(game: SEADGame, n_iters: int = 300, log_interval: int = 10,
            verbose: bool = True):
    """Run CFR+ and return results + evolution data."""
    cfr = CFRPlusSolver(game)
    evolution = {'attacker': {}, 'defender': {}}
    exploit_hist, def_hist, att_hist = [], [], []

    for i in range(1, n_iters + 1):
        cfr.iteration = i
        cfr._cfr_traverse(game.root, 1.0, 1.0, 1.0, 'attacker')
        cfr._cfr_traverse(game.root, 1.0, 1.0, 1.0, 'defender')

        if i % log_interval == 0 or i == 1:
            exploit = cfr.compute_exploitability()
            game_val = cfr._compute_expected_utility(game.root)
            exploit_hist.append((i, exploit))
            def_hist.append((i, game_val))
            att_hist.append((i, -game_val))
            rs = cfr.get_root_strategies()
            evolution['attacker'][i] = rs['attacker']
            evolution['defender'][i] = rs['defender']
            if verbose and (i % (log_interval * 5) == 0 or i == 1):
                print(f"  CFR+ iter {i:4d}: exploit={exploit:.6f} def={game_val:.4f}")

    final_exploit = cfr.compute_exploitability()
    final_value = cfr._compute_expected_utility(game.root)
    cfr.exploitability_history = exploit_hist
    cfr.def_utility_history = def_hist
    cfr.att_utility_history = att_hist

    result = {
        'exploitability': final_exploit,
        'defender_value': final_value,
        'attacker_value': -final_value,
        'exploitability_history': exploit_hist,
        'defender_utility_history': def_hist,
        'attacker_utility_history': att_hist,
    }
    return cfr, result, evolution


def run_single_config(config: SEADConfig, cfr_iters=300, stack_grid=6,
                       verbose=True):
    """Run both solvers on one configuration."""
    label = (f"R={config.num_rounds} D={config.attacker_decoys} "
             f"M={config.defender_missiles} p_dec={config.p_deceive}")
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Config: {label}")
        print(f"{'='*55}")

    # Build game
    t0 = time.time()
    game = SEADGame(config)
    stats = game.get_game_stats()
    if verbose:
        print(f"  Game: {stats['total_nodes']} nodes, {stats['attacker_infosets']} att IS, "
              f"{stats['defender_infosets']} def IS ({time.time()-t0:.2f}s)")

    # CFR+
    t0 = time.time()
    cfr, nash_result, evolution = run_cfr(game, cfr_iters, log_interval=max(cfr_iters//20, 1),
                                           verbose=verbose)
    nash_strats = cfr.get_root_strategies()
    all_nash_strats = cfr.get_all_strategies()
    if verbose:
        print(f"  CFR+ done ({time.time()-t0:.1f}s): def={nash_result['defender_value']:.4f} "
              f"exploit={nash_result['exploitability']:.6f}")

    # Stackelberg
    t0 = time.time()
    stack = StackelbergSolver(game)
    stack_result = stack.solve(grid_resolution=stack_grid, verbose=False)
    if verbose:
        print(f"  Stackelberg done ({time.time()-t0:.1f}s): "
              f"def={stack_result['defender_value']:.4f}")

    return {
        'config': config,
        'game': game,
        'game_stats': stats,
        'cfr': cfr,
        'nash_result': nash_result,
        'nash_strats': nash_strats,
        'all_nash_strats': all_nash_strats,
        'stack_result': stack_result,
        'evolution': evolution,
    }


def run_sensitivity(base_config, cfr_iters=200, stack_grid=5):
    """Sensitivity analysis varying one param at a time."""
    sensitivity_data = []

    # Vary decoys
    print("\n--- Sensitivity: Decoy Budget ---")
    vals = [1, 2, 3]
    nd, sd, na, sa = [], [], [], []
    for d in vals:
        cfg = SEADConfig(num_rounds=base_config.num_rounds,
                         attacker_decoys=d, attacker_real=base_config.attacker_real,
                         defender_missiles=base_config.defender_missiles,
                         p_deceive=base_config.p_deceive, p_hit_real=base_config.p_hit_real,
                         p_hit_decoy=base_config.p_hit_decoy)
        r = run_single_config(cfg, cfr_iters, stack_grid, verbose=False)
        nd.append(r['nash_result']['defender_value'])
        sd.append(r['stack_result']['defender_value'])
        na.append(r['nash_result']['attacker_value'])
        sa.append(r['stack_result']['attacker_value'])
        print(f"  D={d}: Nash def={nd[-1]:.3f} Stack def={sd[-1]:.3f}")
    sensitivity_data.append({
        'param_name': 'Attacker Decoy Budget',
        'param_values': vals,
        'nash_def_values': nd, 'stack_def_values': sd,
        'nash_att_values': na, 'stack_att_values': sa,
    })

    # Vary missiles
    print("\n--- Sensitivity: Missile Inventory ---")
    vals = [1, 2, 3]
    nd, sd, na, sa = [], [], [], []
    for m in vals:
        cfg = SEADConfig(num_rounds=base_config.num_rounds,
                         attacker_decoys=base_config.attacker_decoys,
                         attacker_real=base_config.attacker_real,
                         defender_missiles=m,
                         p_deceive=base_config.p_deceive, p_hit_real=base_config.p_hit_real,
                         p_hit_decoy=base_config.p_hit_decoy)
        r = run_single_config(cfg, cfr_iters, stack_grid, verbose=False)
        nd.append(r['nash_result']['defender_value'])
        sd.append(r['stack_result']['defender_value'])
        na.append(r['nash_result']['attacker_value'])
        sa.append(r['stack_result']['attacker_value'])
        print(f"  M={m}: Nash def={nd[-1]:.3f} Stack def={sd[-1]:.3f}")
    sensitivity_data.append({
        'param_name': 'Defender Missile Inventory',
        'param_values': vals,
        'nash_def_values': nd, 'stack_def_values': sd,
        'nash_att_values': na, 'stack_att_values': sa,
    })

    # Vary deception probability
    print("\n--- Sensitivity: Deception Probability ---")
    vals = [0.3, 0.5, 0.7]
    nd, sd, na, sa = [], [], [], []
    for p in vals:
        cfg = SEADConfig(num_rounds=base_config.num_rounds,
                         attacker_decoys=base_config.attacker_decoys,
                         attacker_real=base_config.attacker_real,
                         defender_missiles=base_config.defender_missiles,
                         p_deceive=p, p_hit_real=base_config.p_hit_real,
                         p_hit_decoy=base_config.p_hit_decoy)
        r = run_single_config(cfg, cfr_iters, stack_grid, verbose=False)
        nd.append(r['nash_result']['defender_value'])
        sd.append(r['stack_result']['defender_value'])
        na.append(r['nash_result']['attacker_value'])
        sa.append(r['stack_result']['attacker_value'])
        print(f"  p_dec={p}: Nash def={nd[-1]:.3f} Stack def={sd[-1]:.3f}")
    sensitivity_data.append({
        'param_name': 'Decoy Deception Prob.',
        'param_values': vals,
        'nash_def_values': nd, 'stack_def_values': sd,
        'nash_att_values': na, 'stack_att_values': sa,
    })

    return sensitivity_data


def main():
    ensure_output_dir()
    total_start = time.time()

    # ================================================================
    # MAIN ANALYSIS
    # ================================================================
    print("\n" + "#"*55)
    print("  SEAD GAME ANALYSIS")
    print("  Phase 1: Nash (CFR+) | Phase 2: Stackelberg (LP)")
    print("#"*55)

    base_config = SEADConfig(
        num_rounds=3, attacker_decoys=2, attacker_real=1,
        defender_missiles=3, p_deceive=0.5, p_hit_real=0.7, p_hit_decoy=0.9,
    )

    main_res = run_single_config(base_config, cfr_iters=300, stack_grid=6,
                                  verbose=True)

    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n\n" + "#"*55)
    print("  GENERATING VISUALIZATIONS")
    print("#"*55)

    # 1. Convergence
    print("\n[1/8] CFR+ Convergence...")
    plot_convergence(
        main_res['nash_result']['exploitability_history'],
        main_res['nash_result']['defender_utility_history'],
        main_res['nash_result']['attacker_utility_history'],
        OUTPUT_DIR)

    # 2. Strategy comparison
    print("[2/8] Strategy Comparison...")
    plot_strategy_comparison(main_res['nash_strats'], main_res['stack_result'],
                              OUTPUT_DIR)

    # 3. Value comparison
    print("[3/8] Value Comparison...")
    plot_value_comparison(main_res['nash_result'], main_res['stack_result'],
                           OUTPUT_DIR)

    # 4. Payoff heatmap
    print("[4/8] Payoff Heatmap...")
    plot_payoff_heatmap(main_res['game'], OUTPUT_DIR)

    # 5. Strategy evolution
    print("[5/8] Strategy Evolution...")
    plot_strategy_evolution(main_res['evolution'], OUTPUT_DIR)

    # 6. Detailed strategy profiles
    print("[6/8] Strategy Profiles...")
    plot_strategy_profile(main_res['all_nash_strats'], 'attacker', 'Nash', OUTPUT_DIR)
    plot_strategy_profile(main_res['all_nash_strats'], 'defender', 'Nash', OUTPUT_DIR)

    # 7. Sensitivity analysis
    print("[7/8] Sensitivity Analysis...")
    sensitivity = run_sensitivity(base_config, cfr_iters=200, stack_grid=5)
    plot_resource_sensitivity(sensitivity, OUTPUT_DIR)

    # 8. Multi-config comparison
    print("\n[8/8] Multi-Config Comparison...")
    multi_configs = [
        SEADConfig(num_rounds=2, attacker_decoys=1, attacker_real=1,
                   defender_missiles=2, p_deceive=0.5, p_hit_real=0.7),
        SEADConfig(num_rounds=3, attacker_decoys=2, attacker_real=1,
                   defender_missiles=3, p_deceive=0.5, p_hit_real=0.7),
        SEADConfig(num_rounds=2, attacker_decoys=2, attacker_real=1,
                   defender_missiles=2, p_deceive=0.7, p_hit_real=0.7),
    ]
    multi_results = []
    for cfg in multi_configs:
        r = run_single_config(cfg, cfr_iters=200, stack_grid=5, verbose=False)
        multi_results.append(r)
    plot_config_comparison_table(multi_results, OUTPUT_DIR)

    # 9. Dashboard
    print("\n[9/9] Dashboard...")
    plot_dashboard(
        main_res['nash_result'], main_res['stack_result'],
        main_res['nash_strats'], main_res['stack_result'],
        main_res['game_stats'], OUTPUT_DIR)

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    print("\n\n" + "="*55)
    print("  ANALYSIS COMPLETE")
    print("="*55)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")

    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png'))
    print(f"\n  {len(files)} visualizations generated:")
    for f in files:
        print(f"    {f}")

    nr = main_res['nash_result']
    sr = main_res['stack_result']
    diff = nr['defender_value'] - sr['defender_value']
    print(f"\n  Main Results (R=3, D=2, M=3):")
    print(f"    Nash defender value:     {nr['defender_value']:+.4f}")
    print(f"    Stackelberg def value:   {sr['defender_value']:+.4f}")
    print(f"    Difference:              {diff:+.4f}")
    print(f"    CFR+ exploitability:     {nr['exploitability']:.6f}")
    print(f"    Nash attacker strategy:  {main_res['nash_strats']['attacker']}")
    print(f"    Nash defender strategy:  {main_res['nash_strats']['defender']}")


if __name__ == '__main__':
    main()
