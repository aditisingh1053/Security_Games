"""
Comprehensive Visualization Suite for SEAD Game Analysis

Generates publication-quality figures comparing Nash (CFR+) and
Stackelberg equilibria, showing:

1. CFR+ Convergence         - Exploitability vs iterations (log scale)
2. Strategy Comparison       - Side-by-side bar charts of action probabilities
3. Game Value Comparison     - Defender/attacker payoffs under both solutions
4. Resource Sensitivity      - How equilibrium changes across configurations
5. Game Tree Visualization   - Annotated decision tree with probabilities
6. Strategy Evolution        - How CFR+ strategies evolve during training
7. Payoff Heatmaps           - Defender payoff for action pairs
8. Radar/Spider Charts       - Multi-dimensional strategy comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, List, Tuple, Optional


# Style setup
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Color palettes
NASH_COLOR = '#2196F3'      # Blue for Nash/CFR+
STACK_COLOR = '#FF5722'     # Red-orange for Stackelberg
ATT_COLOR = '#E91E63'       # Pink for attacker
DEF_COLOR = '#4CAF50'       # Green for defender
ACCENT_COLORS = ['#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B']

ACTION_COLORS = {
    'DEPLOY_DECOY': '#FFC107',
    'DEPLOY_REAL': '#F44336',
    'HOLD': '#9E9E9E',
    'CEASEFIRE': '#8BC34A',
    'FIRE': '#FF5722',
    'TRACK_ONLY': '#2196F3',
    'CONSERVE': '#4CAF50',
}


def save_fig(fig, path, name):
    """Save figure to path."""
    full_path = os.path.join(path, name)
    fig.savefig(full_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {full_path}")


# ======================================================================
# 1. CFR+ CONVERGENCE PLOT
# ======================================================================

def plot_convergence(exploit_history: List[Tuple[int, float]],
                     def_util_history: List[Tuple[int, float]],
                     att_util_history: List[Tuple[int, float]],
                     output_dir: str):
    """
    Two-panel plot:
      Left:  Exploitability vs iteration (log-log) with 1/sqrt(T) reference
      Right: Defender and attacker expected utility vs iteration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iters = [x[0] for x in exploit_history]
    exploits = [x[1] for x in exploit_history]

    # Left panel: Exploitability
    ax1.plot(iters, exploits, color=NASH_COLOR, linewidth=2.5,
             label='CFR+ Exploitability', zorder=3)

    # Reference line: O(1/sqrt(T))
    if len(iters) > 1 and exploits[0] > 0:
        ref = [exploits[0] * np.sqrt(iters[0]) / np.sqrt(t) for t in iters]
        ax1.plot(iters, ref, '--', color='#999999', linewidth=1.5,
                 label=r'$O(1/\sqrt{T})$ reference', alpha=0.7)

    ax1.set_xlabel('CFR+ Iteration')
    ax1.set_ylabel('Exploitability')
    ax1.set_title('CFR+ Convergence to Nash Equilibrium')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(left=0)

    # Add annotation for final value
    if exploits:
        final_exp = exploits[-1]
        ax1.annotate(f'Final: {final_exp:.4f}',
                     xy=(iters[-1], final_exp),
                     xytext=(-80, 20), textcoords='offset points',
                     fontsize=9, color=NASH_COLOR,
                     arrowprops=dict(arrowstyle='->', color=NASH_COLOR, lw=1.5))

    # Right panel: Utilities
    def_utils = [x[1] for x in def_util_history]
    att_utils = [x[1] for x in att_util_history]

    ax2.plot(iters, def_utils, color=DEF_COLOR, linewidth=2.5,
             label='Defender utility', zorder=3)
    ax2.plot(iters, att_utils, color=ATT_COLOR, linewidth=2.5,
             label='Attacker utility', zorder=3)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

    ax2.set_xlabel('CFR+ Iteration')
    ax2.set_ylabel('Expected Utility')
    ax2.set_title('Expected Payoffs During CFR+ Training')
    ax2.legend(fontsize=10)
    ax2.set_xlim(left=0)

    # Add final values
    if def_utils:
        ax2.annotate(f'Def: {def_utils[-1]:.3f}',
                     xy=(iters[-1], def_utils[-1]),
                     xytext=(-80, 10), textcoords='offset points',
                     fontsize=9, color=DEF_COLOR)
        ax2.annotate(f'Att: {att_utils[-1]:.3f}',
                     xy=(iters[-1], att_utils[-1]),
                     xytext=(-80, -15), textcoords='offset points',
                     fontsize=9, color=ATT_COLOR)

    fig.suptitle('Phase 1: CFR+ Nash Equilibrium Computation', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, '01_cfr_convergence.png')


# ======================================================================
# 2. STRATEGY COMPARISON (Nash vs Stackelberg)
# ======================================================================

def plot_strategy_comparison(nash_strats: dict, stack_strats: dict,
                              output_dir: str):
    """
    Side-by-side grouped bar charts comparing action probabilities
    at the root information set for both solution concepts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, player in enumerate(['attacker', 'defender']):
        ax = axes[idx]

        nash_s = nash_strats.get(player, {})
        stack_s = stack_strats.get(f'root_{player}_strategy', {})

        all_actions = sorted(set(list(nash_s.keys()) + list(stack_s.keys())))
        if not all_actions:
            continue

        x = np.arange(len(all_actions))
        width = 0.35

        nash_vals = [nash_s.get(a, 0.0) for a in all_actions]
        stack_vals = [stack_s.get(a, 0.0) for a in all_actions]

        bars1 = ax.bar(x - width/2, nash_vals, width,
                       label='Nash (CFR+)', color=NASH_COLOR, alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, stack_vals, width,
                       label='Stackelberg', color=STACK_COLOR, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                        color=NASH_COLOR, fontweight='bold')

        for bar in bars2:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                        color=STACK_COLOR, fontweight='bold')

        # Clean action names
        clean_names = [a.replace('DEPLOY_', 'Deploy\n').replace('TRACK_ONLY', 'Track\nOnly')
                       for a in all_actions]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_names, fontsize=9)
        ax.set_ylabel('Probability')
        ax.set_title(f'{player.capitalize()} Strategy (Round 1)', fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, loc='upper right')

        # Color the bars by action type
        for bar, action in zip(bars1, all_actions):
            if action in ACTION_COLORS:
                bar.set_facecolor(ACTION_COLORS[action])
                bar.set_alpha(0.7)

    fig.suptitle('Strategy Comparison: Nash Equilibrium vs Stackelberg Commitment',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, '02_strategy_comparison.png')


# ======================================================================
# 3. GAME VALUE COMPARISON
# ======================================================================

def plot_value_comparison(nash_result: dict, stack_result: dict,
                           output_dir: str):
    """
    Bar chart comparing defender and attacker values under both solutions.
    Includes annotations explaining what the difference means.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Defender\nPayoff', 'Attacker\nPayoff']
    nash_vals = [nash_result['defender_value'], nash_result['attacker_value']]
    stack_vals = [stack_result['defender_value'], stack_result['attacker_value']]

    x = np.arange(len(categories))
    width = 0.3

    bars1 = ax.bar(x - width/2, nash_vals, width,
                   label='Nash (CFR+)', color=NASH_COLOR, alpha=0.85,
                   edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, stack_vals, width,
                   label='Stackelberg', color=STACK_COLOR, alpha=0.85,
                   edgecolor='white', linewidth=1)

    # Value labels
    for bars, color in [(bars1, NASH_COLOR), (bars2, STACK_COLOR)]:
        for bar in bars:
            h = bar.get_height()
            y_pos = h + 0.02 if h >= 0 else h - 0.06
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=11, fontweight='bold', color=color)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Expected Utility', fontsize=12)
    ax.set_title('Game Values: Nash vs Stackelberg', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')

    # Add explanatory text box
    diff = nash_result['defender_value'] - stack_result['defender_value']
    if diff > 0:
        explain = f"Defender gains {diff:.3f} under Nash\n(commitment hurts in SEAD)"
    elif diff < 0:
        explain = f"Defender gains {-diff:.3f} under Stackelberg\n(commitment advantage)"
    else:
        explain = "Both solutions yield equal payoff"

    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='gray', alpha=0.9)
    ax.text(0.98, 0.02, explain, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=props)

    plt.tight_layout()
    save_fig(fig, output_dir, '03_value_comparison.png')


# ======================================================================
# 4. RESOURCE SENSITIVITY ANALYSIS
# ======================================================================

def plot_resource_sensitivity(sensitivity_data: List[dict], output_dir: str):
    """
    Line plots showing how equilibrium values change as we vary
    resource parameters (decoys, missiles).

    sensitivity_data: list of dicts with keys:
      'param_name', 'param_values', 'nash_def_values', 'stack_def_values',
      'nash_att_values', 'stack_att_values'
    """
    n_params = len(sensitivity_data)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for idx, data in enumerate(sensitivity_data):
        ax = axes[idx]
        pv = data['param_values']

        ax.plot(pv, data['nash_def_values'], 'o-', color=DEF_COLOR,
                linewidth=2.5, markersize=8, label='Nash Defender',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(pv, data['stack_def_values'], 's--', color=DEF_COLOR,
                linewidth=2, markersize=7, label='Stackelberg Defender',
                alpha=0.7, markeredgecolor='white')

        ax.plot(pv, data['nash_att_values'], 'o-', color=ATT_COLOR,
                linewidth=2.5, markersize=8, label='Nash Attacker',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(pv, data['stack_att_values'], 's--', color=ATT_COLOR,
                linewidth=2, markersize=7, label='Stackelberg Attacker',
                alpha=0.7, markeredgecolor='white')

        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        ax.set_xlabel(data['param_name'], fontsize=11)
        ax.set_ylabel('Expected Utility', fontsize=11)
        ax.set_title(f'Sensitivity to {data["param_name"]}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks(pv)

    fig.suptitle('Resource Sensitivity: How Equilibria Shift with Parameters',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, '04_resource_sensitivity.png')


# ======================================================================
# 5. STRATEGY EVOLUTION DURING CFR+
# ======================================================================

def plot_strategy_evolution(evolution_data: dict, output_dir: str):
    """
    Stacked area chart showing how action probabilities evolve
    over CFR+ iterations for both players.

    evolution_data: {'attacker': {iteration: {action: prob}},
                     'defender': {iteration: {action: prob}}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, player in enumerate(['attacker', 'defender']):
        ax = axes[idx]
        data = evolution_data.get(player, {})
        if not data:
            continue

        iters = sorted(data.keys())
        all_actions = sorted(set(a for d in data.values() for a in d.keys()))

        # Build arrays for stacked area
        action_series = {a: [] for a in all_actions}
        for it in iters:
            for a in all_actions:
                action_series[a].append(data[it].get(a, 0.0))

        # Stack the areas
        bottom = np.zeros(len(iters))
        for a in all_actions:
            vals = np.array(action_series[a])
            color = ACTION_COLORS.get(a, '#999999')
            clean_name = a.replace('DEPLOY_', 'Deploy ').replace('TRACK_ONLY', 'Track Only')
            ax.fill_between(iters, bottom, bottom + vals,
                           alpha=0.7, color=color, label=clean_name)
            bottom += vals

        ax.set_xlabel('CFR+ Iteration')
        ax.set_ylabel('Action Probability')
        ax.set_title(f'{player.capitalize()} Strategy Evolution', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='upper right', ncol=2)

    fig.suptitle('How Strategies Evolve During CFR+ Training',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, '05_strategy_evolution.png')


# ======================================================================
# 6. PAYOFF HEATMAP
# ======================================================================

def plot_payoff_heatmap(game, output_dir: str):
    """
    Heatmap of expected defender payoffs for each (attacker action, defender action)
    pair at the initial state.
    """
    from sead_game import AttackerAction, DefenderAction

    root = game.root
    att_actions = game.att_infosets[root.att_infoset]
    def_actions = game.def_infosets[root.def_infoset]

    n_att = len(att_actions)
    n_def = len(def_actions)

    payoff_matrix = np.zeros((n_att, n_def))

    for ai, att_a in enumerate(att_actions):
        for di, def_a in enumerate(def_actions):
            pair = (att_a, def_a)
            if pair in root.children:
                val = sum(p * c.terminal_utility for p, c in root.children[pair])
                payoff_matrix[ai, di] = val

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(payoff_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-3, vmax=3)

    # Labels
    att_labels = [a.name.replace('DEPLOY_', 'Deploy\n') for a in att_actions]
    def_labels = [a.name.replace('TRACK_ONLY', 'Track\nOnly') for a in def_actions]

    ax.set_xticks(range(n_def))
    ax.set_xticklabels(def_labels, fontsize=9)
    ax.set_yticks(range(n_att))
    ax.set_yticklabels(att_labels, fontsize=9)

    ax.set_xlabel('Defender Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attacker Action', fontsize=12, fontweight='bold')
    ax.set_title('Round 1 Payoff Matrix (Defender Perspective)\n'
                 'Green = Good for Defender, Red = Bad for Defender',
                 fontsize=12, fontweight='bold')

    # Add text annotations
    for ai in range(n_att):
        for di in range(n_def):
            val = payoff_matrix[ai, di]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(di, ai, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Defender Expected Utility', fontsize=10)

    plt.tight_layout()
    save_fig(fig, output_dir, '06_payoff_heatmap.png')


# ======================================================================
# 7. COMPREHENSIVE DASHBOARD
# ======================================================================

def plot_dashboard(nash_result: dict, stack_result: dict,
                   nash_strats: dict, stack_strats: dict,
                   game_stats: dict, output_dir: str):
    """
    Single-figure dashboard summarizing the entire comparison.
    4 panels: game info, strategy bars, value comparison, key findings.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Game configuration info
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    info_text = (
        f"SEAD Game Configuration\n"
        f"{'='*30}\n\n"
        f"Rounds:          {game_stats['num_rounds']}\n"
        f"Attacker Decoys: {game_stats['attacker_decoys']}\n"
        f"Attacker Real:   {game_stats['attacker_real']}\n"
        f"Defender Missiles:{game_stats['defender_missiles']}\n"
        f"P(deceive):      {game_stats['p_deceive']}\n"
        f"P(hit real):     {game_stats['p_hit_real']}\n\n"
        f"Game Tree Size\n"
        f"{'='*30}\n\n"
        f"Total nodes:     {game_stats['total_nodes']}\n"
        f"Terminal nodes:  {game_stats['terminal_nodes']}\n"
        f"Att info sets:   {game_stats['attacker_infosets']}\n"
        f"Def info sets:   {game_stats['defender_infosets']}"
    )
    ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

    # Panel 2: Defender strategy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    nash_ds = nash_strats.get('defender', {})
    stack_ds = stack_strats.get('root_defender_strategy', {})
    all_da = sorted(set(list(nash_ds.keys()) + list(stack_ds.keys())))
    if all_da:
        x = np.arange(len(all_da))
        w = 0.35
        ax2.bar(x - w/2, [nash_ds.get(a, 0) for a in all_da], w,
                label='Nash', color=NASH_COLOR, alpha=0.85)
        ax2.bar(x + w/2, [stack_ds.get(a, 0) for a in all_da], w,
                label='Stackelberg', color=STACK_COLOR, alpha=0.85)
        clean = [a.replace('DEPLOY_', 'Dep.\n').replace('TRACK_ONLY', 'Track') for a in all_da]
        ax2.set_xticks(x)
        ax2.set_xticklabels(clean, fontsize=8)
        ax2.set_ylabel('Probability')
        ax2.set_title('Defender Strategy', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.15)

    # Panel 3: Attacker strategy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    nash_as = nash_strats.get('attacker', {})
    stack_as = stack_strats.get('root_attacker_strategy', {})
    all_aa = sorted(set(list(nash_as.keys()) + list(stack_as.keys())))
    if all_aa:
        x = np.arange(len(all_aa))
        w = 0.35
        ax3.bar(x - w/2, [nash_as.get(a, 0) for a in all_aa], w,
                label='Nash', color=NASH_COLOR, alpha=0.85)
        ax3.bar(x + w/2, [stack_as.get(a, 0) for a in all_aa], w,
                label='Stackelberg', color=STACK_COLOR, alpha=0.85)
        clean = [a.replace('DEPLOY_', 'Dep.\n').replace('TRACK_ONLY', 'Track') for a in all_aa]
        ax3.set_xticks(x)
        ax3.set_xticklabels(clean, fontsize=8)
        ax3.set_ylabel('Probability')
        ax3.set_title('Attacker Strategy', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 1.15)

    # Panel 4: Value comparison bars
    ax4 = fig.add_subplot(gs[1, 0])
    cats = ['Defender', 'Attacker']
    nv = [nash_result['defender_value'], nash_result['attacker_value']]
    sv = [stack_result['defender_value'], stack_result['attacker_value']]
    x = np.arange(2)
    w = 0.3
    ax4.bar(x - w/2, nv, w, label='Nash', color=NASH_COLOR, alpha=0.85)
    ax4.bar(x + w/2, sv, w, label='Stackelberg', color=STACK_COLOR, alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cats)
    ax4.axhline(0, color='black', lw=0.5)
    ax4.set_ylabel('Expected Utility')
    ax4.set_title('Game Values', fontweight='bold')
    ax4.legend(fontsize=8)

    for i, (v1, v2) in enumerate(zip(nv, sv)):
        ax4.text(i - w/2, v1 + 0.02 if v1 >= 0 else v1 - 0.06,
                 f'{v1:.3f}', ha='center', fontsize=8, color=NASH_COLOR,
                 fontweight='bold')
        ax4.text(i + w/2, v2 + 0.02 if v2 >= 0 else v2 - 0.06,
                 f'{v2:.3f}', ha='center', fontsize=8, color=STACK_COLOR,
                 fontweight='bold')

    # Panel 5: Exploitability convergence (small)
    ax5 = fig.add_subplot(gs[1, 1])
    if nash_result.get('exploitability_history'):
        iters = [x[0] for x in nash_result['exploitability_history']]
        exploits = [x[1] for x in nash_result['exploitability_history']]
        ax5.plot(iters, exploits, color=NASH_COLOR, linewidth=2)
        ax5.set_yscale('log')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Exploitability (log)')
        ax5.set_title('CFR+ Convergence', fontweight='bold')

    # Panel 6: Key findings text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    nash_dv = nash_result['defender_value']
    stack_dv = stack_result['defender_value']
    diff = nash_dv - stack_dv

    findings = (
        f"Key Findings\n"
        f"{'='*30}\n\n"
    )
    if diff > 0.01:
        findings += (
            f"Nash > Stackelberg for Defender\n"
            f"  (by {diff:.3f})\n\n"
            f"In SEAD, the commitment\n"
            f"assumption HURTS the defender.\n"
            f"Without ability to commit,\n"
            f"the simultaneous-move Nash\n"
            f"equilibrium is more favorable.\n"
        )
    elif diff < -0.01:
        findings += (
            f"Stackelberg > Nash for Defender\n"
            f"  (by {-diff:.3f})\n\n"
            f"Commitment power benefits the\n"
            f"defender in this configuration.\n"
            f"Being able to commit to a\n"
            f"strategy and have the attacker\n"
            f"observe it is advantageous.\n"
        )
    else:
        findings += (
            f"Nash ~ Stackelberg\n"
            f"  (difference: {diff:.3f})\n\n"
            f"Both solution concepts yield\n"
            f"similar payoffs. The commitment\n"
            f"assumption has minimal impact.\n"
        )

    findings += (
        f"\nNash exploitability:\n"
        f"  {nash_result['exploitability']:.6f}\n"
    )

    ax6.text(0.05, 0.95, findings, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8))

    fig.suptitle('SEAD Game Analysis Dashboard: Nash (CFR+) vs Stackelberg',
                 fontsize=16, fontweight='bold', y=1.01)
    save_fig(fig, output_dir, '07_dashboard.png')


# ======================================================================
# 8. STRATEGY PROFILE DETAILED VIEW
# ======================================================================

def plot_strategy_profile(all_strategies: dict, player: str,
                           method_name: str, output_dir: str):
    """
    Detailed view of a player's strategy across all information sets.
    Shows a horizontal bar chart for each info set with action probabilities.
    """
    strats = all_strategies.get(player, {})
    if not strats:
        return

    # Take up to 15 most interesting infosets (non-uniform strategies)
    interesting = {}
    for is_key, action_probs in strats.items():
        if len(action_probs) > 1:
            entropy = -sum(p * np.log(p + 1e-10) for p in action_probs.values())
            max_entropy = np.log(len(action_probs))
            if max_entropy > 0 and entropy < 0.95 * max_entropy:
                interesting[is_key] = action_probs

    if not interesting:
        interesting = dict(list(strats.items())[:10])

    items = list(interesting.items())[:15]
    n = len(items)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.8)))

    y_pos = np.arange(n)
    for i, (is_key, action_probs) in enumerate(items):
        left = 0
        for action, prob in sorted(action_probs.items(), key=lambda x: -x[1]):
            color = ACTION_COLORS.get(action, '#999999')
            bar = ax.barh(i, prob, left=left, height=0.6,
                         color=color, edgecolor='white', linewidth=0.5)
            if prob > 0.08:
                label = action.replace('DEPLOY_', 'D-').replace('TRACK_ONLY', 'Track')
                ax.text(left + prob/2, i, f'{label}\n{prob:.0%}',
                        ha='center', va='center', fontsize=7, fontweight='bold')
            left += prob

    # Clean up info set labels
    short_labels = []
    for is_key, _ in items:
        label = is_key
        if len(label) > 40:
            label = label[:37] + '...'
        short_labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Action Probability')
    ax.set_title(f'{player.capitalize()} Strategy Profile ({method_name})',
                 fontsize=13, fontweight='bold')

    # Legend
    handles = []
    for action, color in ACTION_COLORS.items():
        handles.append(mpatches.Patch(color=color, label=action.replace('DEPLOY_', 'Deploy ')))
    ax.legend(handles=handles, loc='lower right', fontsize=8, ncol=2)

    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, output_dir,
             f'08_strategy_profile_{player}_{method_name.lower()}.png')


# ======================================================================
# 9. MULTI-CONFIG COMPARISON TABLE
# ======================================================================

def plot_config_comparison_table(configs_results: List[dict], output_dir: str):
    """
    Table figure comparing results across multiple game configurations.
    Each row is a config, columns show Nash vs Stackelberg values.
    """
    n = len(configs_results)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(14, max(3, 1.5 + n * 0.6)))
    ax.axis('off')

    headers = ['Config', 'Rounds', 'Decoys', 'Missiles', 'p_dec',
               'Nash Def', 'Stack Def', 'Diff', 'Nash Exploit']
    n_cols = len(headers)

    # Draw table
    cell_colors = []
    table_data = []

    for i, cr in enumerate(configs_results):
        cfg = cr['config']
        nr = cr['nash_result']
        sr = cr['stack_result']
        diff = nr['defender_value'] - sr['defender_value']

        row = [
            f"Config {i+1}",
            str(cfg.num_rounds),
            str(cfg.attacker_decoys),
            str(cfg.defender_missiles),
            f"{cfg.p_deceive:.1f}",
            f"{nr['defender_value']:.3f}",
            f"{sr['defender_value']:.3f}",
            f"{diff:+.3f}",
            f"{nr['exploitability']:.4f}",
        ]
        table_data.append(row)

        colors = ['#f5f5f5'] * n_cols
        if diff > 0.01:
            colors[7] = '#c8e6c9'  # Green - Nash better
        elif diff < -0.01:
            colors[7] = '#ffcdd2'  # Red - Stackelberg better
        cell_colors.append(colors)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellColours=cell_colors,
                     colColours=['#bbdefb'] * n_cols,
                     loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Bold header
    for j in range(n_cols):
        table[0, j].set_text_props(fontweight='bold')

    ax.set_title('Comparison Across Game Configurations\n'
                 '(Green = Nash better for Defender, Red = Stackelberg better)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    save_fig(fig, output_dir, '09_config_comparison.png')
