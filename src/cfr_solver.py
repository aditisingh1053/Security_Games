"""
CFR+ Solver for the SEAD NFGSS Game

Implements Counterfactual Regret Minimisation Plus (CFR+) following:
  - Zinkevich et al. (2007): CFR framework
  - Tammelin et al. (2015): CFR+ improvements (non-negative regret clipping,
    linear averaging)
  - Lisy et al. (2016): Adaptation to NFGSS / sequential security games

The solver computes Nash equilibria for two-player zero-sum games.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sead_game import (SEADGame, GameNode, SEADConfig, InfoSetKey,
                        AttackerAction, DefenderAction)


class CFRPlusSolver:
    """
    CFR+ solver for the SEAD game.

    Maintains cumulative regrets and cumulative strategy sums for each
    information set, and iteratively updates them via tree traversal.
    """

    def __init__(self, game: SEADGame):
        self.game = game

        # Cumulative regrets: infoset -> {action_index: cumulative_regret}
        self.att_regrets: Dict[InfoSetKey, np.ndarray] = {}
        self.def_regrets: Dict[InfoSetKey, np.ndarray] = {}

        # Cumulative strategy sums (for computing average strategy)
        self.att_strategy_sum: Dict[InfoSetKey, np.ndarray] = {}
        self.def_strategy_sum: Dict[InfoSetKey, np.ndarray] = {}

        # Initialize
        for is_key, actions in game.att_infosets.items():
            n = len(actions)
            self.att_regrets[is_key] = np.zeros(n)
            self.att_strategy_sum[is_key] = np.zeros(n)

        for is_key, actions in game.def_infosets.items():
            n = len(actions)
            self.def_regrets[is_key] = np.zeros(n)
            self.def_strategy_sum[is_key] = np.zeros(n)

        # Tracking
        self.iteration = 0
        self.exploitability_history = []
        self.att_utility_history = []
        self.def_utility_history = []

    def _get_strategy(self, regrets: np.ndarray) -> np.ndarray:
        """Compute current strategy from regrets via regret matching."""
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        else:
            return np.ones(len(regrets)) / len(regrets)

    def _get_attacker_strategy(self, infoset: InfoSetKey) -> np.ndarray:
        return self._get_strategy(self.att_regrets[infoset])

    def _get_defender_strategy(self, infoset: InfoSetKey) -> np.ndarray:
        return self._get_strategy(self.def_regrets[infoset])

    def get_average_strategy(self, player: str, infoset: InfoSetKey) -> np.ndarray:
        """Get the average strategy (the converging Nash strategy)."""
        if player == 'attacker':
            strat_sum = self.att_strategy_sum[infoset]
        else:
            strat_sum = self.def_strategy_sum[infoset]

        total = strat_sum.sum()
        if total > 0:
            return strat_sum / total
        else:
            n = len(strat_sum)
            return np.ones(n) / n

    def _cfr_traverse(self, node: GameNode,
                      att_reach: float, def_reach: float,
                      chance_reach: float,
                      traversing_player: str) -> float:
        """
        Recursive CFR traversal.

        Returns the counterfactual utility for the traversing player.
        Uses the simultaneous-move structure: at each node, both players
        act, and the outcome may be stochastic (chance).
        """
        if node.is_terminal:
            # Return utility (positive = good for defender)
            if traversing_player == 'defender':
                return node.terminal_utility
            else:
                return -node.terminal_utility

        att_infoset = node.att_infoset
        def_infoset = node.def_infoset
        att_actions = self.game.att_infosets[att_infoset]
        def_actions = self.game.def_infosets[def_infoset]

        att_strategy = self._get_attacker_strategy(att_infoset)
        def_strategy = self._get_defender_strategy(def_infoset)

        n_att = len(att_actions)
        n_def = len(def_actions)

        # Value for each action of the traversing player
        if traversing_player == 'attacker':
            action_values = np.zeros(n_att)
            node_value = 0.0

            for ai, att_a in enumerate(att_actions):
                action_val = 0.0
                for di, def_a in enumerate(def_actions):
                    action_pair = (att_a, def_a)
                    if action_pair not in node.children:
                        continue
                    for prob, child in node.children[action_pair]:
                        # Accumulate intermediate utility
                        if traversing_player == 'defender':
                            immediate = child.terminal_utility if child.is_terminal else 0
                        else:
                            immediate = 0

                        child_val = self._cfr_traverse(
                            child,
                            att_reach * att_strategy[ai],
                            def_reach * def_strategy[di],
                            chance_reach * prob,
                            traversing_player
                        )
                        # Add intermediate payoff from this transition
                        trans_util = -child.terminal_utility if not child.is_terminal else 0
                        action_val += def_strategy[di] * prob * child_val

                action_values[ai] = action_val
                node_value += att_strategy[ai] * action_val

            # Update regrets for attacker
            for ai in range(n_att):
                regret = action_values[ai] - node_value
                # CFR+: clip to non-negative
                self.att_regrets[att_infoset][ai] = max(
                    0, self.att_regrets[att_infoset][ai] + def_reach * regret
                )

            # Update strategy sum with linear weighting (CFR+)
            weight = max(self.iteration, 1)
            self.att_strategy_sum[att_infoset] += att_reach * att_strategy * weight

            return node_value

        else:  # traversing_player == 'defender'
            action_values = np.zeros(n_def)
            node_value = 0.0

            for di, def_a in enumerate(def_actions):
                action_val = 0.0
                for ai, att_a in enumerate(att_actions):
                    action_pair = (att_a, def_a)
                    if action_pair not in node.children:
                        continue
                    for prob, child in node.children[action_pair]:
                        child_val = self._cfr_traverse(
                            child,
                            att_reach * att_strategy[ai],
                            def_reach * def_strategy[di],
                            chance_reach * prob,
                            traversing_player
                        )
                        action_val += att_strategy[ai] * prob * child_val

                action_values[di] = action_val
                node_value += def_strategy[di] * action_val

            # Update regrets for defender
            for di in range(n_def):
                regret = action_values[di] - node_value
                self.def_regrets[def_infoset][di] = max(
                    0, self.def_regrets[def_infoset][di] + att_reach * regret
                )

            weight = max(self.iteration, 1)
            self.def_strategy_sum[def_infoset] += def_reach * def_strategy * weight

            return node_value

    def _compute_expected_utility(self, node: GameNode) -> float:
        """Compute expected utility under current average strategies."""
        if node.is_terminal:
            return node.terminal_utility

        att_infoset = node.att_infoset
        def_infoset = node.def_infoset
        att_actions = self.game.att_infosets[att_infoset]
        def_actions = self.game.def_infosets[def_infoset]

        att_strategy = self.get_average_strategy('attacker', att_infoset)
        def_strategy = self.get_average_strategy('defender', def_infoset)

        total = 0.0
        for ai, att_a in enumerate(att_actions):
            for di, def_a in enumerate(def_actions):
                action_pair = (att_a, def_a)
                if action_pair not in node.children:
                    continue
                for prob, child in node.children[action_pair]:
                    child_val = self._compute_expected_utility(child)
                    total += att_strategy[ai] * def_strategy[di] * prob * child_val

        return total

    def _best_response_value(self, node: GameNode, br_player: str) -> float:
        """
        Compute the value of the best response for br_player
        against the opponent's average strategy.
        """
        if node.is_terminal:
            if br_player == 'defender':
                return node.terminal_utility
            else:
                return -node.terminal_utility

        att_infoset = node.att_infoset
        def_infoset = node.def_infoset
        att_actions = self.game.att_infosets[att_infoset]
        def_actions = self.game.def_infosets[def_infoset]

        if br_player == 'attacker':
            # Attacker best-responds, defender plays average strategy
            def_strategy = self.get_average_strategy('defender', def_infoset)
            best_val = float('-inf')
            for ai, att_a in enumerate(att_actions):
                action_val = 0.0
                for di, def_a in enumerate(def_actions):
                    action_pair = (att_a, def_a)
                    if action_pair not in node.children:
                        continue
                    for prob, child in node.children[action_pair]:
                        child_val = self._best_response_value(child, br_player)
                        action_val += def_strategy[di] * prob * child_val
                best_val = max(best_val, action_val)
            return best_val

        else:  # defender best-responds
            att_strategy = self.get_average_strategy('attacker', att_infoset)
            best_val = float('-inf')
            for di, def_a in enumerate(def_actions):
                action_val = 0.0
                for ai, att_a in enumerate(att_actions):
                    action_pair = (att_a, def_a)
                    if action_pair not in node.children:
                        continue
                    for prob, child in node.children[action_pair]:
                        child_val = self._best_response_value(child, br_player)
                        action_val += att_strategy[ai] * prob * child_val
                best_val = max(best_val, action_val)
            return best_val

    def compute_exploitability(self) -> float:
        """
        Compute exploitability: sum of best-response improvements for both players.
        At Nash equilibrium, exploitability = 0.
        """
        game_value = self._compute_expected_utility(self.game.root)
        att_br_val = self._best_response_value(self.game.root, 'attacker')
        def_br_val = self._best_response_value(self.game.root, 'defender')

        # Exploitability = (attacker BR value - attacker's current value)
        #                + (defender BR value - defender's current value)
        # Current attacker value = -game_value (zero-sum)
        exploitability = (att_br_val - (-game_value)) + (def_br_val - game_value)
        return max(exploitability, 0.0)

    def solve(self, num_iterations: int = 1000, log_interval: int = 50,
              verbose: bool = True) -> dict:
        """
        Run CFR+ for the specified number of iterations.

        Returns a dict with convergence history and final strategies.
        """
        if verbose:
            stats = self.game.get_game_stats()
            print(f"=== CFR+ Solver ===")
            print(f"Game: {stats['total_nodes']} nodes, "
                  f"{stats['terminal_nodes']} terminal, "
                  f"{stats['attacker_infosets']} att infosets, "
                  f"{stats['defender_infosets']} def infosets")
            print(f"Running {num_iterations} iterations...")

        for i in range(1, num_iterations + 1):
            self.iteration = i

            # Traverse for both players (alternating updates)
            self._cfr_traverse(self.game.root, 1.0, 1.0, 1.0, 'attacker')
            self._cfr_traverse(self.game.root, 1.0, 1.0, 1.0, 'defender')

            # Log progress
            if i % log_interval == 0 or i == 1:
                exploit = self.compute_exploitability()
                game_val = self._compute_expected_utility(self.game.root)
                self.exploitability_history.append((i, exploit))
                self.def_utility_history.append((i, game_val))
                self.att_utility_history.append((i, -game_val))

                if verbose:
                    print(f"  Iter {i:5d}: exploitability={exploit:.6f}, "
                          f"def_value={game_val:.4f}")

        # Final results
        final_exploit = self.compute_exploitability()
        final_value = self._compute_expected_utility(self.game.root)

        if verbose:
            print(f"\n=== CFR+ Results ===")
            print(f"Final exploitability: {final_exploit:.6f}")
            print(f"Game value (defender): {final_value:.4f}")
            print(f"Game value (attacker): {-final_value:.4f}")

        return {
            'exploitability': final_exploit,
            'defender_value': final_value,
            'attacker_value': -final_value,
            'exploitability_history': self.exploitability_history,
            'defender_utility_history': self.def_utility_history,
            'attacker_utility_history': self.att_utility_history,
        }

    def get_all_strategies(self) -> dict:
        """Extract readable strategies for both players."""
        result = {'attacker': {}, 'defender': {}}

        for is_key, actions in self.game.att_infosets.items():
            avg_strat = self.get_average_strategy('attacker', is_key)
            action_probs = {a.name: p for a, p in zip(actions, avg_strat) if p > 0.001}
            if action_probs:
                result['attacker'][str(is_key)] = action_probs

        for is_key, actions in self.game.def_infosets.items():
            avg_strat = self.get_average_strategy('defender', is_key)
            action_probs = {a.name: p for a, p in zip(actions, avg_strat) if p > 0.001}
            if action_probs:
                result['defender'][str(is_key)] = action_probs

        return result

    def get_root_strategies(self) -> dict:
        """Get strategies at the root (round 1) information sets."""
        root = self.game.root
        att_is = root.att_infoset
        def_is = root.def_infoset

        att_actions = self.game.att_infosets[att_is]
        def_actions = self.game.def_infosets[def_is]

        att_strat = self.get_average_strategy('attacker', att_is)
        def_strat = self.get_average_strategy('defender', def_is)

        return {
            'attacker': {a.name: float(p) for a, p in zip(att_actions, att_strat)},
            'defender': {a.name: float(p) for a, p in zip(def_actions, def_strat)},
        }
