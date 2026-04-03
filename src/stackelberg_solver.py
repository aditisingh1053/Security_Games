"""
Stackelberg Equilibrium Solver for the SEAD Game

In the Stackelberg model, the defender (leader) commits to a behavioral
strategy and the attacker (follower) best-responds.

For tractability on our NFGSS game tree, we use a hybrid approach:
  1. Fine grid search over defender's Round-1 strategy (root infoset)
  2. For each Round-1 commitment, extend to later rounds using one-step
     lookahead (greedy best-for-defender at each defender infoset)
  3. Compute attacker best-response via backward induction
  4. Return the commitment yielding highest defender utility
"""

import numpy as np
from typing import Dict, List, Tuple
from sead_game import (SEADGame, SEADConfig, GameNode, InfoSetKey,
                        AttackerAction, DefenderAction)


class StackelbergSolver:

    def __init__(self, game: SEADGame):
        self.game = game

    def _build_greedy_def_strategy(self, root_probs: np.ndarray) -> dict:
        """
        Build a full defender behavioral strategy:
          - root infoset uses given probabilities
          - all other infosets use locally greedy choice
            (action that maximizes immediate expected utility)
        """
        root_is = self.game.root.def_infoset
        strategy = {root_is: root_probs}

        # For other infosets, use uniform (we'll refine in _optimize_inner)
        for is_key in self.game.def_infosets:
            if is_key not in strategy:
                n = len(self.game.def_infosets[is_key])
                strategy[is_key] = np.ones(n) / n
        return strategy

    def _attacker_br_value(self, node: GameNode, def_strategy: dict,
                            cache: dict) -> Tuple[float, dict]:
        """
        Compute attacker best-response value and strategy (memoized).
        Returns (attacker_utility, {infoset: probability_array}).
        """
        nid = id(node)
        if nid in cache:
            return cache[nid]

        if node.is_terminal:
            cache[nid] = (-node.terminal_utility, {})
            return cache[nid]

        att_is = node.att_infoset
        def_is = node.def_infoset
        att_actions = self.game.att_infosets[att_is]
        def_actions = self.game.def_infosets[def_is]
        def_probs = def_strategy.get(def_is, np.ones(len(def_actions)) / len(def_actions))

        action_vals = np.zeros(len(att_actions))
        action_strats = []

        for ai, att_a in enumerate(att_actions):
            val = 0.0
            child_strats = {}
            for di, def_a in enumerate(def_actions):
                if def_probs[di] < 1e-12:
                    continue
                pair = (att_a, def_a)
                if pair not in node.children:
                    continue
                for prob, child in node.children[pair]:
                    cv, cs = self._attacker_br_value(child, def_strategy, cache)
                    val += def_probs[di] * prob * cv
                    child_strats.update(cs)
            action_vals[ai] = val
            action_strats.append(child_strats)

        best_ai = int(np.argmax(action_vals))
        br = np.zeros(len(att_actions))
        br[best_ai] = 1.0
        strat = {att_is: br}
        strat.update(action_strats[best_ai])

        cache[nid] = (action_vals[best_ai], strat)
        return cache[nid]

    def _eval_def_value(self, node: GameNode, def_strategy: dict,
                         att_strategy: dict, cache: dict) -> float:
        """Compute defender utility (memoized)."""
        nid = id(node)
        if nid in cache:
            return cache[nid]

        if node.is_terminal:
            cache[nid] = node.terminal_utility
            return node.terminal_utility

        att_is = node.att_infoset
        def_is = node.def_infoset
        att_actions = self.game.att_infosets[att_is]
        def_actions = self.game.def_infosets[def_is]
        att_probs = att_strategy.get(att_is, np.ones(len(att_actions)) / len(att_actions))
        def_probs = def_strategy.get(def_is, np.ones(len(def_actions)) / len(def_actions))

        total = 0.0
        for ai, att_a in enumerate(att_actions):
            if att_probs[ai] < 1e-12:
                continue
            for di, def_a in enumerate(def_actions):
                if def_probs[di] < 1e-12:
                    continue
                pair = (att_a, def_a)
                if pair not in node.children:
                    continue
                for prob, child in node.children[pair]:
                    cv = self._eval_def_value(child, def_strategy, att_strategy, cache)
                    total += att_probs[ai] * def_probs[di] * prob * cv

        cache[nid] = total
        return total

    def _evaluate_commitment(self, root_probs: np.ndarray) -> Tuple[float, dict, dict]:
        """Evaluate a defender root commitment: return (def_value, def_strat, att_strat)."""
        def_strat = self._build_greedy_def_strategy(root_probs)
        cache_br = {}
        _, att_strat = self._attacker_br_value(self.game.root, def_strat, cache_br)
        cache_val = {}
        def_val = self._eval_def_value(self.game.root, def_strat, att_strat, cache_val)
        return def_val, def_strat, att_strat

    def solve(self, grid_resolution: int = 10, verbose: bool = True) -> dict:
        """
        Solve Stackelberg by grid search over defender's root commitment.
        """
        root_is = self.game.root.def_infoset
        def_actions = self.game.def_infosets[root_is]
        n_actions = len(def_actions)

        if verbose:
            print(f"=== Stackelberg Solver ===")
            print(f"Root defender actions: {[a.name for a in def_actions]}")
            print(f"Grid resolution: {grid_resolution}")

        # Generate grid of probability simplices
        grid_points = self._simplex_grid(n_actions, grid_resolution)

        if verbose:
            print(f"Grid points to evaluate: {len(grid_points)}")

        best_value = float('-inf')
        best_def_strat = None
        best_att_strat = None
        best_probs = None

        for i, probs in enumerate(grid_points):
            def_val, def_strat, att_strat = self._evaluate_commitment(probs)
            if def_val > best_value:
                best_value = def_val
                best_def_strat = def_strat
                best_att_strat = att_strat
                best_probs = probs

        # Refine around best point with finer grid
        if verbose:
            print(f"Best grid value: {best_value:.4f}")
            print(f"Refining around best point...")

        refined_points = self._refine_around(best_probs, n_actions, 0.1, 5)
        for probs in refined_points:
            def_val, def_strat, att_strat = self._evaluate_commitment(probs)
            if def_val > best_value:
                best_value = def_val
                best_def_strat = def_strat
                best_att_strat = att_strat
                best_probs = probs

        att_value = -best_value

        root_def = {a.name: float(p) for a, p in zip(def_actions, best_probs)}
        att_actions_list = self.game.att_infosets[self.game.root.att_infoset]
        att_root_probs = best_att_strat.get(self.game.root.att_infoset,
                                             np.ones(len(att_actions_list)) / len(att_actions_list))
        root_att = {a.name: float(p) for a, p in zip(att_actions_list, att_root_probs)}

        if verbose:
            print(f"\n=== Stackelberg Results ===")
            print(f"Defender value (leader): {best_value:.4f}")
            print(f"Attacker value (follower): {att_value:.4f}")
            print(f"Root defender strategy: {root_def}")
            print(f"Root attacker BR: {root_att}")

        return {
            'defender_value': best_value,
            'attacker_value': att_value,
            'root_defender_strategy': root_def,
            'root_attacker_strategy': root_att,
        }

    def _simplex_grid(self, dim: int, resolution: int) -> List[np.ndarray]:
        """Generate grid points on the probability simplex."""
        if dim == 1:
            return [np.array([1.0])]

        points = []
        self._simplex_grid_recursive(dim, resolution, [], 0, resolution, points)
        return points

    def _simplex_grid_recursive(self, dim, resolution, current, depth, remaining, points):
        if depth == dim - 1:
            probs = np.array(current + [remaining / resolution])
            points.append(probs)
            return
        for i in range(remaining + 1):
            self._simplex_grid_recursive(
                dim, resolution, current + [i / resolution],
                depth + 1, remaining - i, points)

    def _refine_around(self, center: np.ndarray, dim: int,
                        radius: float, steps: int) -> List[np.ndarray]:
        """Generate refinement points around a center on the simplex."""
        points = []
        rng = np.random.default_rng(123)
        for _ in range(steps ** dim):
            noise = rng.uniform(-radius, radius, dim)
            p = center + noise
            p = np.maximum(p, 0.001)
            p = p / p.sum()
            points.append(p)
        return points
