"""
SEAD Game Model: Normal-Form Game with Sequential Strategies (NFGSS)

Models the Suppression of Enemy Air Defences engagement as a two-player
zero-sum extensive-form game with imperfect information. Each player's
strategy space is defined by a finite acyclic MDP.

Players:
  - Attacker (SEAD aircraft + decoys)
  - Defender (SAM battery)

The game unfolds over R rounds with simultaneous moves each round.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from copy import deepcopy


# ======================================================================
# Action Definitions
# ======================================================================

class AttackerAction(Enum):
    DEPLOY_DECOY = auto()     # Deploy a decoy (costs 1 decoy)
    DEPLOY_REAL = auto()      # Deploy a real aircraft (risks losing it)
    HOLD = auto()             # No action this round
    CEASEFIRE = auto()        # Signal ceasefire

class DefenderAction(Enum):
    FIRE = auto()             # Fire a missile at incoming track (costs 1 missile)
    TRACK_ONLY = auto()       # Maintain radar lock, don't fire
    CONSERVE = auto()         # Power down radar
    CEASEFIRE = auto()        # Accept ceasefire


# ======================================================================
# Game Configuration
# ======================================================================

@dataclass
class SEADConfig:
    """Parameters for a SEAD game instance."""
    num_rounds: int = 3           # R_max: number of engagement rounds
    attacker_decoys: int = 2      # D: initial decoy budget
    attacker_real: int = 1        # A_real: initial real aircraft count
    defender_missiles: int = 3    # M: initial missile inventory

    p_deceive: float = 0.5        # Probability decoy fools the defender
    p_hit_real: float = 0.7       # Probability missile hits real aircraft
    p_hit_decoy: float = 0.9      # Probability missile hits decoy (if fired at)

    # Utility values for different outcomes
    util_hit_real: float = 3.0    # Defender reward for hitting real aircraft
    util_hit_decoy: float = -1.0  # Defender cost for wasting missile on decoy
    util_miss: float = -0.5       # Defender cost for firing and missing
    util_real_survives: float = -2.0  # Defender cost if real aircraft survives
    util_ceasefire: float = 0.0   # Both ceasefire: neutral
    util_hold_vs_hold: float = 0.0
    util_conserve: float = 0.1    # Small benefit for conserving resources


# ======================================================================
# Game State
# ======================================================================

@dataclass(frozen=True)
class AttackerState:
    round: int
    decoys: int
    real_aircraft: int

    def __str__(self):
        return f"Att(r={self.round},d={self.decoys},real={self.real_aircraft})"

@dataclass(frozen=True)
class DefenderState:
    round: int
    missiles: int

    def __str__(self):
        return f"Def(r={self.round},m={self.missiles})"


# ======================================================================
# Information Set Key
# ======================================================================

@dataclass(frozen=True)
class InfoSetKey:
    """Key for an information set. Contains what the player knows."""
    player: str            # 'attacker' or 'defender'
    round: int
    own_resources: tuple   # (decoys, real) for attacker; (missiles,) for defender
    history: tuple         # sequence of (own_action, observed_outcome) pairs

    def __str__(self):
        res_str = f"res={self.own_resources}"
        hist_str = ",".join(str(h) for h in self.history) if self.history else "start"
        return f"IS({self.player},r={self.round},{res_str},hist=[{hist_str}])"


# ======================================================================
# Game Tree Node
# ======================================================================

class GameNode:
    """A node in the SEAD game tree."""

    def __init__(self, att_state: AttackerState, def_state: DefenderState,
                 att_history: tuple = (), def_history: tuple = (),
                 is_terminal: bool = False, terminal_utility: float = 0.0):
        self.att_state = att_state
        self.def_state = def_state
        self.att_history = att_history
        self.def_history = def_history
        self.is_terminal = is_terminal
        self.terminal_utility = terminal_utility  # From defender's perspective

        # Information set keys
        self.att_infoset = InfoSetKey(
            'attacker', att_state.round,
            (att_state.decoys, att_state.real_aircraft),
            att_history
        )
        self.def_infoset = InfoSetKey(
            'defender', def_state.round,
            (def_state.missiles,),
            def_history
        )

        # Children: maps (att_action, def_action) -> list of (prob, child_node)
        self.children: Dict[Tuple, List[Tuple[float, 'GameNode']]] = {}

    def __str__(self):
        return f"Node({self.att_state}, {self.def_state}, term={self.is_terminal})"


# ======================================================================
# SEAD Game Builder
# ======================================================================

class SEADGame:
    """Builds and stores the complete SEAD game tree."""

    def __init__(self, config: SEADConfig):
        self.config = config
        self.root: Optional[GameNode] = None

        # Information set -> list of actions at that infoset
        self.att_infosets: Dict[InfoSetKey, List[AttackerAction]] = {}
        self.def_infosets: Dict[InfoSetKey, List[DefenderAction]] = {}

        # All nodes grouped by infoset
        self.att_infoset_nodes: Dict[InfoSetKey, List[GameNode]] = {}
        self.def_infoset_nodes: Dict[InfoSetKey, List[GameNode]] = {}

        # Flat list of all nodes for traversal
        self.all_nodes: List[GameNode] = []
        self.terminal_nodes: List[GameNode] = []

        # Build the tree
        self._build_tree()

    def _get_attacker_actions(self, state: AttackerState) -> List[AttackerAction]:
        """Available actions for the attacker at this state."""
        actions = [AttackerAction.HOLD]
        if state.decoys > 0:
            actions.append(AttackerAction.DEPLOY_DECOY)
        if state.real_aircraft > 0:
            actions.append(AttackerAction.DEPLOY_REAL)
        actions.append(AttackerAction.CEASEFIRE)
        return actions

    def _get_defender_actions(self, state: DefenderState) -> List[DefenderAction]:
        """Available actions for the defender at this state."""
        actions = [AttackerAction.HOLD]  # will be overwritten
        actions = [DefenderAction.TRACK_ONLY, DefenderAction.CONSERVE]
        if state.missiles > 0:
            actions.append(DefenderAction.FIRE)
        actions.append(DefenderAction.CEASEFIRE)
        return actions

    def _compute_outcomes(self, att_action: AttackerAction,
                          def_action: DefenderAction,
                          att_state: AttackerState,
                          def_state: DefenderState
                          ) -> List[Tuple[float, AttackerState, DefenderState, float, str, str]]:
        """
        Compute (probability, new_att_state, new_def_state, utility, att_obs, def_obs)
        for each stochastic outcome of the action pair.
        Utility is from the defender's perspective.
        """
        cfg = self.config
        next_round = att_state.round + 1
        outcomes = []

        # Both ceasefire
        if att_action == AttackerAction.CEASEFIRE and def_action == DefenderAction.CEASEFIRE:
            new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
            new_def = DefenderState(next_round, def_state.missiles)
            outcomes.append((1.0, new_att, new_def, cfg.util_ceasefire, 'ceasefire', 'ceasefire'))
            return outcomes

        # Attacker ceasefire but defender doesn't
        if att_action == AttackerAction.CEASEFIRE:
            new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
            new_def = DefenderState(next_round, def_state.missiles)
            u = cfg.util_conserve  # Defender benefits slightly
            outcomes.append((1.0, new_att, new_def, u, 'ceasefire', 'no_target'))
            return outcomes

        # Defender ceasefire but attacker doesn't
        if def_action == DefenderAction.CEASEFIRE:
            if att_action == AttackerAction.DEPLOY_REAL:
                new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                u = cfg.util_real_survives  # Real aircraft gets through
                outcomes.append((1.0, new_att, new_def, u, 'deployed_real', 'ceasefire'))
            elif att_action == AttackerAction.DEPLOY_DECOY:
                new_att = AttackerState(next_round, att_state.decoys - 1, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                u = 0.0  # Decoy wasted, no effect
                outcomes.append((1.0, new_att, new_def, u, 'deployed_decoy', 'ceasefire'))
            else:  # HOLD
                new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, 0.0, 'hold', 'ceasefire'))
            return outcomes

        # Attacker HOLD
        if att_action == AttackerAction.HOLD:
            new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
            if def_action == DefenderAction.FIRE:
                # Defender fires at nothing - waste
                new_def = DefenderState(next_round, def_state.missiles - 1)
                u = cfg.util_miss
                outcomes.append((1.0, new_att, new_def, u, 'hold', 'fired_miss'))
            elif def_action == DefenderAction.TRACK_ONLY:
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, cfg.util_hold_vs_hold, 'hold', 'tracked'))
            else:  # CONSERVE
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, cfg.util_conserve, 'hold', 'conserved'))
            return outcomes

        # Attacker DEPLOY_DECOY
        if att_action == AttackerAction.DEPLOY_DECOY:
            new_att_decoys = att_state.decoys - 1

            if def_action == DefenderAction.FIRE:
                # Defender fires at incoming track (which is a decoy)
                # Decoy may or may not fool the defender's tracking
                # If missile hits decoy: defender wastes missile
                p_hit = cfg.p_hit_decoy
                # Hit outcome: missile wasted on decoy
                new_att_h = AttackerState(next_round, new_att_decoys, att_state.real_aircraft)
                new_def_h = DefenderState(next_round, def_state.missiles - 1)
                outcomes.append((p_hit, new_att_h, new_def_h, cfg.util_hit_decoy,
                                'decoy_hit', 'fired_hit_decoy'))
                # Miss outcome
                new_att_m = AttackerState(next_round, new_att_decoys, att_state.real_aircraft)
                new_def_m = DefenderState(next_round, def_state.missiles - 1)
                outcomes.append((1 - p_hit, new_att_m, new_def_m, cfg.util_miss,
                                'decoy_miss', 'fired_miss'))

            elif def_action == DefenderAction.TRACK_ONLY:
                # Defender tracks but doesn't fire - decoy occupies radar
                new_att = AttackerState(next_round, new_att_decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                # Defender identifies decoy over time (partial info)
                p_id = cfg.p_deceive  # prob defender is fooled
                outcomes.append((p_id, new_att, new_def, -0.2,
                                'decoy_fooled', 'tracked_suspicious'))
                outcomes.append((1 - p_id, new_att, new_def, 0.3,
                                'decoy_identified', 'tracked_identified'))

            else:  # CONSERVE
                new_att = AttackerState(next_round, new_att_decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, 0.0,
                                'decoy_undetected', 'conserved'))
            return outcomes

        # Attacker DEPLOY_REAL
        if att_action == AttackerAction.DEPLOY_REAL:
            if def_action == DefenderAction.FIRE:
                p_hit = cfg.p_hit_real
                # Hit: real aircraft destroyed - big defender win
                new_att_h = AttackerState(next_round, att_state.decoys,
                                          att_state.real_aircraft - 1)
                new_def_h = DefenderState(next_round, def_state.missiles - 1)
                outcomes.append((p_hit, new_att_h, new_def_h, cfg.util_hit_real,
                                'real_destroyed', 'fired_hit_real'))
                # Miss: real aircraft survives
                new_att_m = AttackerState(next_round, att_state.decoys,
                                          att_state.real_aircraft)
                new_def_m = DefenderState(next_round, def_state.missiles - 1)
                outcomes.append((1 - p_hit, new_att_m, new_def_m, cfg.util_real_survives,
                                'real_survived', 'fired_miss'))

            elif def_action == DefenderAction.TRACK_ONLY:
                # Real aircraft gets through without being engaged
                new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, cfg.util_real_survives,
                                'real_through', 'tracked_only'))

            else:  # CONSERVE
                # Real aircraft gets through, radar is down
                new_att = AttackerState(next_round, att_state.decoys, att_state.real_aircraft)
                new_def = DefenderState(next_round, def_state.missiles)
                outcomes.append((1.0, new_att, new_def, cfg.util_real_survives * 1.2,
                                'real_through_undetected', 'conserved'))
            return outcomes

        return outcomes

    def _build_tree(self):
        """Build the complete game tree recursively."""
        init_att = AttackerState(1, self.config.attacker_decoys, self.config.attacker_real)
        init_def = DefenderState(1, self.config.defender_missiles)
        self.root = GameNode(init_att, init_def)
        self._expand_node(self.root)

    def _expand_node(self, node: GameNode):
        """Recursively expand a game node."""
        self.all_nodes.append(node)

        # Terminal conditions
        if node.att_state.round > self.config.num_rounds:
            node.is_terminal = True
            # Terminal payoff based on remaining resources
            att_s = node.att_state
            def_s = node.def_state
            resource_bonus = (def_s.missiles / max(self.config.defender_missiles, 1)) * 0.5
            aircraft_penalty = (att_s.real_aircraft / max(self.config.attacker_real, 1)) * (-1.0)
            node.terminal_utility = resource_bonus + aircraft_penalty
            self.terminal_nodes.append(node)
            return

        # Get available actions
        att_actions = self._get_attacker_actions(node.att_state)
        def_actions = self._get_defender_actions(node.def_state)

        # Register information sets
        if node.att_infoset not in self.att_infosets:
            self.att_infosets[node.att_infoset] = att_actions
            self.att_infoset_nodes[node.att_infoset] = []
        self.att_infoset_nodes[node.att_infoset].append(node)

        if node.def_infoset not in self.def_infosets:
            self.def_infosets[node.def_infoset] = def_actions
            self.def_infoset_nodes[node.def_infoset] = []
        self.def_infoset_nodes[node.def_infoset].append(node)

        # Expand for each action pair
        for att_a in att_actions:
            for def_a in def_actions:
                action_pair = (att_a, def_a)
                outcomes = self._compute_outcomes(att_a, def_a,
                                                  node.att_state, node.def_state)
                children = []
                for prob, new_att, new_def, util, att_obs, def_obs in outcomes:
                    if prob <= 0:
                        continue
                    new_att_hist = node.att_history + ((att_a.name, att_obs),)
                    new_def_hist = node.def_history + ((def_a.name, def_obs),)

                    child = GameNode(new_att, new_def,
                                     new_att_hist, new_def_hist)
                    child.terminal_utility = util  # Intermediate utility
                    children.append((prob, child))
                    self._expand_node(child)

                node.children[action_pair] = children

    def get_game_stats(self) -> dict:
        """Return statistics about the game tree."""
        return {
            'total_nodes': len(self.all_nodes),
            'terminal_nodes': len(self.terminal_nodes),
            'attacker_infosets': len(self.att_infosets),
            'defender_infosets': len(self.def_infosets),
            'num_rounds': self.config.num_rounds,
            'attacker_decoys': self.config.attacker_decoys,
            'attacker_real': self.config.attacker_real,
            'defender_missiles': self.config.defender_missiles,
            'p_deceive': self.config.p_deceive,
            'p_hit_real': self.config.p_hit_real,
        }
