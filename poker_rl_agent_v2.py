#!/usr/bin/env python3
"""
Poker Reinforcement Learning Agent
===================================
A reinforcement learning project for poker using:
- Q-Learning (table-based)
- PPO (Proximal Policy Optimization with neural networks)

Training:
1. Against a random agent
2. Self-play against previous versions

Dependencies: pip install pypokerengine torch numpy matplotlib tqdm
"""

import random
import pickle
import copy
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyPokerEngine
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

# PyTorch for PPO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

ACTIONS = ["fold", "call", "raise_min", "raise_half_pot", "raise_pot", "all_in"]
NUM_ACTIONS = len(ACTIONS)

# Q-Learning hyperparameters
Q_LEARNING_CONFIG = {
    "alpha": 0.1,        # Learning rate
    "gamma": 0.95,       # Discount factor
    "epsilon": 1.0,      # Initial exploration
    "epsilon_min": 0.05,  # Exploration minimale
    "epsilon_decay": 0.9995,  # Exploration decay
}

# PPO hyperparameters
PPO_CONFIG = {
    "lr": 3e-4,           # Learning rate (faster with clean reward signal)
    "gamma": 0.99,        # Discount factor
    "gae_lambda": 0.95,   # GAE lambda (less bias than 0.90)
    "clip_epsilon": 0.2,  # PPO clipping

    "entropy_coef": 0.05, # Entropy coefficient (more exploration)
    "value_coef": 0.5,    # Value coefficient
    "epochs": 4,          # Epochs per update
    "batch_size": 64,     # Batch size (was 256, too large for heads-up)
}

# DQN hyperparameters
DQN_CONFIG = {
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9995,
    "buffer_size": 10000,
    "batch_size": 128,
    "target_update": 10, # Update target network every N episodes
}


# ============================================================================
# UTILITIES
# ============================================================================

def extract_state_features(hole_card: List[str], round_state: Dict[str, Any]) -> Tuple:
    """
    Extracts state features for Q-Learning.
    Returns a hashable tuple representing the state.
    """
    # Cartes en main
    cards = gen_cards(hole_card)
    
    # Hand strength estimation (simplified)
    community = round_state.get("community_card", [])
    if community:
        community_cards = gen_cards(community)
        # Basic win rate estimation (100 simulations for speed)
        try:
            win_rate = estimate_hole_card_win_rate(
                nb_simulation=100,
                nb_player=2,
                hole_card=cards,
                community_card=community_cards
            )
        except:
            win_rate = 0.5
    else:
        # Preflop estimation based on cards
        card_values = [card.rank for card in cards]
        same_suit = cards[0].suit == cards[1].suit
        pair = card_values[0] == card_values[1]
        high_card = max(card_values)
        
        # Simplified score
        if pair:
            win_rate = 0.6 + high_card * 0.02
        elif same_suit:
            win_rate = 0.4 + high_card * 0.02
        else:
            win_rate = 0.3 + high_card * 0.015
        win_rate = min(0.95, max(0.1, win_rate))
    
    # Win rate discretization
    win_rate_bucket = int(win_rate * 10)
    
    # Game phase
    street = round_state.get("street", "preflop")
    street_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    street_idx = street_map.get(street, 0)
    
    # Pot position
    pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
    pot_bucket = min(pot // 50, 10)  # Buckets of 50
    
    # Previous actions in round
    action_histories = round_state.get("action_histories", {})
    num_raises = sum(
        1 for street_actions in action_histories.values()
        for action in street_actions
        if action.get("action") == "raise"
    )
    num_raises_bucket = min(num_raises, 5)
    
    return (win_rate_bucket, street_idx, pot_bucket, num_raises_bucket)

def extract_state_vector(hole_card: List[str], round_state: Dict[str, Any], valid_actions: List[Dict] = None, my_uuid: str = None) -> np.ndarray:
    """
    Extracts feature vector for the PPO neural network.
    """
    MAX_STACK = 2000.0 # Total chips in play usually
    features = []

    if my_uuid is None:
        my_uuid = round_state.get("next_player")  # fallback
    
    # 0. Position (Am I Dealer?)
    dealer_btn = round_state.get("dealer_btn")
    seats = round_state.get("seats", [])
    my_seat_idx = -1
    opponent_stack = 0
    my_stack = 0
    
    for i, seat in enumerate(seats):
        if seat.get("uuid") == my_uuid:
            my_seat_idx = i
            my_stack = seat.get("stack", 0)
        else:
            # Assumes Heads-Up for Opponent Stack
            if seat.get("state") != "folded":
                 opponent_stack = seat.get("stack", 0)
    
    is_dealer = 1.0 if my_seat_idx == dealer_btn else 0.0
    features.append(is_dealer)
    
    # Hole cards (simplified one-hot encoding)
    cards = gen_cards(hole_card)
    
    # Ranks and suits
    for card in cards:
        # Rank (2-14 normalized)
        features.append(card.rank / 14.0)
        # Suit (0-3 normalized)
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        features.append(suit_map.get(card.suit, 0) / 3.0)
    
    # Cartes communautaires
    community = round_state.get("community_card", [])
    community_cards = gen_cards(community) if community else []
    
    for i in range(5):
        if i < len(community_cards):
            features.append(community_cards[i].rank / 14.0)
            suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
            features.append(suit_map.get(community_cards[i].suit, 0) / 3.0)
        else:
            features.append(0.0)
            features.append(0.0)
    
    # Game phase (one-hot)
    street = round_state.get("street", "preflop")
    street_map = {"preflop": [1,0,0,0], "flop": [0,1,0,0], "turn": [0,0,1,0], "river": [0,0,0,1]}
    features.extend(street_map.get(street, [1,0,0,0]))
    
    # Normalized pot
    pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
    features.append(min(pot / MAX_STACK, 1.0))
    
    # Normalized player stack (Me)
    features.append(min(my_stack / MAX_STACK, 1.0))
    
    # Normalized opponent stack (NEW)
    features.append(min(opponent_stack / MAX_STACK, 1.0))
    
    # Number of active players (Normalized)
    active_players = sum(1 for s in seats if s.get("state") == "participating")
    features.append(active_players / 6.0)
    
    # Estimation of win rate (simplified for speed/stability)
    if community:
        try:
            win_rate = estimate_hole_card_win_rate(
                nb_simulation=50,
                nb_player=2,
                hole_card=cards,
                community_card=community_cards
            )
        except:
            win_rate = 0.5
    else:
        # Improved preflop heuristic
        card_values = [card.rank for card in cards]
        high = max(card_values)
        low = min(card_values)
        gap = abs(card_values[0] - card_values[1])
        pair = 1 if card_values[0] == card_values[1] else 0
        suited = 1 if cards[0].suit == cards[1].suit else 0
        connected = 1 if gap <= 2 else 0

        win_rate = 0.30
        win_rate += high / 28.0         # High card [0, 0.5]
        win_rate += low / 56.0          # Kicker [0, 0.25]
        win_rate += 0.12 * pair         # Pair bonus
        win_rate += 0.04 * suited       # Suited bonus
        win_rate += 0.03 * connected    # Connectivity
        win_rate -= 0.02 * max(0, gap - 4)  # Gap penalty
        win_rate = min(0.95, max(0.15, win_rate))
    features.append(win_rate)
    
    # --- CRITICAL NEW FEATURES (Opponent Context) ---
    
    # 1. Cost to call (Call Amount) - Normalized
    call_amount = 0
    valid_actions = round_state.get("valid_actions", []) # Need to ensure this is passed or inferred? 
    # Actually round_state usually doesn't have valid_actions inside it in pypokerengine structure passed here?
    # Wait, extract_state_vector signature is (hole_card, round_state).
    # We might not have valid_actions here easily without passing them.
    # Feature engineering from round_state ONLY:
    
    pot_amount = round_state.get("pot", {}).get("main", {}).get("amount", 0)
    
    # Infer call amount from effective stacks? Hard without valid_actions.
    # But usually 'action_histories' tells us the last bet.
    
    # 2. Aggression (Number of raises in this street)
    action_histories = round_state.get("action_histories", {})
    current_street_actions = action_histories.get(street, [])
    num_raises = sum(1 for a in current_street_actions if a["action"] == "raise")
    features.append(min(num_raises / 5.0, 1.0))
    
    # 3. Did opponent raise last?
    opponent_raised = 0.0
    if current_street_actions:
        last_action = current_street_actions[-1]
        # Fix KeyError: action history uses 'uuid', receive_game_update uses 'player_uuid'
        last_uuid = last_action.get("uuid") or last_action.get("player_uuid")
        if last_uuid != round_state.get("next_player") and last_action["action"] == "raise":
            opponent_raised = 1.0
    features.append(opponent_raised)

    # 4. Pot Odds (Cost / (Pot + Cost))
    call_amount = 0
    if valid_actions:
         for a in valid_actions:
             if a["action"] == "call":
                 call_amount = a["amount"]
                 break
    
    # Use MAX_STACK for consistency if needed, but ratio is unitless
    features.append(min(call_amount / MAX_STACK, 1.0)) # NEW: Explicit call cost
    
    total_pot = pot_amount + call_amount
    pot_odds = call_amount / total_pot if total_pot > 0 else 0
    features.append(pot_odds)
    
    return np.array(features, dtype=np.float32)

# ============================================================================
# AGENT RANDOM (BASELINE)
# ============================================================================

class RandomPlayer(BasePokerPlayer):
    """Random player agent (baseline)"""
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        action = random.choice(valid_actions)
        action_type = action["action"]
        
        if action_type == "raise":
            amount = action["amount"]
            if isinstance(amount, dict):
                min_amount = max(0, amount.get("min", 0))
                max_amount = max(min_amount, amount.get("max", min_amount))
                if max_amount > min_amount:
                    amount = random.randint(min_amount, min(max_amount, min_amount * 3 + 1))
                else:
                    amount = min_amount
            return action_type, max(0, amount)
        else:
            return action_type, action["amount"]
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], round_state: Dict) -> None:
        pass

# ============================================================================
# HEURISTIC AGENTS (FOR EVALUATION)
# ============================================================================

class TightAggressivePlayer(BasePokerPlayer):
    """
    Agent Tight-Aggressive (TAG) - Classic effective poker strategy.
    - Tight: Only plays good hands
    - Aggressive: Raise souvent quand il joue
    - Position-aware: Plus loose en position dealer
    """
    
    def __init__(self, tightness: float = 0.55, aggression: float = 0.75):
        super().__init__()
        self.tightness = tightness  # Threshold to play a hand (0-1) - LOWERED from 0.6
        self.aggression = aggression  # Probability of raise vs call - INCREASED from 0.7
        self.bluff_frequency = 0.12  # Occasional bluffs (12%)
    
    def _estimate_hand_strength(self, hole_card: List[str], round_state: Dict) -> float:
        """Estimates hand strength."""
        cards = gen_cards(hole_card)
        community = round_state.get("community_card", [])
        
        if community:
            community_cards = gen_cards(community)
            try:
                return estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=cards,
                    community_card=community_cards
                )
            except:
                return 0.5
        else:
            # Preflop evaluation
            card_values = sorted([card.rank for card in cards], reverse=True)
            same_suit = cards[0].suit == cards[1].suit
            pair = card_values[0] == card_values[1]
            connected = abs(card_values[0] - card_values[1]) <= 2
            
            strength = 0.3
            if pair:
                strength = 0.5 + card_values[0] * 0.03  # Paires: 0.56-0.92
            elif card_values[0] >= 12:  # As ou Roi
                strength = 0.45 + card_values[1] * 0.02
            elif same_suit and connected:
                strength = 0.4 + max(card_values) * 0.015
            elif same_suit:
                strength = 0.35 + max(card_values) * 0.01
            elif connected:
                strength = 0.32 + max(card_values) * 0.01
            else:
                strength = 0.25 + max(card_values) * 0.01
            
            return min(0.95, max(0.15, strength))
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        
        # Position awareness: looser on dealer button
        dealer_btn = round_state.get("dealer_btn", 0)
        seats = round_state.get("seats", [])
        my_position = -1
        for i, seat in enumerate(seats):
            if seat.get("uuid") == self.uuid:
                my_position = i
                break
        
        is_dealer = (my_position == dealer_btn)
        position_adjustment = 0.08 if is_dealer else 0.0  # Looser on button
        adjusted_tightness = self.tightness - position_adjustment
        
        # Decision based on hand strength
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # Occasional bluff (even with weak hand)
        if random.random() < self.bluff_frequency and "raise" in valid_action_types:
            if hand_strength > 0.25:  # Not completely trash
                raise_info = valid_action_types["raise"]["amount"]
                if isinstance(raise_info, dict):
                    min_raise = max(0, raise_info.get("min", 0))
                    return "raise", min_raise  # Small bluff
        
        # Main trop faible -> fold (sauf si check gratuit)
        if hand_strength < adjusted_tightness * 0.5:
            if "call" in valid_action_types and valid_action_types["call"]["amount"] == 0:
                return "call", 0  # Check gratuit
            return "fold", 0
        
        # Medium hand -> call or fold depending on cost
        elif hand_strength < adjusted_tightness:
            if "call" in valid_action_types:
                call_amount = valid_action_types["call"]["amount"]
                pot = round_state.get("pot", {}).get("main", {}).get("amount", 1)
                pot_odds = call_amount / (pot + call_amount + 1)
                
                if pot_odds < hand_strength * 0.85:  # Better pot odds calculation
                    return "call", call_amount
            return "fold", 0
        
        # Bonne main -> jouer agressivement
        else:
            if random.random() < self.aggression and "raise" in valid_action_types:
                raise_info = valid_action_types["raise"]["amount"]
                if isinstance(raise_info, dict):
                    min_raise = max(0, raise_info.get("min", 0))
                    max_raise = max(min_raise, raise_info.get("max", min_raise))
                    # Raise proportional to strenght
                    raise_pct = (hand_strength - self.tightness) / (1 - self.tightness)
                    amount = int(min_raise + (max_raise - min_raise) * raise_pct * 0.5)
                    return "raise", max(min_raise, min(amount, max_raise))
                else:
                    return "raise", max(0, raise_info) if raise_info else 0
            elif "call" in valid_action_types:
                return "call", valid_action_types["call"]["amount"]
            return "fold", 0
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        pass


class LooseAggressivePlayer(BasePokerPlayer):
    """
    Agent Loose-Aggressive (LAG) - Plays many hands aggressively.
    - Loose: Plays many hands (even weak ones)
    - Aggressive: Raises and bluffs often
    """
    
    def __init__(self, looseness: float = 0.3, aggression: float = 0.8):
        super().__init__()
        self.looseness = looseness  # Low threshold to play (lower = more hands)
        self.aggression = aggression  # Probability of raise vs call
    
    def _estimate_hand_strength(self, hole_card: List[str], round_state: Dict) -> float:
        """Estimates hand strength."""
        cards = gen_cards(hole_card)
        community = round_state.get("community_card", [])
        
        if community:
            community_cards = gen_cards(community)
            try:
                return estimate_hole_card_win_rate(
                    nb_simulation=50,
                    nb_player=2,
                    hole_card=cards,
                    community_card=community_cards
                )
            except:
                return 0.5
        else:
            card_values = sorted([card.rank for card in cards], reverse=True)
            same_suit = cards[0].suit == cards[1].suit
            pair = card_values[0] == card_values[1]
            
            strength = 0.35
            if pair:
                strength = 0.5 + card_values[0] * 0.025
            elif card_values[0] >= 10:
                strength = 0.4 + card_values[1] * 0.015
            elif same_suit:
                strength = 0.38 + max(card_values) * 0.01
            else:
                strength = 0.3 + max(card_values) * 0.01
            
            return min(0.9, max(0.2, strength))
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # LAG plays almost all hands
        if hand_strength < self.looseness * 0.3:
            if "call" in valid_action_types and valid_action_types["call"]["amount"] == 0:
                return "call", 0
            return "fold", 0
        
        # Agressive: raises often, even with medium hands (bluff)
        if random.random() < self.aggression and "raise" in valid_action_types:
            raise_info = valid_action_types["raise"]["amount"]
            if isinstance(raise_info, dict):
                min_raise = max(0, raise_info.get("min", 0))
                max_raise = max(min_raise, raise_info.get("max", min_raise))
                # LAG often makes agressive raises
                bluff_factor = random.uniform(0.3, 0.8)
                amount = int(min_raise + (max_raise - min_raise) * bluff_factor)
                return "raise", max(min_raise, min(amount, max_raise))
            else:
                return "raise", max(0, raise_info) if raise_info else 0
        
        if "call" in valid_action_types:
            return "call", valid_action_types["call"]["amount"]
        return "fold", 0
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        pass


class CallingStationPlayer(BasePokerPlayer):
    """
    Agent Calling Station - Plays passively, calls often.
    Easy to exploit but more realistic than random.
    """
    
    def __init__(self, call_threshold: float = 0.3):
        super().__init__()
        self.call_threshold = call_threshold
    
    def _estimate_hand_strength(self, hole_card: List[str], round_state: Dict) -> float:
        """Estimates hand strength."""
        cards = gen_cards(hole_card)
        community = round_state.get("community_card", [])
        
        if community:
            community_cards = gen_cards(community)
            try:
                return estimate_hole_card_win_rate(
                    nb_simulation=50,
                    nb_player=2,
                    hole_card=cards,
                    community_card=community_cards
                )
            except:
                return 0.5
        else:
            card_values = [card.rank for card in cards]
            pair = card_values[0] == card_values[1]
            high_card = max(card_values)
            return 0.4 + high_card * 0.02 + (0.15 if pair else 0)
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # Very rarely folds
        if hand_strength < self.call_threshold:
            if "call" in valid_action_types and valid_action_types["call"]["amount"] == 0:
                return "call", 0
            if random.random() < 0.3:  # 30% of fold when hand is weak
                return "fold", 0
        
        # Appelle presque toujours
        if "call" in valid_action_types:
            return "call", valid_action_types["call"]["amount"]
        
        return "fold", 0
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        pass


class HonestPlayer(BasePokerPlayer):
    """
    Agent Honest/Straightforward - Plays proportionally to hand strenght.
    Reasonable baseline for evaluation.
    """
    
    def _estimate_hand_strength(self, hole_card: List[str], round_state: Dict) -> float:
        """Estimates hand strength."""
        cards = gen_cards(hole_card)
        community = round_state.get("community_card", [])
        
        if community:
            community_cards = gen_cards(community)
            try:
                return estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=cards,
                    community_card=community_cards
                )
            except:
                return 0.5
        else:
            card_values = sorted([card.rank for card in cards], reverse=True)
            same_suit = cards[0].suit == cards[1].suit
            pair = card_values[0] == card_values[1]
            
            strength = 0.3 + max(card_values) * 0.02
            if pair:
                strength += 0.2
            if same_suit:
                strength += 0.05
            return min(0.9, max(0.2, strength))
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # Action proportional to strenght
        action_roll = random.random()
        
        if action_roll < hand_strength * 0.5:
            # Raise with good hands
            if "raise" in valid_action_types and hand_strength > 0.5:
                raise_info = valid_action_types["raise"]["amount"]
                if isinstance(raise_info, dict):
                    min_raise = max(0, raise_info.get("min", 0))
                    return "raise", min_raise
                return "raise", max(0, raise_info) if raise_info else 0
        
        if action_roll < hand_strength:
            # Call with medium to good hands
            if "call" in valid_action_types:
                return "call", valid_action_types["call"]["amount"]
        
        # Fold with weak hands
        if "call" in valid_action_types and valid_action_types["call"]["amount"] == 0:
            return "call", 0
        return "fold", 0
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        pass


class RaisedPlayer(BasePokerPlayer):
    """
    External Benchmark: Simple aggressive player that always raises.
    From PyPokerEngine examples - useful sanity check.
    Should be beatable by any decent RL agent.
    """
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str],
                       round_state: Dict) -> Tuple[str, int]:
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # Always raise if possible
        if "raise" in valid_action_types:
            raise_info = valid_action_types["raise"]["amount"]
            if isinstance(raise_info, dict):
                return "raise", raise_info.get("min", 0)
            return "raise", raise_info if raise_info else 0
        
        # Otherwise call
        if "call" in valid_action_types:
            return "call", valid_action_types["call"]["amount"]
        
        return "fold", 0
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        pass
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict]) -> None:
        pass
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict],
                                     round_state: Dict) -> None:
        pass


# ============================================================================
# AGENT Q-LEARNING
# ============================================================================

class QLearningPlayer(BasePokerPlayer):
    """
    Poker agent using Q-Learning with a Q table
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or Q_LEARNING_CONFIG
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS)
        )
        self.epsilon = self.config["epsilon"]
        
        # For training
        self.training = True
        self.last_state = None
        self.last_action = None
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def set_training(self, training: bool):
        """Enables or disables training mode."""
        self.training = training
        
    def get_action_index(self, action_type: str) -> int:
        """Converts action type to index."""
        if action_type == "raise":
            return 2 # Map generic raise (pypokerengine) to raise_min (index 2) for Q-Learning
        return ACTIONS.index(action_type) if action_type in ACTIONS else 1
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        state = extract_state_features(hole_card, round_state)
        
        # Actions valides
        valid_action_types = [a["action"] for a in valid_actions]
        valid_indices = [self.get_action_index(a) for a in valid_action_types]
        
        # Politique epsilon-greedy
        if self.training and random.random() < self.epsilon:
            action_idx = random.choice(valid_indices)
        else:
            # Choose best action among valid ones
            # Handle case where state is not in Q table
            if state in self.q_table:
                q_values = self.q_table[state]
                valid_q = [(i, q_values[i]) for i in valid_indices]
                action_idx = max(valid_q, key=lambda x: x[1])[0]
            else:
                # Unknown state: random action among valid ones
                action_idx = random.choice(valid_indices)
        
        action_type = ACTIONS[action_idx]
        
        # Mapping extended actions to pypokerengine actions
        pypoker_action = "fold"
        amount = 0
        
        if action_type == "fold":
            pypoker_action = "fold"
        elif action_type == "call":
            pypoker_action = "call"
            for a in valid_actions:
                if a["action"] == "call":
                    amount = a["amount"]
                    break
        else: # any raise
            pypoker_action = "raise"
            # Fallback to min raise for Q-Learning legacy interaction
            # Get min raise
            found_raise = False
            for a in valid_actions:
                if a["action"] == "raise":
                    amt = a["amount"]
                    if isinstance(amt, dict):
                         amount = amt.get("min", 0)
                    else:
                         amount = amt
                    found_raise = True
                    break
            if not found_raise:
                pypoker_action = "call" # Should not happen if logic is correct
                for a in valid_actions:
                    if a["action"] == "call":
                        amount = a["amount"]
                        break

        
        # Si l'action n'est pas valide, fall back on call
        if action_type not in valid_action_types:
            action_type = "call"
            for a in valid_actions:
                if a["action"] == "call":
                    amount = a["amount"] if a["amount"] else 0
                    break
            action_idx = self.get_action_index(action_type)
        
        self.last_state = state
        self.last_action = action_idx
        
        return action_type, amount
    
    def update_q_value(self, reward: float, next_state: Optional[Tuple] = None, 
                       done: bool = False):
        """Updates Q value."""
        if self.last_state is None or not self.training:
            return
            
        state = self.last_state
        action = self.last_action
        
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.config["gamma"] * np.max(self.q_table[next_state])
        
        # Q update
        self.q_table[state][action] += self.config["alpha"] * (
            target - self.q_table[state][action]
        )
        
        self.current_episode_reward += reward
    
    def decay_epsilon(self):
        """Decays epsilon."""
        self.epsilon = max(
            self.config["epsilon_min"],
            self.epsilon * self.config["epsilon_decay"]
        )
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        self.current_episode_reward = 0
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        self.last_state = None
        self.last_action = None
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        # Small intermediate reward for staying in the game
        if self.training and self.last_state is not None:
            self.update_q_value(0.01)
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        # Calculate final reward
        my_uuid = self.uuid
        reward = 0
        
        for winner in winners:
            if winner.get("uuid") == my_uuid:
                reward = winner.get("stack", 0) / 100.0  # Normaliser
                break
        
        # Penalty if we didnt win
        if reward == 0:
            reward = -0.5
        
        self.update_q_value(reward, done=True)
        self.episode_rewards.append(self.current_episode_reward)
        self.decay_epsilon()
    
    def save(self, filepath: str):
        """Saves the agent."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "config": self.config,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Loads the agent."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS), data["q_table"])
        self.epsilon = data["epsilon"]
        self.config = data["config"]


# ============================================================================
# RESEAU DE NEURONES PPO
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy) — outputs raw logits, NO Softmax
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value

    def get_action(self, state: np.ndarray, valid_mask: Optional[np.ndarray] = None,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Selects action with validity mask. Uses argmax if deterministic."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_tensor)

        # Mask invalid actions by setting logits to -inf
        if valid_mask is not None:
            mask = torch.BoolTensor(valid_mask == 0).unsqueeze(0)
            logits = logits.masked_fill(mask, float('-inf'))

        if deterministic:
            action = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob, value


class DQNNetwork(nn.Module):
    """
    Deep Q-Network (Value-based)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output Q-values for all actions
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNMemory:
    """Replay buffer for DQN."""
    buffer: list = field(default_factory=list)
    capacity: int = 10000
    
    def push(self, transition: Tuple):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


@dataclass
class PPOMemory:
    """Memory buffer for PPO."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    valid_masks: List[np.ndarray] = field(default_factory=list)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_masks.clear()
    
    def __len__(self):
        return len(self.states)


class PPOPlayer(BasePokerPlayer):
    """
    Poker agent using PPO 
    """
    
    STATE_DIM = 28  # Fixed: Match actual element count (28)
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or PPO_CONFIG
        
        # Actor-Critic network
        self.network = ActorCritic(self.STATE_DIM, NUM_ACTIONS)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.config["lr"]
        )
        
        # Memory
        self.memory = PPOMemory()
        
        # For training
        self.training = True
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.last_valid_mask = None
        
        # Running state normalization
        self.state_mean = np.zeros(self.STATE_DIM, dtype=np.float32)
        self.state_m2 = np.ones(self.STATE_DIM, dtype=np.float32)
        self.state_count = 0

        # Statistiques
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.losses = []
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Welford's online normalization."""
        if self.training:
            self.state_count += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.state_count
            delta2 = state - self.state_mean
            self.state_m2 += delta * delta2
        std = np.sqrt(self.state_m2 / max(self.state_count, 1)) + 1e-8
        return (state - self.state_mean) / std

    def set_training(self, training: bool):
        """Enables or disables training mode."""
        self.training = training
        self.network.train(training)
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str],
                       round_state: Dict) -> Tuple[str, int]:
        state = extract_state_vector(hole_card, round_state, valid_actions, my_uuid=self.uuid)
        state = self.normalize_state(state)

        # Create validity mask
        # Map pypokerengine actions to our extended actions
        valid_action_types = {a["action"] for a in valid_actions}
        valid_mask = np.zeros(NUM_ACTIONS)
        
        for i, action_name in enumerate(ACTIONS):
            if action_name == "fold":
                if "fold" in valid_action_types:
                    valid_mask[i] = 1.0
            elif action_name == "call":
                if "call" in valid_action_types:
                    valid_mask[i] = 1.0
            elif "raise" in action_name or action_name == "all_in":
                # Enable all raise variants if 'raise' is valid
                if "raise" in valid_action_types:
                    valid_mask[i] = 1.0
        
        # Action selection (argmax in inference, sample in training)
        with torch.no_grad() if not self.training else torch.enable_grad():
            action_idx, log_prob, value = self.network.get_action(
                state, valid_mask, deterministic=not self.training
            )
        
        action_name = ACTIONS[action_idx]
        
        # Find corresponding amount & pypokerengine action
        pypoker_action = "fold"
        amount = 0
        
        if action_name == "fold":
            pypoker_action = "fold"
        elif action_name == "call":
            pypoker_action = "call"
            for a in valid_actions:
                if a["action"] == "call":
                    amount = a["amount"]
                    break
        elif "raise" in action_name or action_name == "all_in":
            pypoker_action = "raise"
            
            # Get raise limits
            raise_limits = {}
            for a in valid_actions:
                if a["action"] == "raise":
                    raise_limits = a["amount"]
                    break
            
            if not raise_limits:
                 # Fallback if raise not valid (e.g. capped)
                 pypoker_action = "call"
                 for a in valid_actions:
                     if a["action"] == "call":
                         amount = a["amount"]
                         break
            else:
                 min_raise = raise_limits.get("min", 0)
                 max_raise = raise_limits.get("max", 0)
                 pot_amount = round_state.get("pot", {}).get("main", {}).get("amount", 0)
                 
                 if action_name == "raise_min":
                     amount = min_raise
                 elif action_name == "raise_half_pot":
                     target = pot_amount * 0.5
                     amount = int(max(min_raise, min(target, max_raise)))
                 elif action_name == "raise_pot":
                     target = pot_amount
                     amount = int(max(min_raise, min(target, max_raise)))
                 elif action_name == "all_in":
                     amount = max_raise
                 else:
                     amount = min_raise
        
        # Save for update (CRITICAL: Must be done before returning)
        self.last_state = state
        self.last_action = action_idx
        self.last_log_prob = log_prob
        self.last_value = value
        self.last_valid_mask = valid_mask
        
        return pypoker_action, amount
    
    def store_transition(self, reward: float, done: bool = False):
        """Stores a transition in memory."""
        if self.last_state is not None and self.training:
            self.memory.states.append(self.last_state)
            self.memory.actions.append(self.last_action)
            self.memory.rewards.append(reward)
            self.memory.log_probs.append(self.last_log_prob.detach())
            self.memory.values.append(self.last_value.detach())
            self.memory.dones.append(done)
            self.memory.valid_masks.append(self.last_valid_mask)
            
            self.current_episode_reward += reward
    
    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], 
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes GAE advantages."""
        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]
        
        advantages = []
        returns = []
        gae = 0
        
        values_np = [v.item() for v in values] + [0]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values_np[t]
                gae = delta
            else:
                delta = rewards[t] + gamma * values_np[t + 1] - values_np[t]
                gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values_np[t])
        
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def update(self):
        """Updates network with PPO."""
        if len(self.memory) < 32:
            return
        
        # Prepare data
        states = torch.FloatTensor(np.array(self.memory.states))
        actions = torch.LongTensor(self.memory.actions)
        old_log_probs = torch.stack(self.memory.log_probs)
        valid_masks = torch.FloatTensor(np.array(self.memory.valid_masks))
        
        advantages, returns = self.compute_gae(
            self.memory.rewards, 
            self.memory.values, 
            self.memory.dones
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update on multiple epochs
        for _ in range(self.config["epochs"]):
            # Random mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.config["batch_size"]):
                end = start + self.config["batch_size"]
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = valid_masks[batch_indices]
                
                # Forward pass
                logits, values = self.network(batch_states)

                # Mask invalid actions with -inf
                invalid_mask = (batch_masks == 0)
                logits = logits.masked_fill(invalid_mask, float('-inf'))

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config["clip_epsilon"], 
                    1 + self.config["clip_epsilon"]
                ) * batch_advantages
                
                # Loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.view(-1), batch_returns)
                
                loss = (actor_loss + 
                        self.config["value_coef"] * critic_loss - 
                        self.config["entropy_coef"] * entropy)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                self.losses.append(loss.item())
        
        self.memory.clear()
    
    def receive_game_start_message(self, game_info: Dict) -> None:
        self.current_episode_reward = 0
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], 
                                    seats: List[Dict]) -> None:
        self.last_state = None
        self.last_action = None
        
        # Track initial stack for this round
        my_uuid = self.uuid
        for seat in seats:
            if seat["uuid"] == my_uuid:
                self.initial_stack = seat["stack"]
                break
        else:
            self.initial_stack = 1000  # Fallback

    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass  # Only store transitions at round end (receive_round_result_message)
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        my_uuid = self.uuid
        final_stack = 0
        
        # Determine final stack from round_state seats
        seats = round_state.get("seats", [])
        for seat in seats:
            if seat["uuid"] == my_uuid:
                final_stack = seat["stack"]
                break
        
        # Calculate Net Profit
        net_profit = final_stack - self.initial_stack
        
        # Normalize reward (Buy-in 1000)
        # Scale to [-1, 1] roughly. 
        # Winning 1000 -> +1.0. Losing 1000 -> -1.0.
        reward = net_profit / 1000.0
        
        # Clip for stability
        reward = max(-1.0, min(1.0, reward))
        
        self.store_transition(reward, done=True)
        self.episode_rewards.append(self.current_episode_reward)
    
    def save(self, filepath: str):
        """Saves the agent."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "state_mean": self.state_mean,
            "state_m2": self.state_m2,
            "state_count": self.state_count,
        }, filepath)

    def load(self, filepath: str):
        """Loads the agent."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = checkpoint["config"]
        if "state_mean" in checkpoint:
            self.state_mean = checkpoint["state_mean"]
            self.state_m2 = checkpoint["state_m2"]
            self.state_count = checkpoint["state_count"]
    
    def clone(self) -> "PPOPlayer":
        """Creates a copy of the agent for self-play."""
        clone = PPOPlayer(self.config.copy())
        clone.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        clone.state_mean = self.state_mean.copy()
        clone.state_m2 = self.state_m2.copy()
        clone.state_count = self.state_count
        clone.set_training(False)
        return clone


# ============================================================================
# TRAINING
# ============================================================================

def run_games(player1: BasePokerPlayer, player2: BasePokerPlayer, 
              num_games: int = 500, initial_stack: int = 1000,
              small_blind: int = 10, verbose: bool = False) -> Dict[str, int]:
    """
    Runs a series of games between 2 players
    """
    results = {"player1_wins": 0, "player2_wins": 0, "ties": 0}
    
    for _ in range(num_games):
        config = setup_config(
            max_round=20,
            initial_stack=initial_stack,
            small_blind_amount=small_blind
        )
        config.register_player(name="player1", algorithm=player1)
        config.register_player(name="player2", algorithm=player2)
        
        game_result = start_poker(config, verbose=0)
        
        stacks = {p["name"]: p["stack"] for p in game_result["players"]}
        
        if stacks["player1"] > stacks["player2"]:
            results["player1_wins"] += 1
        elif stacks["player2"] > stacks["player1"]:
            results["player2_wins"] += 1
        else:
            results["ties"] += 1
    
    return results


def evaluate_agent(agent: BasePokerPlayer, opponent: BasePokerPlayer, num_games: int = 100) -> float:
    """
    Evaluates an agent against an opponent and returns win rate.
    """
    results = run_games(agent, opponent, num_games=num_games)
    return results["player1_wins"] / num_games


def evaluate_head_to_head(agent1: BasePokerPlayer, agent2: BasePokerPlayer, num_games: int = 500) -> float:
    """
    Evaluates agent1 vs agent2 head-to-head and returns win rate for agent1.
    """
    results = run_games(agent1, agent2, num_games=num_games)
    return results["player1_wins"] / num_games


def train_qlearning_vs_random(num_episodes: int = 5000, 
                               eval_interval: int = 500) -> QLearningPlayer:
    """
    Trains a Q-Learning agent against random.
    """
    print("\n" + "="*60)
    print(" TRAINING Q-LEARNING vs RANDOM")
    print("="*60)
    
    q_agent = QLearningPlayer()
    random_agent = RandomPlayer()
    
    win_rates = []
    eval_episodes = []
    
    for episode in tqdm(range(num_episodes), desc="Q-Learning Training"):
        config = setup_config(
            max_round=10,
            initial_stack=1000,
            small_blind_amount=10
        )
        config.register_player(name="q_learner", algorithm=q_agent)
        config.register_player(name="random", algorithm=random_agent)
        
        start_poker(config, verbose=0)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            q_agent.set_training(False)
            results = run_games(q_agent, RandomPlayer(), num_games=100)
            win_rate = results["player1_wins"] / 100
            win_rates.append(win_rate)
            eval_episodes.append(episode + 1)
            
            print(f"\n Episode {episode + 1}: Win rate = {win_rate:.2%}, "
                  f"Epsilon = {q_agent.epsilon:.3f}, "
                  f"Q-table size = {len(q_agent.q_table)}")
            
            q_agent.set_training(True)
    
    # Graph
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, win_rates, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random baseline (50%)')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Win rate', fontsize=12)
    plt.title(' Q-Learning: Evolution of win rate vs Random', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('qlearning_vs_random.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Graph saved: qlearning_vs_random.png")
    
    return q_agent



# ============================================================================
# AGENT DQN
# ============================================================================

class DQNPlayer(BasePokerPlayer):
    """
    Poker agent using Deep Q-Network (DQN)
    """
    
    STATE_DIM = 28
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or DQN_CONFIG
        
        # Networks (Policy + Target)
        self.policy_net = DQNNetwork(self.STATE_DIM, NUM_ACTIONS)
        self.target_net = DQNNetwork(self.STATE_DIM, NUM_ACTIONS)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["lr"])
        self.memory = DQNMemory(capacity=self.config["buffer_size"])
        
        self.epsilon = self.config["epsilon"]
        self.training = True
        
        # Episode tracking
        self.last_state = None
        self.last_action = None
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.losses = []
        
    def set_training(self, training: bool):
        self.training = training
        self.policy_net.train(training)
        
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str],
                       round_state: Dict) -> Tuple[str, int]:
        state = extract_state_vector(hole_card, round_state, valid_actions, my_uuid=self.uuid)

        # Valid mask
        valid_action_types = {a["action"] for a in valid_actions}
        valid_mask = np.zeros(NUM_ACTIONS)
        for i, action_name in enumerate(ACTIONS):
            if action_name == "fold":
                if "fold" in valid_action_types: valid_mask[i] = 1.0
            elif action_name == "call":
                if "call" in valid_action_types: valid_mask[i] = 1.0
            elif "raise" in action_name or action_name == "all_in":
                if "raise" in valid_action_types: valid_mask[i] = 1.0
            
        # Select action
        if self.training and random.random() < self.epsilon:
            # Random valid action
            valid_indices = [i for i, v in enumerate(valid_mask) if v == 1.0]
            action_idx = random.choice(valid_indices)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                # Mask invalid actions with -infinity
                q_values[0, valid_mask == 0] = -float('inf')
                action_idx = q_values.argmax().item()
        
        action_name = ACTIONS[action_idx]
        
        # Calculate amount (Same logic as PPO)
        pypoker_action = "fold"
        amount = 0
        
        if action_name == "fold":
            pypoker_action = "fold"
        elif action_name == "call":
            pypoker_action = "call"
            for a in valid_actions:
                 if a["action"] == "call": amount = a["amount"]; break
        elif "raise" in action_name or action_name == "all_in":
            pypoker_action = "raise"
            raise_limits = {}
            for a in valid_actions:
                if a["action"] == "raise": raise_limits = a["amount"]; break
            
            if not raise_limits:
                 # Fallback
                 pypoker_action = "call"
                 for a in valid_actions:
                     if a["action"] == "call": amount = a["amount"]; break
            else:
                 min_raise = raise_limits.get("min", 0)
                 max_raise = raise_limits.get("max", 0)
                 pot_amount = round_state.get("pot", {}).get("main", {}).get("amount", 0)
                 
                 if action_name == "raise_min": amount = min_raise
                 elif action_name == "raise_half_pot": amount = int(max(min_raise, min(pot_amount * 0.5, max_raise)))
                 elif action_name == "raise_pot": amount = int(max(min_raise, min(pot_amount, max_raise)))
                 elif action_name == "all_in": amount = max_raise
                 else: amount = min_raise

        # Store PREVIOUS transition if exists (S, A, 0, S', Done=False)
        if self.last_state is not None and self.training:
             self.memory.push((self.last_state, self.last_action, 0.0, state, False))
             self.update()

        # Store current state/action for next step
        self.last_state = state
        self.last_action = action_idx
        
        return pypoker_action, amount

    def receive_game_start_message(self, game_info: Dict) -> None:
        self.current_episode_reward = 0
    
    def receive_round_start_message(self, round_count: int, hole_card: List[str], seats: List[Dict]) -> None:
        self.last_state = None
        self.last_action = None
        my_uuid = self.uuid
        for seat in seats:
            if seat["uuid"] == my_uuid:
                self.initial_stack = seat["stack"]
                break
        else:
            self.initial_stack = 1000

    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        pass
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], round_state: Dict) -> None:
        my_uuid = self.uuid
        final_stack = 0
        seats = round_state.get("seats", [])
        for seat in seats:
            if seat["uuid"] == my_uuid:
                final_stack = seat["stack"]
                break
        
        # Reward Normalization
        reward = (final_stack - self.initial_stack) / 1000.0
        reward = max(-1.0, min(1.0, reward))
        
        # Store terminal transition (S, A, R, None, Done=True)
        if self.last_state is not None:
            self.memory.push((self.last_state, self.last_action, reward, None, True))
        
        self.episode_rewards.append(reward)
        self.decay_epsilon()
        self.update() # Learn at end of round
    
    def decay_epsilon(self):
        self.epsilon = max(self.config["epsilon_min"], self.epsilon * self.config["epsilon_decay"])
        
    def update(self):
        if len(self.memory) < self.config["batch_size"]: return
        
        transitions = self.memory.sample(self.config["batch_size"])
        # Transpose batch
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        
        # Handle None next_states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)
        if any(non_final_mask):
             non_final_next_states = torch.FloatTensor(np.array([s for s in batch[3] if s is not None]))
        else:
             non_final_next_states = torch.empty(0, self.STATE_DIM)
        
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)
        
        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # V(s') = max Q(s', a)
        next_q_values = torch.zeros(self.config["batch_size"], 1)
        with torch.no_grad():
             if len(non_final_next_states) > 0:
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
        
        # Target = R + gamma * V(s')
        target_q_values = rewards + (self.config["gamma"] * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())


def train_ppo_curriculum(num_episodes: int = 5000, 
                         eval_interval: int = 500,
                         update_interval: int = 100) -> PPOPlayer:
    """
    Trains a PPO agent against a curriculum of opponents.
    """
    print("\n" + "="*60)
    print(" TRAINING PPO (CURRICULUM)")
    print("="*60)
    
    ppo_agent = PPOPlayer()
    
    # Pool of opponents
    opponents_pool = [
        RandomPlayer(),
        HonestPlayer(),
        CallingStationPlayer(call_threshold=0.3),
        TightAggressivePlayer(tightness=0.6, aggression=0.7),
        LooseAggressivePlayer(looseness=0.3, aggression=0.8)
    ]
    
    win_rates = []
    eval_episodes = []
    
    for episode in tqdm(range(num_episodes), desc="PPO Curriculum Training"):
        # Select random opponent
        opponent = random.choice(opponents_pool)
        opponent_name = opponent.__class__.__name__
        
        config = setup_config(
            max_round=10,
            initial_stack=1000,
            small_blind_amount=10
        )
        config.register_player(name="ppo", algorithm=ppo_agent)
        config.register_player(name="opponent", algorithm=opponent)
        
        start_poker(config, verbose=0)
        
        # PPO periodic update
        if (episode + 1) % update_interval == 0:
            ppo_agent.update()
        
        # Periodic evaluation (always against TAG for consistency)
        if (episode + 1) % eval_interval == 0:
            ppo_agent.set_training(False)
            # Evaluate against a strong heuristic (TAG)
            results = run_games(ppo_agent, TightAggressivePlayer(), num_games=100)
            win_rate = results["player1_wins"] / 100
            win_rates.append(win_rate)
            eval_episodes.append(episode + 1)
            
            avg_loss = np.mean(ppo_agent.losses[-100:]) if ppo_agent.losses else 0
            print(f"\n Episode {episode + 1}: Win rate vs TAG = {win_rate:.2%}, "
                  f"Avg Loss = {avg_loss:.4f}")
            
            ppo_agent.set_training(True)
    
    # Graph based on TAG performance
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, win_rates, 'g-o', linewidth=2, markersize=6)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline 50%')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Win rate vs TAG', fontsize=12)
    plt.title(' PPO Curriculum: Evolution of win rate vs TAG', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ppo_curriculum.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Graph saved: ppo_curriculum.png")
    
    return ppo_agent


def train_self_play(agent: PPOPlayer, num_generations: int = 10,
                    games_per_gen: int = 1000, eval_games: int = 100) -> PPOPlayer:
    """
    Trains PPO agent via self-play against prevous versions.
    """
    print("\n" + "="*60)
    print(" TRAINING SELF-PLAY")
    print("="*60)
    
    win_rates_vs_random = []
    win_rates_vs_previous = []
    generations = []
    
    # Pool of opponents (Fictitious Self-Play)
    # Start with a heuristic to ground the learning
    opponents_pool = [RandomPlayer(), TightAggressivePlayer()]
    
    current_agent = agent
    # Add initial copy to pool
    opponents_pool.append(current_agent.clone())
    
    for gen in range(num_generations):
        print(f"\n Generation {gen + 1}/{num_generations}")
        
        current_agent.set_training(True)
        
        # Training against pool of predecessors
        for episode in tqdm(range(games_per_gen), desc=f"Gen {gen + 1}"):
            opponent = random.choice(opponents_pool)
            
            config = setup_config(
                max_round=10,
                initial_stack=1000,
                small_blind_amount=10
            )
            config.register_player(name="current", algorithm=current_agent)
            config.register_player(name="opponent", algorithm=opponent)
            
            start_poker(config, verbose=0)
            
            # Periodic update (Collect more data before update)
            if (episode + 1) % 100 == 0: # Increased from 50
                current_agent.update()
        
        # Evaluation vs Random
        current_agent.set_training(False)
        results_random = run_games(current_agent, RandomPlayer(), num_games=eval_games)
        win_rate_random = results_random["player1_wins"] / eval_games
        win_rates_vs_random.append(win_rate_random)
        
        # Evaluation vs latest version in pool (excluding heuristics if pos)
        last_opponent = opponents_pool[-1]
        results_prev = run_games(current_agent, last_opponent, num_games=eval_games)
        win_rate_prev = results_prev["player1_wins"] / eval_games
        win_rates_vs_previous.append(win_rate_prev)
        
        generations.append(gen + 1)
        
        print(f"    vs Random: {win_rate_random:.2%}")
        print(f"    vs Pool[-1]: {win_rate_prev:.2%}")
        
        # Always add new version to pool (FSP)
        # This creates a diverse population of ancestors
        print("    Current version added to opponent pool.")
        opponents_pool.append(current_agent.clone())
    
    # Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(generations, win_rates_vs_random, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Baseline 50%')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Win rate', fontsize=12)
    ax1.set_title(' Self-Play: Performance vs Random', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(generations, win_rates_vs_previous, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Baseline 50%')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Win rate', fontsize=12)
    ax2.set_title(' Self-Play: Performance vs Previous Version', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('self_play_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Graph saved: self_play_evolution.png")
    
    return current_agent


def comprehensive_evaluation(q_agent: QLearningPlayer, ppo_agent: PPOPlayer, dqn_agent: DQNPlayer, num_games: int = 500):
    """
    Evaluates Q-Learning vs PPO vs DQN against ALL benchmarks.
    """
    print("\n" + "="*70)
    print(" COMPLETE AGENT EVALUATION (Q-Learning vs PPO vs DQN)")
    print("="*70)
    
    test_opponents = {
        "Random": RandomPlayer(),
        "TAG": TightAggressivePlayer(),
        "LAG": LooseAggressivePlayer(),
        "CallingStation": CallingStationPlayer(),
        "Honest": HonestPlayer(),
        "RaisedPlayer": RaisedPlayer()  # External benchmark
    }
    
    agents = {"Q-Learning": q_agent, "PPO": ppo_agent, "DQN": dqn_agent}
    results = {name: {} for name in agents}
    
    for agent_name, agent_obj in agents.items():
        print(f"\n {agent_name}:")
        if hasattr(agent_obj, "set_training"):
            agent_obj.set_training(False)
            
        for opp_name, opp_obj in test_opponents.items():
            win_rate = evaluate_agent(agent_obj, opp_obj, num_games=num_games)
            results[agent_name][opp_name] = win_rate
            print(f"   vs {opp_name:<20}: {win_rate*100:5.1f}% ({int(win_rate*num_games)}/{num_games})")

    # Direct Matches
    print("\n DIRECT MATCHES:")
    ql_ppo = evaluate_head_to_head(q_agent, ppo_agent, num_games)
    print(f"   Q-Learning vs PPO: {ql_ppo*100:.1f}% for Q-Learning")
    
    dqn_ppo = evaluate_head_to_head(dqn_agent, ppo_agent, num_games)
    print(f"   DQN vs PPO:        {dqn_ppo*100:.1f}% for DQN")
    
    dqn_ql = evaluate_head_to_head(dqn_agent, q_agent, num_games)
    print(f"   DQN vs Q-Learning: {dqn_ql*100:.1f}% for DQN")
    
    # Plotting
    labels = list(test_opponents.keys())
    x = np.arange(len(labels))
    width = 0.25 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, [results["Q-Learning"][l] for l in labels], width, label='Q-Learning', color='gray')
    rects2 = ax.bar(x, [results["PPO"][l] for l in labels], width, label='PPO', color='blue')
    rects3 = ax.bar(x + width, [results["DQN"][l] for l in labels], width, label='DQN', color='green')
    
    ax.set_ylabel('Win Rate')
    ax.set_title('Agent Performance Comparison (Q-Learning vs PPO vs DQN)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.05)  # Slightly higher to fit labels
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig("comprehensive_evaluation.png")
    print("\n Graph saved: comprehensive_evaluation.png")

    
    return results



def compare_agents(q_agent: QLearningPlayer, ppo_agent: PPOPlayer, 
                   num_games: int = 1000):
    """
    Compares Q-Learning and PPO agent performances (simplified).
    """
    return comprehensive_evaluation(q_agent, ppo_agent, num_games=num_games)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    POKER REINFORCEMENT LEARNING AGENT                         ║
    ║                                                               ║
    ║   • Q-Learning (Table-based)                                  ║
    ║   • PPO (Proximal Policy Optimization)                        ║
    ║   • Self-Play Training                                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    Q_EPISODES = 3000      # Episodes for Q-Learning
    PPO_EPISODES = 5000    # Episodes for PPO
    SELF_PLAY_GENS = 5     # Self-play generations
    GAMES_PER_GEN = 2000    # Games per generation
    
    # -------------------------------------------------------------------------
    # Step 1: Load Best Q-Learning Agent
    # -------------------------------------------------------------------------
    q_agent_path = "q_learning_agent_vboost.pkl" # 419 states
    q_agent = QLearningPlayer()
    
    if os.path.exists(q_agent_path):
        print(f"\n Loading existing Q-Learning agent: {q_agent_path}")
        q_agent.load(q_agent_path)
    else:
        print(f"\n {q_agent_path} not found. Training from scratch...")
        q_agent = train_qlearning_vs_random(num_episodes=Q_EPISODES, eval_interval=500)
        q_agent.save("q_learning_agent.pkl")
        print(f" Agent Q-Learning saved: q_learning_agent.pkl")
    
    # -------------------------------------------------------------------------
    # Step 2: Train PPO Curriculum
    # -------------------------------------------------------------------------
    ppo_agent = train_ppo_curriculum(num_episodes=PPO_EPISODES, eval_interval=200)
    ppo_agent.save("ppo_agent_v1.pt")
    print(f" Agent PPO saved: ppo_agent_v1.pt")
    
    # -------------------------------------------------------------------------
    # Step 3: Self-Play to improve PPO
    # -------------------------------------------------------------------------
    ppo_agent_selfplay = train_self_play(
        ppo_agent, 
        num_generations=SELF_PLAY_GENS,
        games_per_gen=GAMES_PER_GEN
    )
    ppo_agent_selfplay.save("ppo_agent_selfplay.pt")
    print(f" Agent PPO Self-Play saved: ppo_agent_selfplay.pt")
    
    # -------------------------------------------------------------------------
    # Step 4: Train DQN (New Benchmark)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(" TRAINING DQN (Deep Q-Network)")
    print("="*60)
    
    dqn_agent = DQNPlayer()
    
    # DQN uses replay buffer, so we use a simpler loop than PPO curriculum logic for now,
    # or reuse similar curriculum. Let's do a simplified curriculum loop for 3000 episodes.
    num_episodes_dqn = 3000
    pbar = tqdm(range(num_episodes_dqn), desc="DQN Training")
    
    for episode in pbar:
        # Curriculum
        if episode < 500: opponent = RandomPlayer()
        elif episode < 1500: opponent = CallingStationPlayer()
        else: opponent = TightAggressivePlayer()
        
        config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="dqn", algorithm=dqn_agent)
        config.register_player(name="opponent", algorithm=opponent)
        
        dqn_agent.set_training(True)
        start_poker(config, verbose=0)
        
        if (episode+1) % 500 == 0:
             dqn_agent.set_training(False)
             wr = evaluate_agent(dqn_agent, TightAggressivePlayer(), 100)
             pbar.set_description(f"DQN Training (WR vs TAG: {wr*100:.1f}%)")
             
    torch.save(dqn_agent.policy_net.state_dict(), "dqn_agent.pt")
    print(f" Agent DQN saved: dqn_agent.pt")

    # -------------------------------------------------------------------------
    # Step 5: Final Comparison
    # -------------------------------------------------------------------------
    comprehensive_evaluation(q_agent, ppo_agent_selfplay, dqn_agent, num_games=500)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED!")
    print("="*60)
    print("""
    Generated files:
       • q_learning_agent_vboost.pkl - Trained Q-Learning agent
       • ppo_agent_v1.pt             - Agent PPO (curriculum)
       • ppo_agent_selfplay.pt       - Agent PPO (after Self-Play)
       • dqn_agent.pt                - Agent DQN (Deep Q-Network)
       • qlearning_vs_random.png     - Evolution graph Q-Learning
       • ppo_curriculum.png          - Evolution graph PPO
       • self_play_evolution.png     - Graph Self-Play
       • comprehensive_evaluation.png - Final comparison (3 agents)
    """)


def evaluate_only():
    """
    Loads saved agents and runs complete evaluation.
    No re-training !
    """
    print("\n" + "="*70)
    print(" LOADING SAVED AGENTS")
    print("="*70)
    
    # Charger Q-Learning
    q_agent = QLearningPlayer()
    try:
        with open("q_learning_agent_vboost.pkl", "rb") as f:
            q_agent.q_table = pickle.load(f)
        print(" Agent Q-Learning loaded (q_learning_agent_vboost.pkl)")
    except FileNotFoundError:
        print(" q_learning_agent_vboost.pkl not found. Train first with main()")
        return
    
    # Charger PPO
    ppo_agent = PPOPlayer()
    try:
        ppo_agent.load("ppo_agent_selfplay.pt")
        print(" Agent PPO loaded (ppo_agent_selfplay.pt)")
    except FileNotFoundError:
        print(" ppo_agent_selfplay.pt not found. Train first with main()")
        return
    
    # Charger DQN
    dqn_agent = DQNPlayer()
    try:
        dqn_agent.policy_net.load_state_dict(torch.load("dqn_agent.pt"))
        dqn_agent.target_net.load_state_dict(torch.load("dqn_agent.pt"))
        print(" Agent DQN loaded (dqn_agent.pt)")
    except FileNotFoundError:
        print(" dqn_agent.pt not found. Train first with main()")
        return
    
    # Run complete evaluation
    comprehensive_evaluation(q_agent, ppo_agent, dqn_agent, num_games=500)
    
    print("\n Evaluation completed !")
    print(" Graph saved: comprehensive_evaluation.png")



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate_only()
    else:
        main()