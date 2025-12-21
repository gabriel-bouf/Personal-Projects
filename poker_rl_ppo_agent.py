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

ACTIONS = ["fold", "call", "raise"]
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
    "lr": 3e-4,          # Learning rate
    "gamma": 0.99,       # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_epsilon": 0.2, # PPO clipping
    "entropy_coef": 0.01, # Entropy coefficient
    "value_coef": 0.5,   # Value coefficient
    "epochs": 4,         # Epochs per update
    "batch_size": 64,    # Batch size
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

def extract_state_vector(hole_card: List[str], round_state: Dict[str, Any]) -> np.ndarray:
    """
    Extracts feature vector for the PPO neural network.
    """
    features = []
    
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
    features.append(min(pot / 1000.0, 1.0))
    
    # Normalized player stack
    seats = round_state.get("seats", [])
    my_stack = 1000
    for seat in seats:
        if seat.get("uuid") == round_state.get("next_player"):
            my_stack = seat.get("stack", 1000)
            break
    features.append(min(my_stack / 1000.0, 1.0))
    
    # Number of active players
    active_players = sum(1 for s in seats if s.get("state") == "participating")
    features.append(active_players / 6.0)
    
    # Estimation of win rate
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
        # Preflop estimation
        card_values = [card.rank for card in cards]
        pair = card_values[0] == card_values[1]
        high_card = max(card_values)
        win_rate = 0.5 + (high_card / 28.0) + (0.2 if pair else 0)
        win_rate = min(0.95, max(0.2, win_rate))
    features.append(win_rate)
    
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
    """
    
    def __init__(self, tightness: float = 0.6, aggression: float = 0.7):
        super().__init__()
        self.tightness = tightness  # Threshold to play a hand (0-1)
        self.aggression = aggression  # Probability of raise vs call
    
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
        
        # Decision based on hand strenght
        valid_action_types = {a["action"]: a for a in valid_actions}
        
        # Main trop faible -> fold (sauf si check gratuit)
        if hand_strength < self.tightness * 0.5:
            if "call" in valid_action_types and valid_action_types["call"]["amount"] == 0:
                return "call", 0  # Check gratuit
            return "fold", 0
        
        # Medium hand -> call or fold depending on cost
        elif hand_strength < self.tightness:
            if "call" in valid_action_types:
                call_amount = valid_action_types["call"]["amount"]
                pot = round_state.get("pot", {}).get("main", {}).get("amount", 1)
                pot_odds = call_amount / (pot + call_amount + 1)
                
                if pot_odds < hand_strength * 0.8:
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
        
        # Find corresponding amount
        amount = 0
        for action in valid_actions:
            if action["action"] == action_type:
                if action_type == "raise":
                    amt = action["amount"]
                    if isinstance(amt, dict):
                        amount = max(0, amt.get("min", 0))
                    else:
                        amount = max(0, amt) if amt else 0
                else:
                    amount = action["amount"] if action["amount"] else 0
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
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value
    
    def get_action(self, state: np.ndarray, valid_mask: Optional[np.ndarray] = None
                   ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Selects action with validity mask."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state_tensor)
        
        # Apply mask if provided
        if valid_mask is not None:
            mask = torch.FloatTensor(valid_mask).unsqueeze(0)
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value


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
    
    STATE_DIM = 22  # State vector dimension
    
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
        
        # Statistiques
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.losses = []
    
    def set_training(self, training: bool):
        """Enables or disables training mode."""
        self.training = training
        self.network.train(training)
    
    def declare_action(self, valid_actions: List[Dict], hole_card: List[str], 
                       round_state: Dict) -> Tuple[str, int]:
        state = extract_state_vector(hole_card, round_state)
        
        # Create validity mask
        valid_action_types = [a["action"] for a in valid_actions]
        valid_mask = np.zeros(NUM_ACTIONS)
        for i, action_type in enumerate(ACTIONS):
            if action_type in valid_action_types:
                valid_mask[i] = 1.0
        
        # Action selection
        with torch.no_grad() if not self.training else torch.enable_grad():
            action_idx, log_prob, value = self.network.get_action(state, valid_mask)
        
        action_type = ACTIONS[action_idx]
        
        # Find amount
        amount = 0
        for action in valid_actions:
            if action["action"] == action_type:
                if action_type == "raise":
                    amt = action["amount"]
                    if isinstance(amt, dict):
                        amount = max(0, amt.get("min", 0))
                    else:
                        amount = max(0, amt) if amt else 0
                else:
                    amount = action["amount"] if action["amount"] else 0
                break
        
        # Fallback si action invalide
        if action_type not in valid_action_types:
            action_type = "call" if "call" in valid_action_types else valid_action_types[0]
            for a in valid_actions:
                if a["action"] == action_type:
                    amt = a["amount"]
                    if isinstance(amt, dict):
                        amount = max(0, amt.get("min", 0))
                    else:
                        amount = max(0, amt) if amt else 0
                    break
            action_idx = ACTIONS.index(action_type) if action_type in ACTIONS else 1
        
        # Save for update
        self.last_state = state
        self.last_action = action_idx
        self.last_log_prob = log_prob
        self.last_value = value
        self.last_valid_mask = valid_mask
        
        return action_type, amount
    
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
        if len(self.memory) < self.config["batch_size"]:
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
                action_probs, values = self.network(batch_states)
                
                # Apply masks
                action_probs = action_probs * batch_masks
                action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                dist = Categorical(action_probs)
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
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
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
    
    def receive_street_start_message(self, street: str, round_state: Dict) -> None:
        pass
    
    def receive_game_update_message(self, action: Dict, round_state: Dict) -> None:
        if self.training and self.last_state is not None:
            self.store_transition(0.01)  # Small reward for staying in game
    
    def receive_round_result_message(self, winners: List[Dict], hand_info: List[Dict], 
                                     round_state: Dict) -> None:
        my_uuid = self.uuid
        reward = 0
        
        for winner in winners:
            if winner.get("uuid") == my_uuid:
                reward = winner.get("stack", 0) / 100.0
                break
        
        if reward == 0:
            reward = -0.5
        
        self.store_transition(reward, done=True)
        self.episode_rewards.append(self.current_episode_reward)
    
    def save(self, filepath: str):
        """Saves the agent."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, filepath)
    
    def load(self, filepath: str):
        """Loads the agent."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = checkpoint["config"]
    
    def clone(self) -> "PPOPlayer":
        """Creates a copy of the agent for self-play."""
        clone = PPOPlayer(self.config.copy())
        clone.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        clone.set_training(False)
        return clone


# ============================================================================
# TRAINING
# ============================================================================

def run_games(player1: BasePokerPlayer, player2: BasePokerPlayer, 
              num_games: int = 100, initial_stack: int = 1000,
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


def train_ppo_vs_random(num_episodes: int = 3000, 
                        eval_interval: int = 300,
                        update_interval: int = 100) -> PPOPlayer:
    """
    Trains a PPO agent against random.
    """
    print("\n" + "="*60)
    print(" TRAINING PPO vs RANDOM")
    print("="*60)
    
    ppo_agent = PPOPlayer()
    random_agent = RandomPlayer()
    
    win_rates = []
    eval_episodes = []
    
    for episode in tqdm(range(num_episodes), desc="PPO Training"):
        config = setup_config(
            max_round=10,
            initial_stack=1000,
            small_blind_amount=10
        )
        config.register_player(name="ppo", algorithm=ppo_agent)
        config.register_player(name="random", algorithm=random_agent)
        
        start_poker(config, verbose=0)
        
        # PPO periodic update
        if (episode + 1) % update_interval == 0:
            ppo_agent.update()
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            ppo_agent.set_training(False)
            results = run_games(ppo_agent, RandomPlayer(), num_games=100)
            win_rate = results["player1_wins"] / 100
            win_rates.append(win_rate)
            eval_episodes.append(episode + 1)
            
            avg_loss = np.mean(ppo_agent.losses[-100:]) if ppo_agent.losses else 0
            print(f"\n Episode {episode + 1}: Win rate = {win_rate:.2%}, "
                  f"Avg Loss = {avg_loss:.4f}")
            
            ppo_agent.set_training(True)
    
    # Graph
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, win_rates, 'g-o', linewidth=2, markersize=6)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random baseline (50%)')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Win rate', fontsize=12)
    plt.title(' PPO: Evolution of win rate vs Random', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ppo_vs_random.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Graph saved: ppo_vs_random.png")
    
    return ppo_agent


def train_self_play(agent: PPOPlayer, num_generations: int = 10,
                    games_per_gen: int = 500, eval_games: int = 100) -> PPOPlayer:
    """
    Trains PPO agent via self-play against prevous versions.
    """
    print("\n" + "="*60)
    print(" TRAINING SELF-PLAY")
    print("="*60)
    
    win_rates_vs_random = []
    win_rates_vs_previous = []
    generations = []
    
    current_agent = agent
    previous_agent = current_agent.clone()
    
    for gen in range(num_generations):
        print(f"\n Generation {gen + 1}/{num_generations}")
        
        current_agent.set_training(True)
        
        # Training against previous version
        for episode in tqdm(range(games_per_gen), desc=f"Gen {gen + 1}"):
            config = setup_config(
                max_round=10,
                initial_stack=1000,
                small_blind_amount=10
            )
            config.register_player(name="current", algorithm=current_agent)
            config.register_player(name="previous", algorithm=previous_agent)
            
            start_poker(config, verbose=0)
            
            # Periodic update
            if (episode + 1) % 50 == 0:
                current_agent.update()
        
        # Evaluation vs Random
        current_agent.set_training(False)
        results_random = run_games(current_agent, RandomPlayer(), num_games=eval_games)
        win_rate_random = results_random["player1_wins"] / eval_games
        win_rates_vs_random.append(win_rate_random)
        
        # Evaluation vs previous version
        results_prev = run_games(current_agent, previous_agent, num_games=eval_games)
        win_rate_prev = results_prev["player1_wins"] / eval_games
        win_rates_vs_previous.append(win_rate_prev)
        
        generations.append(gen + 1)
        
        print(f"    vs Random: {win_rate_random:.2%}")
        print(f"    vs Previous: {win_rate_prev:.2%}")
        
        # Update opponent if we improved
        if win_rate_prev > 0.55:
            previous_agent = current_agent.clone()
            print("    New version saved as opponent!")
    
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


def comprehensive_evaluation(q_agent: QLearningPlayer, ppo_agent: PPOPlayer, 
                             num_games: int = 200):
    """
    Complete evaluation of agents against different opponents.
    """
    print("\n" + "="*70)
    print(" COMPLETE AGENT EVALUATION")
    print("="*70)
    
    # Disable training mode
    q_agent.set_training(False)
    ppo_agent.set_training(False)
    
    # Create opponents
    opponents = {
        "Random": RandomPlayer(),
        "TAG (Tight-Aggressive)": TightAggressivePlayer(tightness=0.6, aggression=0.7),
        "LAG (Loose-Aggressive)": LooseAggressivePlayer(looseness=0.3, aggression=0.8),
        "Calling Station": CallingStationPlayer(call_threshold=0.3),
        "Honest": HonestPlayer(),
    }
    
    # Results
    results = {
        "Q-Learning": {},
        "PPO": {},
    }
    
    print(f"\n Evaluation on {num_games} games per matchup...\n")
    
    # Test Q-Learning against all opponents
    print(" Q-Learning:")
    for opp_name, opponent in opponents.items():
        result = run_games(q_agent, opponent, num_games=num_games)
        win_rate = result["player1_wins"] / num_games
        results["Q-Learning"][opp_name] = win_rate
        print(f"   vs {opp_name:25s}: {win_rate:6.1%} ({result['player1_wins']}/{num_games})")
    
    # Test PPO against all opponents
    print("\n PPO:")
    for opp_name, opponent in opponents.items():
        result = run_games(ppo_agent, opponent, num_games=num_games)
        win_rate = result["player1_wins"] / num_games
        results["PPO"][opp_name] = win_rate
        print(f"   vs {opp_name:25s}: {win_rate:6.1%} ({result['player1_wins']}/{num_games})")
    
    # Match direct: Q-Learning vs PPO
    print("\n  DIRECT MATCH:")
    result_q_vs_ppo = run_games(q_agent, ppo_agent, num_games=num_games)
    result_ppo_vs_q = run_games(ppo_agent, q_agent, num_games=num_games)
    
    q_total = result_q_vs_ppo["player1_wins"] + result_ppo_vs_q["player2_wins"]
    ppo_total = result_q_vs_ppo["player2_wins"] + result_ppo_vs_q["player1_wins"]
    total = 2 * num_games
    
    print(f"   Q-Learning: {q_total}/{total} ({q_total/total:.1%})")
    print(f"   PPO:        {ppo_total}/{total} ({ppo_total/total:.1%})")
    
    # Moyennes
    print("\n AVERAGES:")
    q_avg = np.mean(list(results["Q-Learning"].values()))
    ppo_avg = np.mean(list(results["PPO"].values()))
    print(f"   Q-Learning (average all opponents): {q_avg:.1%}")
    print(f"   PPO (average all opponents):        {ppo_avg:.1%}")
    
    # Performance contre opponents non-random
    q_avg_hard = np.mean([v for k, v in results["Q-Learning"].items() if k != "Random"])
    ppo_avg_hard = np.mean([v for k, v in results["PPO"].items() if k != "Random"])
    print(f"\n   Q-Learning (excluding Random): {q_avg_hard:.1%}")
    print(f"   PPO (excluding Random):        {ppo_avg_hard:.1%}")
    
    # Comparison graph
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graph 1: Performance par adversaire
    ax1 = axes[0]
    x = np.arange(len(opponents))
    width = 0.35
    
    q_rates = [results["Q-Learning"][opp] for opp in opponents.keys()]
    ppo_rates = [results["PPO"][opp] for opp in opponents.keys()]
    
    bars1 = ax1.bar(x - width/2, q_rates, width, label='Q-Learning', color='#3498db')
    bars2 = ax1.bar(x + width/2, ppo_rates, width, label='PPO', color='#e74c3c')
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% (baseline)')
    ax1.set_ylabel('Win rate', fontsize=12)
    ax1.set_xlabel('Opponent', fontsize=12)
    ax1.set_title('Performance against different opponents', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([o.replace(' ', '\n') for o in opponents.keys()], fontsize=9)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Graph 2: Summary
    ax2 = axes[1]
    categories = ['vs Random', 'vs TAG', 'vs Calling\nStation', 'vs Honest', 'MOYENNE\n(excluding Random)', 'vs Autre RL']
    q_summary = [
        results["Q-Learning"]["Random"],
        results["Q-Learning"]["TAG (Tight-Aggressive)"],
        results["Q-Learning"]["Calling Station"],
        results["Q-Learning"]["Honest"],
        q_avg_hard,
        q_total / total
    ]
    ppo_summary = [
        results["PPO"]["Random"],
        results["PPO"]["TAG (Tight-Aggressive)"],
        results["PPO"]["Calling Station"],
        results["PPO"]["Honest"],
        ppo_avg_hard,
        ppo_total / total
    ]
    
    x2 = np.arange(len(categories))
    bars3 = ax2.bar(x2 - width/2, q_summary, width, label='Q-Learning', color='#3498db')
    bars4 = ax2.bar(x2 + width/2, ppo_summary, width, label='PPO', color='#e74c3c')
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Win rate', fontsize=12)
    ax2.set_title('Performance summary', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Graph saved: comprehensive_evaluation.png")
    
    return results


def compare_agents(q_agent: QLearningPlayer, ppo_agent: PPOPlayer, 
                   num_games: int = 500):
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
    PPO_EPISODES = 2000    # Episodes for PPO
    SELF_PLAY_GENS = 5     # Self-play generations
    GAMES_PER_GEN = 300    # Games per generation
    
    # -------------------------------------------------------------------------
    # Step 1: Train Q-Learning vs Random
    # -------------------------------------------------------------------------
    q_agent = train_qlearning_vs_random(num_episodes=Q_EPISODES, eval_interval=500)
    q_agent.save("q_learning_agent.pkl")
    print(f" Agent Q-Learning saved: q_learning_agent.pkl")
    
    # -------------------------------------------------------------------------
    # Step 2: Train PPO vs Random
    # -------------------------------------------------------------------------
    ppo_agent = train_ppo_vs_random(num_episodes=PPO_EPISODES, eval_interval=200)
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
    # Étape 4: Comparaison finale
    # -------------------------------------------------------------------------
    compare_agents(q_agent, ppo_agent_selfplay, num_games=500)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED!")
    print("="*60)
    print("""
    Generated files:
       • q_learning_agent.pkl     - Trained Q-Learning agent
       • ppo_agent_v1.pt          - Agent PPO (vs Random)
       • ppo_agent_selfplay.pt    - Agent PPO (after Self-Play)
       • qlearning_vs_random.png  - Evolution graph Q-Learning
       • ppo_vs_random.png        - Evolution graph PPO
       • self_play_evolution.png  - Graph Self-Play
       • agents_comparison.png    - Comparaison finale
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
        with open("q_learning_agent.pkl", "rb") as f:
            q_agent.q_table = pickle.load(f)
        print(" Agent Q-Learning loaded (q_learning_agent.pkl)")
    except FileNotFoundError:
        print(" q_learning_agent.pkl non found. Train first with main()")
        return
    
    # Charger PPO
    ppo_agent = PPOPlayer()
    try:
        ppo_agent.load("ppo_agent_v1.pt")
        print(" Agent PPO loaded (ppo_agent_v1.pt)")
    except FileNotFoundError:
        print(" ppo_agent_v1.pt non found. Train first with main()")
        return
    
    # Run complete evaluation
    comprehensive_evaluation(q_agent, ppo_agent, num_games=300)
    
    print("\n Evaluation completed !")
    print(" Graph saved: comprehensive_evaluation.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate_only()
    else:
        main()

