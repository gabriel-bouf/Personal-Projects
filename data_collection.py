"""
Data Collection for Diffusion Model Training
=============================================
Collects game histories with full hand information for both players.
Uses instrumented player wrappers that share a data collector object
to capture hole cards from both sides.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from diffusion_utils import (
    encode_hand,
    encode_action_type,
    parse_card_string,
    SUIT_MAP,
    COND_DIM,
    HAND_DIM,
)


MAX_STACK = 2000.0
STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class HandRecord:
    """A single training example: opponent hand + game context."""
    # Target: opponent's hole cards encoded as 4D vector
    hand_vector: np.ndarray  # (4,)
    # Conditioning: observable game state
    condition_vector: np.ndarray  # (27,)
    # Whether hand went to showdown (higher quality label)
    went_to_showdown: bool = False


class SharedDataCollector:
    """
    Shared object between two InstrumentedPlayers.
    Captures hole cards from both players and builds training records.
    """

    def __init__(self):
        self.records: List[HandRecord] = []
        self._hole_cards: Dict[str, List[str]] = {}
        self._current_round_actions: Dict[str, List[Tuple[str, float]]] = {}
        self._initial_stacks: Dict[str, float] = {}

    def record_hole_cards(self, uuid: str, hole_card: List[str]):
        self._hole_cards[uuid] = hole_card

    def record_initial_stack(self, uuid: str, stack: float):
        self._initial_stacks[uuid] = stack

    def record_action(self, actor_uuid: str, action: str, amount: float):
        if actor_uuid not in self._current_round_actions:
            self._current_round_actions[actor_uuid] = []
        self._current_round_actions[actor_uuid].append((action, amount))

    def finalize_round(self, round_state: Dict, hand_info: List[Dict]):
        """Build training records from the completed round."""
        went_to_showdown = len(hand_info) > 0

        uuids = list(self._hole_cards.keys())
        if len(uuids) < 2:
            self._reset_round()
            return

        # For each player, the OTHER player is the "opponent" to predict
        for my_uuid in uuids:
            opp_uuid = [u for u in uuids if u != my_uuid]
            if not opp_uuid:
                continue
            opp_uuid = opp_uuid[0]

            opp_hole = self._hole_cards.get(opp_uuid)
            if not opp_hole:
                continue

            # Encode opponent hand (target)
            r1, s1 = parse_card_string(opp_hole[0])
            r2, s2 = parse_card_string(opp_hole[1])
            hand_vec = encode_hand(r1, s1, r2, s2)

            # Build conditioning vector from game state
            cond_vec = self._build_condition(round_state, opp_uuid, my_uuid)

            self.records.append(HandRecord(
                hand_vector=hand_vec,
                condition_vector=cond_vec,
                went_to_showdown=went_to_showdown,
            ))

        self._reset_round()

    def _build_condition(self, round_state: Dict, opp_uuid: str, my_uuid: str) -> np.ndarray:
        """Builds the 27D conditioning vector from observable game state."""
        features = []

        # 1. Street one-hot (4)
        street = round_state.get("street", "preflop")
        street_onehot = [0.0, 0.0, 0.0, 0.0]
        street_idx = STREETS.index(street) if street in STREETS else 0
        street_onehot[street_idx] = 1.0
        features.extend(street_onehot)

        # 2. Community cards (10 = 5 slots x 2 values)
        community = round_state.get("community_card", [])
        for i in range(5):
            if i < len(community):
                rank, suit_char = parse_card_string(community[i])
                features.append(rank / 14.0)
                features.append(SUIT_MAP.get(suit_char, 0) / 3.0)
            else:
                features.append(0.0)
                features.append(0.0)

        # 3. Game context (5)
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        features.append(min(pot / MAX_STACK, 1.0))

        opp_stack = 0
        for seat in round_state.get("seats", []):
            if seat["uuid"] == opp_uuid:
                opp_stack = seat.get("stack", 0)
                break
        features.append(min(opp_stack / MAX_STACK, 1.0))

        # Opponent raises count
        opp_actions = self._current_round_actions.get(opp_uuid, [])
        num_raises = sum(1 for a, _ in opp_actions if a == "raise")
        features.append(min(num_raises / 5.0, 1.0))

        # Opponent total bet
        total_bet = sum(amt for _, amt in opp_actions)
        features.append(min(total_bet / MAX_STACK, 1.0))

        # Opponent VPIP (voluntarily put money in)
        vpip = 1.0 if any(a in ("call", "raise") for a, _ in opp_actions) else 0.0
        features.append(vpip)

        # 4. Opponent action per street (4 streets x 2 = 8)
        action_histories = round_state.get("action_histories", {})
        for st in STREETS:
            street_actions = action_histories.get(st, [])
            opp_action_type = 0.0
            opp_action_amount = 0.0
            for a in street_actions:
                a_uuid = a.get("uuid") or a.get("player_uuid", "")
                if a_uuid == opp_uuid:
                    opp_action_type = encode_action_type(a.get("action", ""))
                    opp_action_amount = min(a.get("amount", 0) / MAX_STACK, 1.0)
            features.append(opp_action_type)
            features.append(opp_action_amount)

        cond = np.array(features, dtype=np.float32)

        # Pad or truncate to COND_DIM
        if len(cond) < COND_DIM:
            cond = np.pad(cond, (0, COND_DIM - len(cond)))
        elif len(cond) > COND_DIM:
            cond = cond[:COND_DIM]

        return cond

    def _reset_round(self):
        self._hole_cards.clear()
        self._current_round_actions.clear()
        self._initial_stacks.clear()


class InstrumentedPlayer(BasePokerPlayer):
    """
    Wraps any BasePokerPlayer to intercept game messages
    and record data to a SharedDataCollector.
    """

    def __init__(self, inner_player: BasePokerPlayer, collector: SharedDataCollector):
        super().__init__()
        self.inner = inner_player
        self.collector = collector

    def set_uuid(self, uuid):
        self.uuid = uuid
        self.inner.set_uuid(uuid)

    def declare_action(self, valid_actions, hole_card, round_state):
        return self.inner.declare_action(valid_actions, hole_card, round_state)

    def receive_game_start_message(self, game_info):
        self.inner.receive_game_start_message(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.collector.record_hole_cards(self.uuid, hole_card)
        for seat in seats:
            if seat["uuid"] == self.uuid:
                self.collector.record_initial_stack(self.uuid, seat["stack"])
                break
        self.inner.receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        self.inner.receive_street_start_message(street, round_state)

    def receive_game_update_message(self, action, round_state):
        actor_uuid = action.get("player_uuid", "")
        action_type = action.get("action", "")
        amount = action.get("amount", 0)
        self.collector.record_action(actor_uuid, action_type, amount)
        self.inner.receive_game_update_message(action, round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.collector.finalize_round(round_state, hand_info)
        self.inner.receive_round_result_message(winners, hand_info, round_state)


class HandDataset(Dataset):
    """PyTorch Dataset for training the diffusion model."""

    def __init__(self, records: List[HandRecord], showdown_only: bool = False):
        if showdown_only:
            records = [r for r in records if r.went_to_showdown]
        self.hand_vectors = np.array([r.hand_vector for r in records], dtype=np.float32)
        self.conditions = np.array([r.condition_vector for r in records], dtype=np.float32)

    def __len__(self):
        return len(self.hand_vectors)

    def __getitem__(self, idx):
        return {
            "hand_vector": torch.FloatTensor(self.hand_vectors[idx]),
            "condition": torch.FloatTensor(self.conditions[idx]),
        }


def collect_game_data(num_games: int = 10000, max_round: int = 10,
                      initial_stack: int = 1000, small_blind: int = 10) -> List[HandRecord]:
    """
    Runs games between all pairs of heuristic agents and collects hand data.
    Returns list of HandRecord.
    """
    # Import heuristic agents from main module
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poker_rl_agent_v2 import (
        RandomPlayer,
        HonestPlayer,
        CallingStationPlayer,
        TightAggressivePlayer,
        LooseAggressivePlayer,
    )

    collector = SharedDataCollector()

    agent_factories = [
        ("Random", lambda: RandomPlayer()),
        ("Honest", lambda: HonestPlayer()),
        ("CallingStation", lambda: CallingStationPlayer()),
        ("TAG", lambda: TightAggressivePlayer()),
        ("LAG", lambda: LooseAggressivePlayer()),
    ]

    num_pairs = len(agent_factories) ** 2
    games_per_pair = max(1, num_games // num_pairs)

    from tqdm import tqdm
    total_games = games_per_pair * num_pairs
    pbar = tqdm(total=total_games, desc="Collecting data")

    for name1, factory1 in agent_factories:
        for name2, factory2 in agent_factories:
            for _ in range(games_per_pair):
                p1 = InstrumentedPlayer(factory1(), collector)
                p2 = InstrumentedPlayer(factory2(), collector)

                config = setup_config(
                    max_round=max_round,
                    initial_stack=initial_stack,
                    small_blind_amount=small_blind,
                )
                config.register_player(name=name1, algorithm=p1)
                config.register_player(name=name2, algorithm=p2)

                try:
                    start_poker(config, verbose=0)
                except Exception:
                    pass

                pbar.update(1)

    pbar.close()

    showdown_count = sum(1 for r in collector.records if r.went_to_showdown)
    print(f"Collected {len(collector.records)} records ({showdown_count} from showdowns)")

    return collector.records
