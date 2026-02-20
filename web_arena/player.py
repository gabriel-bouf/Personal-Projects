
from pypokerengine.players import BasePokerPlayer
import threading
import time


class BotWrapper(BasePokerPlayer):
    """Wraps any bot to capture its hole cards for the UI."""

    def __init__(self, bot, shared_state):
        super().__init__()
        self.bot = bot
        self.shared_state = shared_state

    def declare_action(self, valid_actions, hole_card, round_state):
        return self.bot.declare_action(valid_actions, hole_card, round_state)

    def receive_game_start_message(self, game_info):
        # Propagate uuid set by pypokerengine to inner bot
        if hasattr(self, 'uuid'):
            self.bot.uuid = self.uuid
        self.bot.receive_game_start_message(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Capture bot's hole cards
        self.shared_state["bot_cards"] = hole_card
        self.bot.receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        self.bot.receive_street_start_message(street, round_state)

    def receive_game_update_message(self, action, round_state):
        self.bot.receive_game_update_message(action, round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.shared_state["showdown"] = True
        self.bot.receive_round_result_message(winners, hand_info, round_state)


class SpectatorBotWrapper(BasePokerPlayer):
    """Wraps a bot for Bot-vs-Bot spectator mode with delays and full state capture."""

    def __init__(self, bot, shared_state, player_label, delay_func):
        super().__init__()
        self.bot = bot
        self.shared_state = shared_state
        self.player_label = player_label  # "bot1" or "bot2"
        self.delay_func = delay_func
        self.uuid_map = {}

    def declare_action(self, valid_actions, hole_card, round_state):
        # Wait for pause to be released
        while self.shared_state.get("bvb_paused", False):
            if not self.shared_state.get("game_running", True):
                return "fold", 0
            time.sleep(0.1)

        # Apply delay for watchability
        delay = self.delay_func()
        if delay > 0:
            time.sleep(delay)

        # Update UI state before bot acts
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["pot"] = pot
        self.shared_state["community_card"] = round_state.get("community_card", [])
        self.shared_state["seats"] = round_state.get("seats", [])

        action, amount = self.bot.declare_action(valid_actions, hole_card, round_state)

        # Track last action for display
        action_str = f"{action} {amount}"
        self.shared_state[f"{self.player_label}_last_action"] = action_str

        return action, amount

    def receive_game_start_message(self, game_info):
        if hasattr(self, 'uuid'):
            self.bot.uuid = self.uuid
        self.bot.receive_game_start_message(game_info)
        for seat in game_info["seats"]:
            self.uuid_map[seat["uuid"]] = seat["name"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.shared_state[f"{self.player_label}_cards"] = hole_card
        self.shared_state["round_count"] = round_count
        self.shared_state["seats"] = seats
        self.shared_state["community_card"] = []
        self.shared_state["showdown"] = False
        self.shared_state["bot1_last_action"] = ""
        self.shared_state["bot2_last_action"] = ""
        if self.player_label == "bot1":
            self.shared_state["logs"].append(f"--- Round {round_count} ---")
        self.bot.receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        self.shared_state["street"] = street
        self.shared_state["community_card"] = round_state["community_card"]
        if self.player_label == "bot1":
            self.shared_state["logs"].append(f"Street: {street}")
        self.bot.receive_street_start_message(street, round_state)

    def receive_game_update_message(self, action, round_state):
        self.shared_state["seats"] = round_state["seats"]
        self.shared_state["community_card"] = round_state["community_card"]
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["pot"] = pot

        player_uuid = action["player_uuid"]
        player_name = self.uuid_map.get(player_uuid, player_uuid)
        action_str = f"{action['action']} {action['amount']}"
        self.shared_state["logs"].append(f"{player_name}: {action_str}")

        self.bot.receive_game_update_message(action, round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.shared_state["showdown"] = True
        self.shared_state["seats"] = round_state["seats"]

        winner_names = [self.uuid_map.get(w['uuid'], w['uuid']) for w in winners]
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["round_winner_names"] = winner_names
        self.shared_state["round_winner_amount"] = pot

        if self.player_label == "bot1":
            self.shared_state["logs"].append(
                f"Winner: {', '.join(winner_names)} (+{pot})"
            )
            # Inter-hand pause (only bot1 handles this to avoid double-delay)
            if self.shared_state.get("game_running", True):
                delay = self.delay_func()
                time.sleep(max(delay * 2, 1.0))

        self.bot.receive_round_result_message(winners, hand_info, round_state)


class HumanWebPlayer(BasePokerPlayer):
    """
    web-interface player that pauses the game 
    waiting for human input from the browser.
    """
    
    def __init__(self, shared_state, action_event, next_hand_event=None):
        super().__init__()
        self.shared_state = shared_state
        self.action_event = action_event
        self.next_hand_event = next_hand_event
        self.uuid_map = {}
        print("[WebPlayer] Initialized")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        print(f"[WebPlayer] declare_action called. Valid: {len(valid_actions)} actions")

        # Build lookup of valid actions
        valid_map = {}
        for a in valid_actions:
            valid_map[a["action"]] = a

        # Determine call amount from engine (the correct one)
        call_amount = 0
        if "call" in valid_map:
            call_amount = valid_map["call"]["amount"]

        # Determine raise limits
        raise_min, raise_max = -1, -1
        if "raise" in valid_map:
            amt = valid_map["raise"]["amount"]
            if isinstance(amt, dict):
                raise_min = amt.get("min", -1)
                raise_max = amt.get("max", -1)
            else:
                raise_min = raise_max = amt

        # Find human stack for all-in detection
        human_stack = 0
        for seat in round_state.get("seats", []):
            if seat["uuid"] == self.uuid:
                human_stack = seat["stack"]
                break

        # Update pot from round_state (so it's not stale at 0)
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["pot"] = pot
        self.shared_state["community_card"] = round_state.get("community_card", [])
        self.shared_state["seats"] = round_state.get("seats", [])

        # Update state for the frontend
        self.shared_state["waiting_for_action"] = True
        self.shared_state["valid_actions"] = valid_actions
        self.shared_state["hole_card"] = hole_card
        self.shared_state["round_state"] = round_state
        self.shared_state["call_amount"] = call_amount
        self.shared_state["human_stack"] = human_stack
        self.shared_state["raise_limits"] = {
            "min": max(0, raise_min),
            "max": max(0, raise_max),
        }
        self.shared_state["can_raise"] = raise_min > 0 and raise_max > 0

        print(f"[WebPlayer] Waiting... call={call_amount}, raise=[{raise_min},{raise_max}], stack={human_stack}")

        # Wait for the front-end to set the event (API call /action)
        self.action_event.wait()
        self.action_event.clear()

        # If game was aborted, fold immediately
        if not self.shared_state.get("game_running", True):
            self.shared_state["waiting_for_action"] = False
            return "fold", 0

        # Get the action set by the API
        action_type = self.shared_state.get("chosen_action", "fold")
        amount = self.shared_state.get("chosen_amount", 0)

        # --- Validate & correct the action before returning to engine ---
        if action_type == "call":
            # Always use the engine's call amount (handles all-in automatically)
            amount = call_amount
        elif action_type == "raise":
            if raise_min <= 0 or raise_max <= 0:
                # Can't raise — fall back to call (or fold if can't call)
                if "call" in valid_map:
                    action_type, amount = "call", call_amount
                else:
                    action_type, amount = "fold", 0
            else:
                # Clamp raise amount to valid range
                amount = max(raise_min, min(raise_max, amount))
        elif action_type == "fold":
            amount = 0
        else:
            # Unknown action — fold safely
            action_type, amount = "fold", 0

        self.shared_state["waiting_for_action"] = False
        print(f"[WebPlayer] Action sent to engine: {action_type} {amount}")

        return action_type, amount

    def receive_game_start_message(self, game_info):
        print(f"[WebPlayer] Game Started: {game_info}")
        self.shared_state["game_info"] = game_info
        # Map UUIDs to Names
        for seat in game_info["seats"]:
            self.uuid_map[seat["uuid"]] = seat["name"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(f"[WebPlayer] Round {round_count} Started")
        self.shared_state["round_count"] = round_count
        self.shared_state["hole_card"] = hole_card
        self.shared_state["seats"] = seats
        self.shared_state["community_card"] = []
        self.shared_state["pot"] = 0
        self.shared_state["bot_cards"] = []
        self.shared_state["bot_cards_revealed"] = False
        self.shared_state["last_bot_action"] = ""
        self.shared_state["showdown"] = False
        self.shared_state["round_winner"] = ""
        self.shared_state["round_winner_amount"] = 0
        self.shared_state["logs"].append(f"--- Round {round_count} ---")

    def receive_street_start_message(self, street, round_state):
        print(f"[WebPlayer] Street: {street}")
        self.shared_state["street"] = street
        self.shared_state["round_state"] = round_state
        self.shared_state["community_card"] = round_state["community_card"]
        self.shared_state["logs"].append(f"Street: {street}")
        
    def receive_game_update_message(self, action, round_state):
        print(f"[WebPlayer] Update: {action}")
        self.shared_state["round_state"] = round_state
        self.shared_state["seats"] = round_state["seats"]
        self.shared_state["community_card"] = round_state["community_card"]

        # Update pot
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["pot"] = pot

        player_uuid = action["player_uuid"]
        player_name = self.uuid_map.get(player_uuid, player_uuid)
        action_str = f"{action['action']} {action['amount']}"
        self.shared_state["logs"].append(f"{player_name}: {action_str}")

        # Track bot's last action for display
        if player_name != "Human":
            self.shared_state["last_bot_action"] = action_str

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(f"[WebPlayer] Round Result: {winners}")
        self.shared_state["round_state"] = round_state
        self.shared_state["seats"] = round_state["seats"]
        self.shared_state["winners"] = winners
        self.shared_state["showdown"] = True
        self.shared_state["bot_cards_revealed"] = True

        # Compute winner info for the UI banner
        winner_names = [self.uuid_map.get(w['uuid'], w['uuid']) for w in winners]
        pot_won = sum(w.get('stack', 0) for w in winners)  # approximate

        # Determine if human won or lost
        human_won = "Human" in winner_names
        if human_won:
            self.shared_state["round_winner"] = "you"
        else:
            self.shared_state["round_winner"] = "bot"
        self.shared_state["round_winner_names"] = winner_names

        # Find how much the pot was
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        self.shared_state["round_winner_amount"] = pot

        self.shared_state["logs"].append(
            f"{'YOU WIN' if human_won else 'Bot wins'} the pot ({pot} chips)"
        )

        # Pause between rounds so player can review bot cards
        if self.next_hand_event and self.shared_state.get("game_running", True):
            self.shared_state["waiting_for_next_hand"] = True
            print("[WebPlayer] Waiting for 'Next Hand'...")
            self.next_hand_event.wait()
            self.next_hand_event.clear()
            self.shared_state["waiting_for_next_hand"] = False
