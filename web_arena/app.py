
from flask import Flask, render_template, jsonify, request
import threading
import sys
import os
import traceback

# Add parent directory to path to import poker agents
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PARENT_DIR)

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.engine.dealer import Dealer
from poker_rl_agent_v2 import (
    PPOPlayer, RandomPlayer, TightAggressivePlayer,
    LooseAggressivePlayer, CallingStationPlayer, HonestPlayer
)
from player import HumanWebPlayer, BotWrapper, SpectatorBotWrapper

# Try importing diffusion agent (optional)
try:
    from train_diffusion_pipeline import PPOPlayerWithDiffusion
    HAS_DIFFUSION = True
except ImportError:
    HAS_DIFFUSION = False

HEURISTIC_BOTS = {
    "random": ("Random", RandomPlayer),
    "tag": ("Tight Aggressive", TightAggressivePlayer),
    "lag": ("Loose Aggressive", LooseAggressivePlayer),
    "calling_station": ("Calling Station", CallingStationPlayer),
    "honest": ("Honest", HonestPlayer),
}

app = Flask(__name__)

# --- Shared Game State ---
def make_fresh_state():
    return {
        "waiting_for_action": False,
        "valid_actions": [],
        "hole_card": [],
        "community_card": [],
        "pot": 0,
        "seats": [],
        "logs": [],
        "chosen_action": None,
        "chosen_amount": 0,
        "game_running": False,
        "raise_limits": {"min": 0, "max": 0},
        "bot_type": "",
        "has_diffusion": HAS_DIFFUSION,
        "bot_cards": [],
        "bot_cards_revealed": False,
        "last_bot_action": "",
        "showdown": False,
        "game_result": "",
        "waiting_for_next_hand": False,
        "round_count": 0,
        "street": "",
        "round_winner": "",
        "round_winner_amount": 0,
        "round_winner_names": [],
        "call_amount": 0,
        "human_stack": 1000,
        "can_raise": True,
        # Buy-in / persistent stacks
        "persistent_human_stack": 0,
        "persistent_bot_stack": 1000,
        "waiting_for_buyin": False,
        "human_total_buyin": 0,
        "bot_total_buyin": 0,
        # BvB spectator mode
        "mode": "pvb",
        "bvb_paused": False,
        "bvb_speed": 1.0,
        "bot1_cards": [],
        "bot2_cards": [],
        "bot1_name": "",
        "bot2_name": "",
        "bot1_last_action": "",
        "bot2_last_action": "",
    }

game_state = make_fresh_state()
action_event = threading.Event()
next_hand_event = threading.Event()
buyin_event = threading.Event()


def load_bot(bot_type="auto"):
    """Loads the best available bot. Returns (bot, bot_name)."""
    models_dir = PARENT_DIR

    # 0. Heuristic bots
    if bot_type in HEURISTIC_BOTS:
        name, cls = HEURISTIC_BOTS[bot_type]
        return cls(), name

    # 1. Try PPO+Diffusion
    if bot_type in ("diffusion", "auto") and HAS_DIFFUSION:
        diff_model = os.path.join(models_dir, "ppo_diffusion.pt")
        if os.path.exists(diff_model):
            bot = PPOPlayerWithDiffusion()
            bot.load(diff_model)
            bot.set_training(False)
            return bot, "PPO + Diffusion"

    # 2. Try PPO (prefer fixed v2 curriculum model)
    for name in ("ppo_agent_fixed_v2.pt", "ppo_agent_selfplay.pt", "ppo_agent_v1.pt"):
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            bot = PPOPlayer()
            bot.load(path)
            bot.set_training(False)
            return bot, f"PPO ({name})"

    # 3. Untrained fallback
    bot = PPOPlayer()
    bot.set_training(False)
    return bot, "PPO (untrained)"


def custom_start_poker(config, player_stacks, verbose=1):
    """Like start_poker(), but allows per-player stacks by hacking the Dealer internals."""
    config.validation()
    dealer = Dealer(config.sb_amount, config.initial_stack, config.ante)
    dealer.set_verbose(verbose)
    dealer.set_blind_structure(config.blind_structure)
    for info in config.players_info:
        dealer.register_player(info["name"], info["algorithm"])

    # Override per-player stacks before the game starts
    for player in dealer.table.seats.players:
        if player.name in player_stacks:
            player.stack = player_stacks[player.name]

    result_message = dealer.start_game(config.max_round)
    return {
        "players": result_message["message"]["game_information"]["seats"]
    }


def game_thread_function(bot_type="auto", mode="pvb", bot2_type="random"):
    """Runs the poker game in a background thread with persistent stacks."""
    global game_state, buyin_event

    if mode == "bvb":
        game_thread_function_bvb(bot_type, bot2_type)
        return

    print("Starting game thread...")
    game_state["game_running"] = True
    game_state["mode"] = "pvb"
    game_state["logs"].append("Game started!")

    # Load bot once (reused across sessions)
    try:
        bot, bot_name = load_bot(bot_type)
        game_state["logs"].append(f"Bot loaded: {bot_name}")
        game_state["bot_type"] = bot_name
    except Exception as e:
        bot = PPOPlayer()
        bot.set_training(False)
        bot_name = "PPO (fallback)"
        game_state["logs"].append(f"Error loading bot: {e}")
        traceback.print_exc()

    # Create player objects (reused across sessions)
    human_player = HumanWebPlayer(game_state, action_event, next_hand_event)
    wrapped_bot = BotWrapper(bot, game_state)

    while game_state["game_running"]:
        human_stack = game_state["persistent_human_stack"]
        bot_stack = game_state["persistent_bot_stack"]

        # Human rebuy if busted
        if human_stack <= 0:
            game_state["waiting_for_buyin"] = True
            game_state["logs"].append("You're out of chips! Buy in to continue.")
            buyin_event.wait()
            buyin_event.clear()
            game_state["waiting_for_buyin"] = False
            if not game_state["game_running"]:
                break
            human_stack = game_state["persistent_human_stack"]

        # Bot auto-rebuy if busted
        if bot_stack <= 0:
            game_state["persistent_bot_stack"] = 1000
            game_state["bot_total_buyin"] = game_state.get("bot_total_buyin", 0) + 1000
            bot_stack = 1000
            game_state["logs"].append("Bot rebuys for 1000 chips.")

        # Set up config for this session
        config = setup_config(
            max_round=9999,
            initial_stack=max(human_stack, bot_stack),
            small_blind_amount=10
        )
        config.register_player(name="Human", algorithm=human_player)
        config.register_player(name="PPO_Bot", algorithm=wrapped_bot)

        # Run session with custom per-player stacks
        try:
            result = custom_start_poker(
                config,
                {"Human": human_stack, "PPO_Bot": bot_stack},
                verbose=1
            )
        except Exception as e:
            traceback.print_exc()
            game_state["logs"].append(f"GAME ERROR: {e}")
            break

        # Update persistent stacks from result
        for p in result["players"]:
            if p["name"] == "Human":
                game_state["persistent_human_stack"] = p["stack"]
            else:
                game_state["persistent_bot_stack"] = p["stack"]

        if not game_state["game_running"]:
            break

    game_state["game_running"] = False
    game_state["waiting_for_action"] = False
    game_state["logs"].append("Game session ended.")
    print("Game thread finished.")


def game_thread_function_bvb(bot1_type, bot2_type):
    """Runs Bot vs Bot spectator game in background thread."""
    global game_state
    game_state["game_running"] = True
    game_state["mode"] = "bvb"
    game_state["logs"].append("Bot vs Bot match started!")

    # Load both bots
    try:
        bot1, bot1_name = load_bot(bot1_type)
        game_state["bot1_name"] = bot1_name
        game_state["logs"].append(f"Bot 1: {bot1_name}")
    except Exception as e:
        bot1 = RandomPlayer()
        bot1_name = "Random (fallback)"
        game_state["bot1_name"] = bot1_name
        traceback.print_exc()

    try:
        bot2, bot2_name = load_bot(bot2_type)
        game_state["bot2_name"] = bot2_name
        game_state["logs"].append(f"Bot 2: {bot2_name}")
    except Exception as e:
        bot2 = RandomPlayer()
        bot2_name = "Random (fallback)"
        game_state["bot2_name"] = bot2_name
        traceback.print_exc()

    def get_delay():
        return game_state.get("bvb_speed", 1.0)

    wrapped1 = SpectatorBotWrapper(bot1, game_state, "bot1", get_delay)
    wrapped2 = SpectatorBotWrapper(bot2, game_state, "bot2", get_delay)

    game_state["persistent_human_stack"] = 1000  # reused as bot1 stack display
    game_state["persistent_bot_stack"] = 1000    # bot2 stack display

    while game_state["game_running"]:
        b1_stack = game_state["persistent_human_stack"]
        b2_stack = game_state["persistent_bot_stack"]

        # Auto-rebuy if either bot busts
        if b1_stack <= 0:
            game_state["persistent_human_stack"] = 1000
            b1_stack = 1000
            game_state["bot_total_buyin"] = game_state.get("bot_total_buyin", 0) + 1000
            game_state["logs"].append(f"{bot1_name} rebuys for 1000.")
        if b2_stack <= 0:
            game_state["persistent_bot_stack"] = 1000
            b2_stack = 1000
            game_state["bot_total_buyin"] = game_state.get("bot_total_buyin", 0) + 1000
            game_state["logs"].append(f"{bot2_name} rebuys for 1000.")

        config = setup_config(
            max_round=9999,
            initial_stack=max(b1_stack, b2_stack),
            small_blind_amount=10
        )
        config.register_player(name=bot1_name, algorithm=wrapped1)
        config.register_player(name=bot2_name, algorithm=wrapped2)

        try:
            result = custom_start_poker(
                config,
                {bot1_name: b1_stack, bot2_name: b2_stack},
                verbose=0
            )
        except Exception as e:
            traceback.print_exc()
            game_state["logs"].append(f"GAME ERROR: {e}")
            break

        # Update persistent stacks from result
        for p in result["players"]:
            if p["name"] == bot1_name:
                game_state["persistent_human_stack"] = p["stack"]
            else:
                game_state["persistent_bot_stack"] = p["stack"]

        if not game_state["game_running"]:
            break

    game_state["game_running"] = False
    game_state["logs"].append("Bot vs Bot match ended.")
    print("BvB game thread finished.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stop', methods=['POST'])
def stop_game():
    """Force-stop the current game."""
    global game_state, action_event, next_hand_event, buyin_event
    if game_state["game_running"]:
        game_state["game_running"] = False
        game_state["game_result"] = "aborted"
        # Unblock any waiting threads
        action_event.set()
        next_hand_event.set()
        buyin_event.set()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


@app.route('/api/buyin', methods=['POST'])
def buyin():
    """Buy in for 1000 chips. Starts game or rebuys mid-game."""
    global game_state, action_event, next_hand_event, buyin_event

    buyin_amount = 1000

    # Mid-game rebuy (human busted)
    if game_state["game_running"] and game_state.get("waiting_for_buyin"):
        game_state["persistent_human_stack"] += buyin_amount
        game_state["human_total_buyin"] = game_state.get("human_total_buyin", 0) + buyin_amount
        game_state["logs"].append(f"You bought in for {buyin_amount} chips.")
        buyin_event.set()
        return jsonify({"status": "rebuy", "stack": game_state["persistent_human_stack"]})

    # Fresh game start
    if game_state["game_running"]:
        # Force stop first
        game_state["game_running"] = False
        action_event.set()
        next_hand_event.set()
        buyin_event.set()
        import time; time.sleep(0.5)

    # Full reset
    game_state = make_fresh_state()
    action_event = threading.Event()
    next_hand_event = threading.Event()
    buyin_event = threading.Event()

    data = request.json or {}
    bot_type = data.get("bot_type", "auto")
    mode = data.get("mode", "pvb")
    bot2_type = data.get("bot2_type", "random")

    game_state["persistent_human_stack"] = buyin_amount
    game_state["persistent_bot_stack"] = 1000
    game_state["human_total_buyin"] = buyin_amount
    game_state["bot_total_buyin"] = 1000

    thread = threading.Thread(
        target=game_thread_function,
        args=(bot_type, mode, bot2_type)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "stack": buyin_amount})


@app.route('/api/start', methods=['POST'])
def start_game():
    """Legacy endpoint — redirects to buyin."""
    return buyin()


@app.route('/api/state')
def get_state():
    return jsonify(game_state)


@app.route('/api/action', methods=['POST'])
def receive_action():
    data = request.json
    print(f"[Flask] /api/action received: {data}")
    action_type = data.get('action')
    amount = data.get('amount', 0)

    if not game_state["waiting_for_action"]:
        return jsonify({"status": "error", "message": "Not your turn"})

    game_state["chosen_action"] = action_type
    game_state["chosen_amount"] = int(amount)

    # Unblock the game thread
    action_event.set()

    return jsonify({"status": "accepted", "action": action_type, "amount": amount})


@app.route('/api/next_hand', methods=['POST'])
def next_hand():
    if not game_state.get("waiting_for_next_hand"):
        return jsonify({"status": "error", "message": "Not waiting for next hand"})
    next_hand_event.set()
    return jsonify({"status": "ok"})


@app.route('/api/bvb/pause', methods=['POST'])
def bvb_pause():
    game_state["bvb_paused"] = True
    return jsonify({"status": "paused"})


@app.route('/api/bvb/resume', methods=['POST'])
def bvb_resume():
    game_state["bvb_paused"] = False
    return jsonify({"status": "resumed"})


@app.route('/api/bvb/speed', methods=['POST'])
def bvb_speed():
    speed = request.json.get("speed", 1.0) if request.json else 1.0
    game_state["bvb_speed"] = float(speed)
    return jsonify({"status": "ok", "speed": game_state["bvb_speed"]})


if __name__ == '__main__':
    print("=" * 50)
    print("  Poker AI Arena")
    print(f"  Diffusion model available: {HAS_DIFFUSION}")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, use_reloader=False)
