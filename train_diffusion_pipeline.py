#!/usr/bin/env python3
"""
Diffusion-Augmented Poker RL Training Pipeline
===============================================
Three-phase training:
  Phase 1: Collect game data (opponent hands + game context)
  Phase 2: Train conditional diffusion model (DDPM)
  Phase 3: Train PPO with diffusion-augmented state vector

Usage:
  python train_diffusion_pipeline.py              # Run all 3 phases
  python train_diffusion_pipeline.py --phase 2    # Run only phase 2
  python train_diffusion_pipeline.py --eval       # Evaluate only
"""

import sys
import os
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_utils import (
    NoiseSchedule,
    DIFFUSION_CONFIG,
    HAND_DIM,
    COND_DIM,
    encode_action_type,
    parse_card_string,
    SUIT_MAP,
)
from diffusion_opponent_model import (
    ConditionalDenoiser,
    diffusion_loss,
    ddim_sample,
    compute_opponent_features,
)
from data_collection import (
    HandRecord,
    HandDataset,
    SharedDataCollector,
    InstrumentedPlayer,
    collect_game_data,
    STREETS,
    MAX_STACK,
)
from poker_rl_agent_v2 import (
    PPOPlayer,
    PPOMemory,
    ActorCritic,
    PPO_CONFIG,
    ACTIONS,
    NUM_ACTIONS,
    extract_state_vector,
    RandomPlayer,
    HonestPlayer,
    CallingStationPlayer,
    TightAggressivePlayer,
    LooseAggressivePlayer,
    run_games,
    evaluate_agent,
    evaluate_head_to_head,
)
from pypokerengine.api.game import setup_config, start_poker


# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def phase1_collect_data(num_games: int = 10000, save_path: str = "hand_data.pt"):
    """Collects game data and saves to disk."""
    print("\n" + "=" * 60)
    print(" PHASE 1: DATA COLLECTION")
    print("=" * 60)

    records = collect_game_data(num_games=num_games)

    torch.save({
        "records": records,
        "num_records": len(records),
        "num_showdown": sum(1 for r in records if r.went_to_showdown),
    }, save_path)

    print(f"Data saved to {save_path}")
    return records


# ============================================================================
# PHASE 2: TRAIN DIFFUSION MODEL
# ============================================================================

def phase2_train_diffusion(
    data_path: str = "hand_data.pt",
    save_path: str = "diffusion_model.pt",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
):
    """Trains the conditional DDPM on collected hand data."""
    print("\n" + "=" * 60)
    print(" PHASE 2: TRAIN DIFFUSION MODEL")
    print("=" * 60)

    # Load data
    data = torch.load(data_path, weights_only=False)
    records = data["records"]
    print(f"Loaded {len(records)} records")

    dataset = HandDataset(records, showdown_only=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Initialize model
    denoiser = ConditionalDenoiser(
        hand_dim=HAND_DIM,
        cond_dim=COND_DIM,
        hidden_dim=DIFFUSION_CONFIG["hidden_dim"],
    )
    noise_schedule = NoiseSchedule(
        num_timesteps=DIFFUSION_CONFIG["num_timesteps"],
        beta_start=DIFFUSION_CONFIG["beta_start"],
        beta_end=DIFFUSION_CONFIG["beta_end"],
    )

    param_count = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser parameters: {param_count:,}")

    optimizer = optim.Adam(denoiser.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []

    for epoch in tqdm(range(epochs), desc="Diffusion Training"):
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            x_0 = batch["hand_vector"]
            cond = batch["condition"]

            loss = diffusion_loss(denoiser, x_0, cond, noise_schedule)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch + 1}: Loss = {avg_loss:.6f}")

    # Save model
    torch.save({
        "denoiser_state_dict": denoiser.state_dict(),
        "diffusion_config": DIFFUSION_CONFIG,
        "training_losses": losses,
    }, save_path)

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("diffusion_training.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nModel saved to {save_path}")
    print(f"Training curve saved to diffusion_training.png")

    return denoiser, noise_schedule


# ============================================================================
# PPO PLAYER WITH DIFFUSION AUGMENTATION
# ============================================================================

class PPOPlayerWithDiffusion(PPOPlayer):
    """PPO agent with diffusion-based opponent modeling."""

    STATE_DIM = 34  # 28 base + 6 diffusion features

    def __init__(self, config: Dict = None, diffusion_model_path: str = None):
        # Initialize with augmented state dim
        self.config = config or PPO_CONFIG
        # Skip PPOPlayer.__init__ to control network creation
        super(PPOPlayer, self).__init__()  # Call BasePokerPlayer.__init__

        self.network = ActorCritic(self.STATE_DIM, NUM_ACTIONS)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config["lr"],
        )
        self.memory = PPOMemory()
        self.training = True
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.last_valid_mask = None
        self.state_mean = np.zeros(self.STATE_DIM, dtype=np.float32)
        self.state_m2 = np.ones(self.STATE_DIM, dtype=np.float32)
        self.state_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.losses = []
        self.initial_stack = 1000

        # Diffusion model
        self.denoiser = None
        self.noise_schedule = None
        self._opponent_actions_this_round = []

        if diffusion_model_path and os.path.exists(diffusion_model_path):
            self._load_diffusion_model(diffusion_model_path)

    def _load_diffusion_model(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint.get("diffusion_config", DIFFUSION_CONFIG)
        self.denoiser = ConditionalDenoiser(
            hand_dim=config["hand_dim"],
            cond_dim=config["cond_dim"],
            hidden_dim=config["hidden_dim"],
        )
        self.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        self.denoiser.eval()
        self.noise_schedule = NoiseSchedule(
            num_timesteps=config["num_timesteps"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
        )

    def declare_action(self, valid_actions, hole_card, round_state):
        base_state = extract_state_vector(hole_card, round_state, valid_actions, my_uuid=self.uuid)

        # Compute diffusion features
        diff_features = self._get_diffusion_features(round_state)

        # Augmented state + normalization
        state = np.concatenate([base_state, diff_features])
        state = self.normalize_state(state)

        # Validity mask
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
                if "raise" in valid_action_types:
                    valid_mask[i] = 1.0

        # Action selection (argmax in inference, sample in training)
        with torch.no_grad() if not self.training else torch.enable_grad():
            action_idx, log_prob, value = self.network.get_action(
                state, valid_mask, deterministic=not self.training
            )

        action_name = ACTIONS[action_idx]

        # Map to pypokerengine action
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
            raise_limits = {}
            for a in valid_actions:
                if a["action"] == "raise":
                    raise_limits = a["amount"]
                    break

            if not raise_limits:
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
                    amount = int(max(min_raise, min(pot_amount * 0.5, max_raise)))
                elif action_name == "raise_pot":
                    amount = int(max(min_raise, min(pot_amount, max_raise)))
                elif action_name == "all_in":
                    amount = max_raise
                else:
                    amount = min_raise

        # Save for PPO update
        self.last_state = state
        self.last_action = action_idx
        self.last_log_prob = log_prob
        self.last_value = value
        self.last_valid_mask = valid_mask

        return pypoker_action, amount

    def _get_diffusion_features(self, round_state: Dict) -> np.ndarray:
        """Generates opponent hand samples and extracts 6 statistical features."""
        if self.denoiser is None:
            return np.zeros(6, dtype=np.float32)

        cond = self._build_diffusion_conditioning(round_state)
        cond_tensor = torch.FloatTensor(cond).unsqueeze(0)

        with torch.no_grad():
            samples = ddim_sample(
                self.denoiser, cond_tensor, self.noise_schedule,
                num_samples=32, ddim_steps=20,
            )

        return compute_opponent_features(samples)

    def _build_diffusion_conditioning(self, round_state: Dict) -> np.ndarray:
        """Builds the 27D conditioning vector for the diffusion model."""
        features = []

        # Street one-hot (4)
        street = round_state.get("street", "preflop")
        street_onehot = [0.0, 0.0, 0.0, 0.0]
        if street in STREETS:
            street_onehot[STREETS.index(street)] = 1.0
        features.extend(street_onehot)

        # Community cards (10)
        community = round_state.get("community_card", [])
        for i in range(5):
            if i < len(community):
                rank, suit_char = parse_card_string(community[i])
                features.append(rank / 14.0)
                features.append(SUIT_MAP.get(suit_char, 0) / 3.0)
            else:
                features.append(0.0)
                features.append(0.0)

        # Game context (5)
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        features.append(min(pot / MAX_STACK, 1.0))

        # Opponent stack
        my_uuid = round_state.get("next_player")
        opp_stack = 0
        for seat in round_state.get("seats", []):
            if seat.get("uuid") != my_uuid:
                opp_stack = seat.get("stack", 0)
                break
        features.append(min(opp_stack / MAX_STACK, 1.0))

        # Opponent aggression from action histories
        action_histories = round_state.get("action_histories", {})
        num_opp_raises = 0
        total_opp_bet = 0
        opp_vpip = 0.0

        for st_actions in action_histories.values():
            for a in st_actions:
                a_uuid = a.get("uuid") or a.get("player_uuid", "")
                if a_uuid != my_uuid:
                    if a.get("action") == "raise":
                        num_opp_raises += 1
                    if a.get("action") in ("call", "raise"):
                        opp_vpip = 1.0
                    total_opp_bet += a.get("amount", 0)

        features.append(min(num_opp_raises / 5.0, 1.0))
        features.append(min(total_opp_bet / MAX_STACK, 1.0))
        features.append(opp_vpip)

        # Opponent action per street (8)
        for st in STREETS:
            street_actions = action_histories.get(st, [])
            opp_action_type = 0.0
            opp_action_amount = 0.0
            for a in street_actions:
                a_uuid = a.get("uuid") or a.get("player_uuid", "")
                if a_uuid != my_uuid:
                    opp_action_type = encode_action_type(a.get("action", ""))
                    opp_action_amount = min(a.get("amount", 0) / MAX_STACK, 1.0)
            features.append(opp_action_type)
            features.append(opp_action_amount)

        cond = np.array(features, dtype=np.float32)

        if len(cond) < COND_DIM:
            cond = np.pad(cond, (0, COND_DIM - len(cond)))
        elif len(cond) > COND_DIM:
            cond = cond[:COND_DIM]

        return cond

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._opponent_actions_this_round = []
        super().receive_round_start_message(round_count, hole_card, seats)

    def receive_game_update_message(self, action, round_state):
        self._opponent_actions_this_round.append(action)
        super().receive_game_update_message(action, round_state)

    def save(self, filepath: str):
        save_data = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "state_dim": self.STATE_DIM,
        }
        if self.denoiser is not None:
            save_data["denoiser_state_dict"] = self.denoiser.state_dict()
            save_data["diffusion_config"] = DIFFUSION_CONFIG
        torch.save(save_data, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = checkpoint["config"]
        if "denoiser_state_dict" in checkpoint:
            config = checkpoint.get("diffusion_config", DIFFUSION_CONFIG)
            self.denoiser = ConditionalDenoiser(
                hand_dim=config["hand_dim"],
                cond_dim=config["cond_dim"],
                hidden_dim=config["hidden_dim"],
            )
            self.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
            self.denoiser.eval()
            self.noise_schedule = NoiseSchedule(
                num_timesteps=config["num_timesteps"],
                beta_start=config["beta_start"],
                beta_end=config["beta_end"],
            )

    def clone(self) -> "PPOPlayerWithDiffusion":
        clone = PPOPlayerWithDiffusion(config=self.config.copy())
        clone.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        if self.denoiser is not None:
            clone.denoiser = ConditionalDenoiser(
                hand_dim=HAND_DIM, cond_dim=COND_DIM,
                hidden_dim=DIFFUSION_CONFIG["hidden_dim"],
            )
            clone.denoiser.load_state_dict(copy.deepcopy(self.denoiser.state_dict()))
            clone.denoiser.eval()
            clone.noise_schedule = self.noise_schedule
        clone.set_training(False)
        return clone


# ============================================================================
# PHASE 3: TRAIN PPO WITH DIFFUSION
# ============================================================================

def phase3_train_augmented_ppo(
    diffusion_path: str = "diffusion_model.pt",
    num_episodes: int = 5000,
    eval_interval: int = 500,
    update_interval: int = 100,
):
    """Trains PPO+Diffusion and a PPO baseline for comparison."""
    print("\n" + "=" * 60)
    print(" PHASE 3: TRAIN PPO WITH DIFFUSION AUGMENTATION")
    print("=" * 60)

    # Augmented agent
    augmented_agent = PPOPlayerWithDiffusion(
        config=PPO_CONFIG.copy(),
        diffusion_model_path=diffusion_path,
    )
    print(f"Augmented agent state_dim: {augmented_agent.STATE_DIM}")

    # Baseline agent (standard PPO, no diffusion)
    baseline_agent = PPOPlayer(config=PPO_CONFIG.copy())
    print(f"Baseline agent state_dim: {baseline_agent.STATE_DIM}")

    opponents_pool = [
        RandomPlayer(),
        HonestPlayer(),
        CallingStationPlayer(),
        TightAggressivePlayer(),
        LooseAggressivePlayer(),
    ]

    aug_win_rates = []
    base_win_rates = []
    eval_episodes = []

    for episode in tqdm(range(num_episodes), desc="Phase 3 Training"):
        opponent = random.choice(opponents_pool)

        # Train augmented agent
        config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="augmented", algorithm=augmented_agent)
        config.register_player(name="opponent", algorithm=opponent)
        start_poker(config, verbose=0)

        # Train baseline agent
        opponent_b = random.choice(opponents_pool)
        config_b = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
        config_b.register_player(name="baseline", algorithm=baseline_agent)
        config_b.register_player(name="opponent", algorithm=opponent_b)
        start_poker(config_b, verbose=0)

        # Periodic PPO updates
        if (episode + 1) % update_interval == 0:
            augmented_agent.update()
            baseline_agent.update()

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            augmented_agent.set_training(False)
            baseline_agent.set_training(False)

            aug_wr = evaluate_agent(augmented_agent, TightAggressivePlayer(), num_games=100)
            base_wr = evaluate_agent(baseline_agent, TightAggressivePlayer(), num_games=100)

            aug_win_rates.append(aug_wr)
            base_win_rates.append(base_wr)
            eval_episodes.append(episode + 1)

            print(f"\n  Episode {episode + 1}: "
                  f"Augmented vs TAG = {aug_wr:.1%}, "
                  f"Baseline vs TAG = {base_wr:.1%}")

            augmented_agent.set_training(True)
            baseline_agent.set_training(True)

    # Save agents
    augmented_agent.save("ppo_diffusion.pt")
    baseline_agent.save("ppo_baseline_comparison.pt")

    # Plot training comparison
    plt.figure(figsize=(12, 6))
    plt.plot(eval_episodes, aug_win_rates, "b-o", linewidth=2, markersize=6, label="PPO + Diffusion")
    plt.plot(eval_episodes, base_win_rates, "r-s", linewidth=2, markersize=6, label="PPO Baseline")
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Win Rate vs TAG", fontsize=12)
    plt.title("PPO + Diffusion vs PPO Baseline: Training Progress", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig("diffusion_training_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nTraining curve saved to diffusion_training_comparison.png")

    # Run full evaluation
    evaluate_diffusion_impact(augmented_agent, baseline_agent)

    return augmented_agent, baseline_agent


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_diffusion_impact(
    augmented_agent,
    baseline_agent,
    num_games: int = 500,
):
    """Compares augmented vs baseline across all opponent types."""
    print("\n" + "=" * 60)
    print(" DIFFUSION IMPACT EVALUATION")
    print("=" * 60)

    augmented_agent.set_training(False)
    baseline_agent.set_training(False)

    opponents = {
        "Random": RandomPlayer(),
        "TAG": TightAggressivePlayer(),
        "LAG": LooseAggressivePlayer(),
        "CallingStation": CallingStationPlayer(),
        "Honest": HonestPlayer(),
    }

    aug_results = {}
    base_results = {}

    for opp_name, opp in opponents.items():
        wr_aug = evaluate_agent(augmented_agent, opp, num_games=num_games)
        wr_base = evaluate_agent(baseline_agent, opp, num_games=num_games)
        aug_results[opp_name] = wr_aug
        base_results[opp_name] = wr_base

        delta = (wr_aug - wr_base) * 100
        print(f"  vs {opp_name:<15}: Augmented={wr_aug:.1%}, Baseline={wr_base:.1%}, Delta={delta:+.1f}pp")

    # Head-to-head
    h2h = evaluate_head_to_head(augmented_agent, baseline_agent, num_games)
    print(f"\n  Head-to-head: Augmented wins {h2h:.1%}")

    # Bar chart comparison
    labels = list(opponents.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, [aug_results[l] for l in labels], width,
                   label="PPO + Diffusion", color="#2196F3")
    bars2 = ax.bar(x + width / 2, [base_results[l] for l in labels], width,
                   label="PPO Baseline", color="#FF9800")

    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Diffusion Opponent Modeling: Impact on Win Rate", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()
    plt.savefig("diffusion_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nEvaluation chart saved to diffusion_evaluation.png")

    return aug_results, base_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diffusion-Augmented Poker RL Pipeline")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1, 2, or 3). 0 = all")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved models only")
    parser.add_argument("--games", type=int, default=10000, help="Number of games for data collection")
    parser.add_argument("--epochs", type=int, default=200, help="Diffusion training epochs")
    parser.add_argument("--episodes", type=int, default=5000, help="PPO training episodes")
    args = parser.parse_args()

    print("")
    print("=" * 58)
    print("  DIFFUSION-AUGMENTED POKER RL PIPELINE")
    print("=" * 58)
    print("  Phase 1: Data Collection")
    print("  Phase 2: Train Diffusion Model (DDPM + FiLM)")
    print("  Phase 3: Train PPO with Opponent Modeling")
    print("=" * 58)
    print("")

    if args.eval:
        # Load and evaluate saved models
        augmented = PPOPlayerWithDiffusion()
        baseline = PPOPlayer()
        augmented.load("ppo_diffusion.pt")
        baseline.load("ppo_baseline_comparison.pt")
        evaluate_diffusion_impact(augmented, baseline)
        return

    if args.phase == 0 or args.phase == 1:
        phase1_collect_data(num_games=args.games)

    if args.phase == 0 or args.phase == 2:
        phase2_train_diffusion(epochs=args.epochs)

    if args.phase == 0 or args.phase == 3:
        phase3_train_augmented_ppo(num_episodes=args.episodes)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETED!")
    print("=" * 60)
    print("""
    Generated files:
       hand_data.pt                    - Collected game data
       diffusion_model.pt              - Trained diffusion model
       diffusion_training.png          - Diffusion loss curve
       ppo_diffusion.pt                - PPO + Diffusion agent
       ppo_baseline_comparison.pt      - PPO baseline agent
       diffusion_training_comparison.png - Training comparison
       diffusion_evaluation.png        - Final evaluation chart
    """)


if __name__ == "__main__":
    main()
