"""
Diffusion Model Utilities
=========================
Card encoding/decoding, noise schedule, and helper functions
for the opponent hand range prediction diffusion model.
"""

import math
import torch
import torch.nn as nn
import numpy as np


# Card constants matching pypokerengine
SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
SUIT_IDS = {v: k for k, v in SUIT_MAP.items()}
RANK_MAP = {str(i): i for i in range(2, 10)}
RANK_MAP.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14})

# Diffusion config
DIFFUSION_CONFIG = {
    "num_timesteps": 100,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "hand_dim": 4,
    "cond_dim": 27,
    "hidden_dim": 128,
}

HAND_DIM = DIFFUSION_CONFIG["hand_dim"]
COND_DIM = DIFFUSION_CONFIG["cond_dim"]


class NoiseSchedule:
    """Precomputes DDPM noise schedule (linear beta schedule)."""

    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def encode_card(rank: int, suit_char: str) -> np.ndarray:
    """Encodes a single card as a 2D normalized vector [rank/14, suit/3]."""
    suit_val = SUIT_MAP.get(suit_char, 0)
    return np.array([rank / 14.0, suit_val / 3.0], dtype=np.float32)


def encode_hand(card1_rank: int, card1_suit: str, card2_rank: int, card2_suit: str) -> np.ndarray:
    """
    Encodes a 2-card hand as a 4D vector.
    Cards are ordered so card1_rank >= card2_rank (canonical ordering).
    """
    if card1_rank < card2_rank or (card1_rank == card2_rank and SUIT_MAP.get(card1_suit, 0) < SUIT_MAP.get(card2_suit, 0)):
        card1_rank, card1_suit, card2_rank, card2_suit = card2_rank, card2_suit, card1_rank, card1_suit

    return np.array([
        card1_rank / 14.0,
        SUIT_MAP.get(card1_suit, 0) / 3.0,
        card2_rank / 14.0,
        SUIT_MAP.get(card2_suit, 0) / 3.0,
    ], dtype=np.float32)


def decode_hand_sample(sample: np.ndarray):
    """
    Decodes a 4D sample back to approximate card values.
    Returns (rank1, suit1, rank2, suit2) as integers.
    """
    rank1 = int(round(sample[0] * 14))
    suit1 = int(round(sample[1] * 3))
    rank2 = int(round(sample[2] * 14))
    suit2 = int(round(sample[3] * 3))

    rank1 = max(2, min(14, rank1))
    rank2 = max(2, min(14, rank2))
    suit1 = max(0, min(3, suit1))
    suit2 = max(0, min(3, suit2))

    return rank1, suit1, rank2, suit2


def parse_card_string(card_str: str):
    """Parses a pypokerengine card string like 'SA' or 'HT' into (rank_int, suit_char)."""
    if len(card_str) == 2:
        suit_char = card_str[0].lower()
        rank_char = card_str[1].upper()
    else:
        suit_char = card_str[0].lower()
        rank_char = card_str[1:].upper()

    rank_int = RANK_MAP.get(rank_char, 0)
    if rank_int == 0:
        try:
            rank_int = int(rank_char)
        except ValueError:
            rank_int = 2

    return rank_int, suit_char


def encode_action_type(action: str) -> float:
    """Encodes a poker action type to a float value."""
    action_encoding = {
        "fold": 0.0,
        "call": 0.33,
        "raise": 0.67,
        "all_in": 1.0,
        "ante": 0.1,
        "small_blind": 0.15,
        "big_blind": 0.2,
    }
    return action_encoding.get(action.lower(), 0.0)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding from DDPM."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
