"""
Conditional Diffusion Model for Opponent Hand Range Prediction
==============================================================
A DDPM (Denoising Diffusion Probabilistic Model) that predicts the distribution
of opponent hole cards conditioned on observable game state and betting actions.

Architecture: MLP with FiLM (Feature-wise Linear Modulation) conditioning.
Target: 4D continuous vector [card1_rank/14, card1_suit/3, card2_rank/14, card2_suit/3]
Condition: 27D vector (street, community cards, pot, opponent actions, etc.)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_utils import (
    NoiseSchedule,
    SinusoidalPositionEmbedding,
    DIFFUSION_CONFIG,
    HAND_DIM,
    COND_DIM,
    decode_hand_sample,
)


class FiLMResBlock(nn.Module):
    """Residual block with FiLM (Feature-wise Linear Modulation) conditioning."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.film = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.net(x)
        h = gamma * h + beta
        return x + h


class ConditionalDenoiser(nn.Module):
    """
    Conditional noise predictor for DDPM.
    Predicts epsilon (noise) given noisy hand x_t, timestep t, and conditioning c.
    """

    def __init__(self, hand_dim=HAND_DIM, cond_dim=COND_DIM, hidden_dim=128, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(hand_dim, hidden_dim)

        self.block1 = FiLMResBlock(hidden_dim)
        self.block2 = FiLMResBlock(hidden_dim)
        self.block3 = FiLMResBlock(hidden_dim)

        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hand_dim),
        )

    def forward(self, x_t, t, cond):
        """
        x_t: (B, 4) noisy hand
        t: (B,) timestep indices
        cond: (B, 27) conditioning vector
        Returns: (B, 4) predicted noise
        """
        t_emb = self.time_mlp(t)
        c_emb = self.cond_encoder(cond)
        film_cond = t_emb + c_emb

        h = self.input_proj(x_t)
        h = self.block1(h, film_cond)
        h = self.block2(h, film_cond)
        h = self.block3(h, film_cond)

        return self.output_proj(h)


def diffusion_loss(denoiser, x_0, cond, noise_schedule):
    """
    Standard DDPM epsilon-prediction training loss.
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    t = torch.randint(0, noise_schedule.num_timesteps, (batch_size,), device=device)
    epsilon = torch.randn_like(x_0)

    alpha_bar = noise_schedule.alpha_bar[t].unsqueeze(-1)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon

    epsilon_pred = denoiser(x_t, t, cond)
    return F.mse_loss(epsilon_pred, epsilon)


@torch.no_grad()
def ddpm_sample(denoiser, cond, noise_schedule, num_samples=32):
    """Full DDPM reverse process sampling (T steps)."""
    device = next(denoiser.parameters()).device
    cond = cond.expand(num_samples, -1).to(device)

    x = torch.randn(num_samples, HAND_DIM, device=device)

    for t in reversed(range(noise_schedule.num_timesteps)):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        epsilon_pred = denoiser(x, t_batch, cond)

        alpha = noise_schedule.alpha[t]
        alpha_bar = noise_schedule.alpha_bar[t]
        beta = noise_schedule.beta[t]

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * epsilon_pred
        )

        if t > 0:
            sigma = torch.sqrt(beta)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

    x[:, 0] = x[:, 0].clamp(2 / 14, 1.0)
    x[:, 1] = x[:, 1].clamp(0.0, 1.0)
    x[:, 2] = x[:, 2].clamp(2 / 14, 1.0)
    x[:, 3] = x[:, 3].clamp(0.0, 1.0)

    return x


@torch.no_grad()
def ddim_sample(denoiser, cond, noise_schedule, num_samples=32, ddim_steps=20):
    """Fast DDIM sampling (fewer steps, deterministic)."""
    device = next(denoiser.parameters()).device
    cond = cond.expand(num_samples, -1).to(device)

    step_size = max(1, noise_schedule.num_timesteps // ddim_steps)
    timesteps = list(range(0, noise_schedule.num_timesteps, step_size))[::-1]

    x = torch.randn(num_samples, HAND_DIM, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        epsilon_pred = denoiser(x, t_batch, cond)

        alpha_bar_t = noise_schedule.alpha_bar[t]

        if i < len(timesteps) - 1:
            alpha_bar_prev = noise_schedule.alpha_bar[timesteps[i + 1]]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(0, 1)

        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * epsilon_pred

    x[:, 0] = x[:, 0].clamp(2 / 14, 1.0)
    x[:, 1] = x[:, 1].clamp(0.0, 1.0)
    x[:, 2] = x[:, 2].clamp(2 / 14, 1.0)
    x[:, 3] = x[:, 3].clamp(0.0, 1.0)

    return x


def estimate_hand_strength_from_ranks(rank1, rank2, suit1, suit2):
    """
    Fast heuristic hand strength estimate from card values.
    Returns a value in [0, 1].
    """
    high = max(rank1, rank2)
    low = min(rank1, rank2)
    paired = rank1 == rank2
    suited = suit1 == suit2
    gap = high - low

    strength = (high + low) / 28.0

    if paired:
        strength += 0.25 + (high / 14.0) * 0.15
    if suited:
        strength += 0.05
    if gap <= 2 and not paired:
        strength += 0.03

    return min(1.0, max(0.0, strength))


def compute_opponent_features(samples, community_cards_ranks=None):
    """
    Extracts 6 statistical features from diffusion-generated opponent hand samples.

    Args:
        samples: (N, 4) tensor of generated opponent hands
        community_cards_ranks: optional list of (rank, suit) tuples for community cards

    Returns:
        np.ndarray of 6 features:
          [mean_strength, std_strength, p25_strength, p75_strength,
           prob_strong (>0.7), prob_weak (<0.3)]
    """
    samples_np = samples.cpu().numpy()
    strengths = []

    for s in samples_np:
        r1, s1, r2, s2 = decode_hand_sample(s)
        strength = estimate_hand_strength_from_ranks(r1, r2, s1, s2)
        strengths.append(strength)

    strengths = np.array(strengths, dtype=np.float32)

    if len(strengths) == 0:
        return np.zeros(6, dtype=np.float32)

    return np.array([
        np.mean(strengths),
        np.std(strengths),
        np.percentile(strengths, 25),
        np.percentile(strengths, 75),
        np.mean(strengths > 0.7),
        np.mean(strengths < 0.3),
    ], dtype=np.float32)
