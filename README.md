# Poker Reinforcement Learning Agent

This project implements and compares Reinforcement Learning (RL) agents for No-Limit Texas Hold'em (simplified) using `PypokerEngine`. The objective is to train agents capable of outperforming heuristic baselines and evolving through Self-Play. The idea originated from an interview with a Mistral AI engineer, who advised me to develop personal machine learning projects. Since I enjoy poker, it became a natural fit.

## Algorithms & Theory

### 1. Q-Learning (Tabular)
The agent uses a Q-table to learn the value of actions in discretized states. The update follows the Bellman equation:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

Where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

### 2. Proximal Policy Optimization (PPO)
An Actor-Critic method using neural networks to approximate policy $\pi_\theta$ and value $V_\phi$.

**Generalized Advantage Estimation (GAE):**

$$
\hat{A}_t^{GAE} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}
$$
With TD error:
$$
\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)
$$

**PPO Clipped Objective:**

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
$$

### 3. Counterfactual Regret Minimization (CFR)
*Currently under development.*

CFR minimizes the cumulative counterfactual regret. The strategy is determined via Regret Matching, where the probability of action $a$ is proportional to its positive cumulative regret $R^+_T(I, a)$:

$$
\sigma_{T+1}(I, a) = \frac{R^+_T(I, a)}{\sum_{a'} R^+_T(I, a')}
$$

## Evaluation and Results

During training, the agents are evaluated against a set of fixed heuristic opponents (Random and Tight-Aggressive). It's currently a 50/50 between PPO and Q-learning, but PPO is just getting started! Its capacity for more complex strategy suggests it will pull ahead with further tuning.

<img width="2382" height="880" alt="comprehensive_evaluation" src="https://github.com/user-attachments/assets/0d497be3-a845-46d3-a157-d6c5422efda8" />


### Performance Analysis

The evaluation results (see `comprehensive_evaluation.png` above) demonstrate a significant performance gap between the agents:

1.  **PPO (Proximal Policy Optimization)**: The clear winner. After stabilizing the training (feature scaling, extensive state vector, normalized rewards), the PPO agent achieves a positive win rate against all heuristic opponents, including the strong TAG (Tight Aggressive) baseline.
2.  **Q-Learning**: Performs adequately against weak opponents (Random) but saturates against stronger strategies due to the discretization of the state space (losing nuance).

#### Self-Play Evolution

The PPO agent was trained using **Fictitious Self-Play** to prevent catastrophic forgetting. The graph below shows the win rate evolution during self-play generations:

<img width="2082" height="730" alt="self_play_evolution" src="https://github.com/user-attachments/assets/d28a4e9f-bcef-4cd7-8359-fcb1d80e3b0b" />


### PPO Configuration

The PPO agent relies on a tuned hyperparameter configuration designed for stability over speed:

*   **Network**: Actor-Critic (2 heads), Hidden Dim `256`
*   **Learning Rate**: `1e-4` (Low to prevent policy collapse)
*   **Batch Size**: `256` (Large batch to reduce variance)
*   **Epochs per update**: `4` (Prevent overfitting to the current batch)
*   **Entropy Coefficient**: `0.02` (Encourages exploration)
*   **GAE Lambda**: `0.90` (Bias-variance trade-off for advantage estimation)
*   **Clip Epsilon**: `0.2` (Usual clipping)

## Usage

**Requirements:** Python 3.8+, `pypokerengine`, `torch`, `numpy`, `matplotlib`.

To run the training and evaluation pipeline:

```bash
python poker_rl_agent.py
```
<img width="1062" height="982" alt="Capture d&#39;écran 2026-02-20 161636" src="https://github.com/user-attachments/assets/4d3e0162-98ab-41d4-9170-34f09586ae36" />

### Work in Progress – CFR implementation, PPO improvement, others features.

Target volatility is shown in green.

---
