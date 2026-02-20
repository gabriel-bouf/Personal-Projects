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


### 4. Conditional Diffusion Model (DDPM) for Opponent Modeling

The PPO agent plays "blind", it makes decisions based only on its own cards and the game state, so I trained a conditional DDPM (Denoising Diffusion Probabilistic Model) that learns to estimate the distribution of the opponent's hole cards given the observable game state to see how it does.

The idea: collect thousands of games between bots where we record both players' hands along with the full context (street, community cards, pot, betting actions). Then train a generative model conditioned on that context to produce plausible opponent hands.

**Conditioning vector** $c$ (27D): encodes the current street, community cards, pot size, and action history.

**Target** $x_0$ (4D): the opponent's hand encoded as $[\text{rank}_1/14, \text{suit}_1/3, \text{rank}_2/14, \text{suit}_2/3]$.

**Forward process** — progressively adds Gaussian noise over $T=100$ timesteps (linear schedule, $\beta \in [10^{-4}, 0.02]$):

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Training objective** — the denoiser $\epsilon_\theta$ predicts the noise added at each step:

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right]
$$

The denoiser is an MLP with FiLM (Feature-wise Linear Modulation) conditioning: the timestep and game state embeddings modulate each residual block via learned scale/shift parameters $(\gamma, \beta)$. This lets the model adapt its predictions depending on the game context rather than treating all situations the same.

**Integrating with PPO:** at inference, the diffusion model generates 32 opponent hand samples given the current game state. From these samples, I extract 6 statistical features (mean hand strength, std, percentiles, probability of strong/weak hands) that are concatenated to the PPO state vector. The training pipeline runs in 3 phases:
1. Collect game data (games between bots, capturing full hand info)
2. Train the diffusion model on this data
3. Train PPO with the augmented state vector

The improvement is modest (a few percentage points in win rate) but the PPO+Diffusion agent shows better stability during training, maintaining consistent performance where the baseline tends to degrade over longer training runs.


## Evaluation and Results

During training, the agents are evaluated against a set of fixed heuristic opponents (Random and Tight-Aggressive). It's currently a 50/50 between PPO and Q-learning, but PPO is just getting started! Its capacity for more complex strategy suggests it will pull ahead with further tuning.

<img width="2382" height="880" alt="comprehensive_evaluation" src="https://github.com/user-attachments/assets/0d497be3-a845-46d3-a157-d6c5422efda8" />


### Performance Analysis

The evaluation results (see `comprehensive_evaluation.png` above) demonstrate a significant performance gap between the agents:

1.  **PPO (Proximal Policy Optimization)**: The clear winner. After stabilizing the training (feature scaling, extensive state vector, normalized rewards), the PPO agent achieves a positive win rate against all heuristic opponents, including the strong TAG (Tight Aggressive) baseline.
2.  **Q-Learning**: Performs adequately against weak opponents (Random) but saturates against stronger strategies due to the discretization of the state space (losing nuance).

#### Self-Play Evolution

The PPO agent was also trained using **Fictitious Self-Play**, where each generation trains against the previous one. The process is still unstable: generation 4 collapses to ~29% win rate against its predecessor before bouncing back to ~80% at generation 5. Performance vs Random stays high throughout though, so the agent doesn't fully forget how to play, it just oscillates in terms of strategy:
<img width="2082" height="730" alt="self_play_evolution" src="https://github.com/user-attachments/assets/d28a4e9f-bcef-4cd7-8359-fcb1d80e3b0b" />


#### Diffusion Opponent Modeling: Impact on Win Rate

After integrating the diffusion model, I compared PPO+Diffusion against the PPO baseline across 5 different opponent types. The results are encouraging but honest — the diffusion model doesn't revolutionize the agent, but it gives a slight edge in some matchups (LAG +3%, Honest +4%) and more consistent behavior overall.

<img width="1782" height="884" alt="diffusion_evaluation" src="https://github.com/user-attachments/assets/8e04937c-9470-473e-aec9-0ae0bea14946" />


The training comparison below shows the win rate against TAG over training episodes. The PPO+Diffusion agent maintains a more stable ~60% win rate on longer training runs, while the baseline tends to degrade after ~2000 episodes.

<img width="1518" height="824" alt="diffusion_training_comparison" src="https://github.com/user-attachments/assets/1264c606-2981-46d0-a1d6-8521cefce3c2" />


### PPO Configuration

The PPO agent relies on a tuned hyperparameter configuration designed for stability over speed:

*   **Network**: Actor-Critic (2 heads), Hidden Dim `256`
*   **Learning Rate**: `3e-4` (Faster with clean reward signal)
*   **Batch Size**: `64` (Smaller batches, better for heads-up)
*   **Epochs per update**: `4` (Prevent overfitting to the current batch)
*   **Entropy Coefficient**: `0.05` (More exploration)
*   **GAE Lambda**: `0.95` (Less bias than 0.90)
*   **Clip Epsilon**: `0.2` (Usual clipping)

## Usage

**Requirements:** Python 3.8+, `pypokerengine`, `torch`, `numpy`, `matplotlib`.

To run the training and evaluation pipeline:

```bash
python poker_rl_agent.py
```

To run the 3-phase diffusion pipeline:
```bash
python train_diffusion_pipeline.py --phase 1  # collect game data
python train_diffusion_pipeline.py --phase 2  # train diffusion model
python train_diffusion_pipeline.py --phase 3  # train PPO with diffusion
```

I also built a small web interface (Flask) to play against the trained agents or watch bot vs bot games. It supports all agent types including PPO+Diffusion.

```bash
python web_arena/app.py
```

<img width="1062" height="982" alt="Capture d&#39;écran 2026-02-20 161636" src="https://github.com/user-attachments/assets/4d3e0162-98ab-41d4-9170-34f09586ae36" />

### Work in Progress – CFR implementation, extending the diffusion model to handle multi-street conditioning, exploring other architectures.


---
