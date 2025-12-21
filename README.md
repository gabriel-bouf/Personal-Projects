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

The agents are evaluated against a set of fixed heuristic opponents (Random, Tight-Aggressive, Honest).

<img width="2382" height="880" alt="comprehensive_evaluation" src="https://github.com/user-attachments/assets/de6c383b-e4d2-4af7-9ea3-d21b796ffc85" />


### Performance Analysis

The graph shows how each agent performs:

1.  **Q-Learning**: Works well against Random players, but struggles with more sophisticated strategies (TAG, Honest). The tabular approach can't handle the complexity of poker states well enough.
2.  **PPO**: Much stronger performance with only 2000 training iterations. The neural network architecture allows it to learn more nuanced strategies across all opponent types.
3.  **Self-Play Impact**: Training against itself (instead of just fixed opponents) helped the PPO agent discover strategies beyond what the heuristics could teach. 

## Usage

**Requirements:** Python 3.8+, `pypokerengine`, `torch`, `numpy`, `matplotlib`.

To run the training and evaluation pipeline:

```bash
python poker_rl_agent.py
```

### Work in Progress – CFR implementation, PPO improvement other features.



---
## Project 3: Skew of volatility

The idea is to use the historical time series of the excess return index assuming interest rates are equal to fed fund to imply what 
implied volatility would on average over the period made the P&L of a delta hedged call option equal to zero.

The project begins by simulating delta hedging using stock prices obtained from yfinance. The simulation takes into account transaction costs, risk-free rates, and daily hedging frequency. It then calculates the P&L as a function of strike prices and volatility, allowing the identification of the break-even volatility. The BEV surface reveals that low strike prices lead to higher BEV. This helps in understanding how volatility and option strikes influence the profitability of delta hedging strategies.

### Delta Hedging

The first step in this project is to simulate delta hedging. I used the `yfinance` library to retrieve Apple stock prices from January 1, 2024, to August 1, 2024, to simulate a delta hedging strategy. By plotting the P&L of the hedging strategy as a function of the strike price and the volatility used in the Black-Scholes model, it is interesting to observe that the P&L cannot be positive if the strike price is too low (deep in the money). In the following example, the initial spot price was $184, and the spot price at maturity was $221. We can see that below K=150, the P&L remains negative, meaning the Break-Even Volatility does not exist.

The simulation was run with the following parameters:
- **Ticker AAPL** : Spot= $184
- **Commission rate:** 0.03%
- **Bid-ask spread:** 0.02%
- **Risk-free rate:** 4.3% (Fed Funds rate assumption)
- **Borrowing interest rate:** 4.3%
- **Delta hedging frequency:** Daily

![skeww 3d k150](https://github.com/user-attachments/assets/4208010f-38e7-4b74-9812-c0b6880470c9)

After finding the break-even volatility based on the P&L simulation, I have obtained the following BEV surface :

![bev4](https://github.com/user-attachments/assets/0f867512-6fd6-439f-8e83-7829a44f0235)


Based on the provided BEV (Break-Even Volatility) surface, we can observe the following:

- Low strikes lead to higher BEV.
- The BEV is near 0 for 450-480 days maturity and for deep OTM options.

---
## Project 2: Portfolio Optimization with the Markowitz Model and Mistral API

This project implements portfolio optimization using the Markowitz model. The goal is to build a portfolio that maximizes Sharpe ratio, using financial data sourced from Yahoo Finance and ```DataFrame``` with ```pandas``` framework.

> **Note** : An API key is required to use Mistral AI. If you don’t have one, stocks are selected randomly from a customizable list.

### Project Description
Here are the key steps of the project:
1. Stock selection for the portfolio using Mistral AI based on a defined number of assets and time period.
2. Calculation of average returns and the covariance matrix.
3. Evaluation of expected return and risk based on weights.
4. Sharpe ratio optimization. 
5. Removal of marginal assets (<0.1% of the portfolio) and generation of random portfolios for comparison.
6. Visualization and backtest to evaluate performance.
    
### Results

With 3 assets:
![p2_5a_puiss08](https://github.com/user-attachments/assets/23492b3a-4ac7-4681-8a03-375796ebb58b)  
*The performance from this asset selection remains modest.*

After modifying the optimization function, the portfolio moves along the efficient frontier:
![p2_5action_puissance01](https://github.com/user-attachments/assets/d3daf5f4-415f-40ca-a271-bc326b1c00ed)  
*This variant favors less risky positions.*

A more diversified portfolio shows a 15% return for a 6.5% risk: 
![Figure_1_projet2](https://github.com/user-attachments/assets/cc11c918-98fd-4b63-8c93-0d36efb651c8)

### Mistral & Backtest
Mistral AI is used for ticker selection. For example, with 5 assets:
```plaintext
Mistral proposition:
['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
Optimal set:
[('AAPL', 0.303), ('GOOGL', 0.139), ('NVDA', 0.557)]
Deleted tickers:
['MSFT', 'AMZN']
```

Return of the portfolio on the tested period:
 17.295 %
Expected return of the portfolio over the period: 45.2%

*Past performance is not indicative of future results* at its best.

![backt3](https://github.com/user-attachments/assets/07167c7a-105a-4e9d-812c-fdfd21c1ab86)  
*Portfolio curve in black.*

You can modify the prompt to include cryptocurrencies and even a hundred tickers to find optimal combinations. We round weights to three significant digits, which eliminates near-zero values. However, Mistral tends to hallucinate many tickers in that case, and only about 20 are retained.

After adapting the prompt to include crypto, here is the result for 15 tickers:

```plaintext
Mistral proposition:
 ['MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'V', 'BABA', 'BNP.PA', 'NKE', 'UNH', 'JNJ', 'PG', 'KO', 'VZ', 'BTC-USD']
Optimal set:
 [('MSFT', 0.003), ('AMZN', 0.014), ('NVDA', 0.035), ('V', 0.038), ('BABA', 0.579), ('UNH', 0.157), ('JNJ', 0.004), ('KO', 0.069), ('BTC-USD', 0.102)]
Deleted tickers:
 ['GOOGL', 'TSLA', 'BNP.PA', 'NKE', 'PG', 'VZ']
```
This portfolio is supposed to have a daily return of 0.1922% and a daily risk of 0.6308% based on 2024-01-01 to 2024-08-01.


Return of the portfolio on the tested period : 
 4.834 %
Expected return of the portfolio over the period : 26.39 %

![back4](https://github.com/user-attachments/assets/16130d77-e207-4040-9588-70c072856db7)

The plateaus on the curves correspond to differences in trading hours across markets, with crypto trading continuously while traditional exchanges close.

This portfolio is supposed to have a daily return of 0.44% and a daily risk of 2.1867% based on 2024-01-01 to 2024-08-01.

Return of the portfolio on the tested period :
 17.295 %
Expected return of the portfolio over the period : 45.233 %

---

## Project 1: Building a S&P 500 Index with 5% Daily Risk Control

The goal is to dynamically adjust the exposure of a portfolio tracking the S&P 500 in order to cap the daily volatility at 5%. This project uses data from the `yfinance` API.

### Project Description

Key steps of the algorithm: 
1. Historical data retrieval using `yfinance`.  
2. Computation of returns and volatility over a rolling window (e.g., 20 days), using exponential weight averaging to calculate the recent realized volatility.
3. Return adjustment to meet the target risk using the formula:  
   $`
   \text{Index Returns} = E \times \text{Returns}  +(1-E) \times \text{Returns} 
    `$
   with
   $`
   E = min(1,\left(\frac{\text{Target Volatility}}{\text{Realized Volatility}}\right)) `$
  

### Results
Compared to the [official model](https://www.spglobal.com/spdji/en/indices/multi-asset/sp-500-daily-risk-control-5-index/#overview), my implementation shows a ~3% deviation over one year.

**Reference model :**  
![image](https://github.com/user-attachments/assets/ab26c652-308f-4ba0-8276-50e3b983942c)  

**My model :**  
![SP500_le_mien](https://github.com/user-attachments/assets/96cc1b53-a108-4230-ae73-cc5de319ec41)

Including the risk-free component (in green):

![sp perf](https://github.com/user-attachments/assets/2bb5cee0-2ef3-40b2-9648-8f2aaf2e38eb)


**Comparison of Volatilities :** 

![SP vols](https://github.com/user-attachments/assets/ab504701-03fb-42b6-b8b4-ed92a2a29e5e)

Target volatility is shown in green.

---
