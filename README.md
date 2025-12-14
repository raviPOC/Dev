## DRL day-trading + price prediction (minimal example)

This repo contains a **small, educational** example of:

- A **custom Gymnasium trading environment** for a single stock/ETF (position sizing, transaction costs)
- Training a **Deep Reinforcement Learning (DRL)** agent with **PPO (Stable-Baselines3)**
- A separate **LSTM price/return predictor** (supervised learning baseline)

This is **not financial advice** and is not production trading code.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train a DRL agent (PPO)

```bash
python3 scripts/train_rl.py --ticker SPY --start 2015-01-01 --timesteps 200000 --out ppo_trading
```

This writes:
- `ppo_trading.zip` (policy)
- `ppo_trading_vecnorm.pkl` (observation normalization stats)

### Backtest the trained agent

```bash
python3 scripts/backtest_rl.py --ticker SPY --start 2015-01-01 --model ppo_trading
```

### Train a price predictor (LSTM)

This predicts **next-day log return** using the same feature window.

```bash
python3 scripts/train_predictor.py --ticker SPY --start 2015-01-01 --epochs 10 --out lstm_predictor.pt
```

### (Optional) Train DRL while feeding a predictor forecast as a feature

This trains an LSTM on the training split, generates a rolling out-of-sample forecast feature (`pred_log_ret`),
then trains PPO with that forecast included in the observation features.

```bash
python3 scripts/train_rl_with_predictor_feature.py --ticker SPY --start 2015-01-01 --epochs 5 --timesteps 200000 --out ppo_trading_with_pred
```

### How this DRL setup works (high-level)

- **State / observation**: last `window_size` rows of engineered features (returns, RSI, moving-average ratios, volatility, volume change) + current position + equity.
- **Action**: a continuous number in [-1, 1] interpreted as a *target position* (short/flat/long) up to `max_leverage`.
- **Reward**: stepwise log equity growth, including commission + slippage when the position changes.

Key code is in:
- `drl_trading/env.py` (the environment)
- `scripts/train_rl.py` (PPO training)
- `scripts/backtest_rl.py` (evaluation)
- `drl_trading/predictor.py` + `scripts/train_predictor.py` (LSTM predictor)
