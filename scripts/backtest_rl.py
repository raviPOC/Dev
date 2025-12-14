from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drl_trading.backtest import run_backtest
from drl_trading.data import build_market_data, download_ohlcv, train_test_split_by_date
from drl_trading.env import TradingConfig, TradingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--interval", type=str, default="1d")
    ap.add_argument("--model", type=str, default="ppo_trading")
    args = ap.parse_args()

    df_ohlcv = download_ohlcv(args.ticker, start=args.start, end=args.end, interval=args.interval)
    md = build_market_data(df_ohlcv)
    train_df, test_df = train_test_split_by_date(md.df, train_ratio=0.8)

    # Build eval env (single, non-vector) and load normalization stats
    eval_env = TradingEnv(
        df=test_df,
        feature_cols=md.feature_cols,
        config=TradingConfig(window_size=30, commission=0.0005, slippage=0.0005, allow_short=True),
    )

    # SB3 models trained with VecNormalize expect the same normalization at inference.
    vec = DummyVecEnv([lambda: eval_env])
    vec = VecNormalize.load(f"{args.model}_vecnorm.pkl", vec)
    vec.training = False
    vec.norm_reward = False

    model = PPO.load(args.model, env=vec)

    # run_backtest expects a non-vector env, so we call predict manually through vec.
    # simplest: step using vec to preserve normalization.
    obs = vec.reset()
    equities = [eval_env.cfg.initial_equity]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = vec.step(action)
        done = bool(terminated[0] or truncated[0])
        equities.append(float(info[0].get("equity")))

    eq = np.asarray(equities)
    plt.figure(figsize=(10, 4))
    plt.plot(eq)
    plt.title(f"Equity curve: {args.ticker}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

    total_return = (eq[-1] / eq[0]) - 1.0
    print(f"Final equity: {eq[-1]:.2f} | Total return: {total_return*100:.2f}%")


if __name__ == "__main__":
    main()
