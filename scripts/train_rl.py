from __future__ import annotations

import argparse

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drl_trading.data import build_market_data, download_ohlcv, train_test_split_by_date
from drl_trading.env import TradingConfig, TradingEnv


def make_env(df, feature_cols, seed: int = 0):
    def _thunk():
        env = TradingEnv(
            df=df,
            feature_cols=feature_cols,
            config=TradingConfig(window_size=30, commission=0.0005, slippage=0.0005, allow_short=True),
        )
        env.reset(seed=seed)
        return env

    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--interval", type=str, default="1d")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="ppo_trading")
    args = ap.parse_args()

    df_ohlcv = download_ohlcv(args.ticker, start=args.start, end=args.end, interval=args.interval)
    md = build_market_data(df_ohlcv)
    train_df, test_df = train_test_split_by_date(md.df, train_ratio=0.8)

    vec_env = DummyVecEnv([make_env(train_df, md.feature_cols, seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        seed=0,
    )

    model.learn(total_timesteps=args.timesteps)

    model.save(args.out)
    vec_env.save(f"{args.out}_vecnorm.pkl")

    print(f"Saved model to: {args.out}.zip")
    print(f"Saved VecNormalize to: {args.out}_vecnorm.pkl")


if __name__ == "__main__":
    main()
