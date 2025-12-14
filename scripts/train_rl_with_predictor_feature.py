from __future__ import annotations

import argparse

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drl_trading.data import build_market_data, download_ohlcv, train_test_split_by_date
from drl_trading.env import TradingConfig, TradingEnv
from drl_trading.predictor import PredictorConfig, build_supervised_sequences, train_lstm_predictor


def add_rolling_lstm_forecast(df, feature_cols, window_size: int, epochs: int = 5):
    """Train LSTM on df[:split] and add an out-of-sample rolling forecast feature.

    The produced column `pred_log_ret` is aligned so that at row t it is computed
    from rows (t-window_size .. t-1), i.e. no lookahead.
    """
    train_df, test_df = train_test_split_by_date(df, train_ratio=0.8)

    model = train_lstm_predictor(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        cfg=PredictorConfig(window_size=window_size, epochs=epochs),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Build sequences for the full df (targets unused)
    xs, _ys = build_supervised_sequences(df, feature_cols, window_size)
    # xs corresponds to rows t in [window_size, len(df)-2]

    preds = []
    bs = 512
    with torch.no_grad():
        for i in range(0, xs.shape[0], bs):
            xb = torch.tensor(xs[i : i + bs], dtype=torch.float32).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)

    out = df.copy()
    out["pred_log_ret"] = 0.0
    out.loc[out.index[window_size : window_size + len(pred)], "pred_log_ret"] = pred
    return out


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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", type=str, default="ppo_trading_with_pred")
    args = ap.parse_args()

    df_ohlcv = download_ohlcv(args.ticker, start=args.start, end=args.end, interval=args.interval)
    md = build_market_data(df_ohlcv)

    df = add_rolling_lstm_forecast(md.df, md.feature_cols, window_size=30, epochs=args.epochs)
    feature_cols = md.feature_cols + ["pred_log_ret"]

    train_df, _test_df = train_test_split_by_date(df, train_ratio=0.8)

    vec_env = DummyVecEnv([make_env(train_df, feature_cols, seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        seed=0,
    )

    model.learn(total_timesteps=args.timesteps)

    model.save(args.out)
    vec_env.save(f"{args.out}_vecnorm.pkl")

    print(f"Saved model to: {args.out}.zip")
    print(f"Saved VecNormalize to: {args.out}_vecnorm.pkl")


if __name__ == "__main__":
    main()
