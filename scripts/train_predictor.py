from __future__ import annotations

import argparse

import numpy as np

from drl_trading.data import build_market_data, download_ohlcv, train_test_split_by_date
from drl_trading.predictor import PredictorConfig, train_lstm_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--interval", type=str, default="1d")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=str, default="lstm_predictor.pt")
    args = ap.parse_args()

    df_ohlcv = download_ohlcv(args.ticker, start=args.start, end=args.end, interval=args.interval)
    md = build_market_data(df_ohlcv)
    train_df, test_df = train_test_split_by_date(md.df, train_ratio=0.8)

    model = train_lstm_predictor(
        train_df=train_df,
        test_df=test_df,
        feature_cols=md.feature_cols,
        cfg=PredictorConfig(window_size=30, epochs=args.epochs),
    )

    import torch

    torch.save(model.state_dict(), args.out)
    print(f"Saved predictor weights to: {args.out}")


if __name__ == "__main__":
    main()
