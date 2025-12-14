from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class TradingConfig:
    window_size: int = 30
    max_leverage: float = 1.0  # target position in [-max_leverage, +max_leverage]
    commission: float = 0.0005  # 5 bps per trade notional
    slippage: float = 0.0005    # 5 bps per trade notional
    initial_equity: float = 10_000.0
    reward_scale: float = 1.0
    allow_short: bool = True


class TradingEnv(gym.Env):
    """A minimal single-asset trading environment.

    Action: 1D continuous target position fraction of equity.
      - If allow_short=True: action in [-1,1] -> position in [-max_leverage, +max_leverage]
      - Else: action in [-1,1] -> position in [0, max_leverage]

    Observation: (window_size, num_features + 2)
      - features for last window
      - current position (scalar, repeated across window)
      - current equity normalized (scalar, repeated across window)

    Reward: log return of equity between steps (with trading costs).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        config: TradingConfig | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=False).rename(columns={"index": "timestamp"})
        self.feature_cols = feature_cols
        self.cfg = config or TradingConfig()

        if len(self.df) <= self.cfg.window_size + 2:
            raise ValueError("Not enough rows for the requested window_size")

        self._n_features = len(self.feature_cols)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_shape = (self.cfg.window_size, self._n_features + 2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

        self._t = 0
        self._equity = self.cfg.initial_equity
        self._position = 0.0  # fraction of equity

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = self.cfg.window_size
        self._equity = float(self.cfg.initial_equity)
        self._position = 0.0
        obs = self._get_obs()
        info = {"equity": self._equity, "position": self._position}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw = float(action[0])

        if self.cfg.allow_short:
            target_pos = np.clip(raw, -1.0, 1.0) * self.cfg.max_leverage
        else:
            # map [-1,1] -> [0,1]
            target_pos = (np.clip(raw, -1.0, 1.0) + 1.0) / 2.0
            target_pos *= self.cfg.max_leverage

        # Trading cost when changing position (rebalance at time t)
        delta = target_pos - self._position
        trade_notional = abs(delta) * self._equity
        cost = trade_notional * (self.cfg.commission + self.cfg.slippage)

        # Compute next-period return (t -> t+1)
        price_t = float(self.df.loc[self._t, "close"])
        price_tp1 = float(self.df.loc[self._t + 1, "close"])
        ret = (price_tp1 / price_t) - 1.0

        # Apply rebalance, then earn PnL over (t -> t+1) on the NEW position
        equity_before = self._equity
        equity_after_cost = self._equity - cost
        self._position = float(target_pos)
        self._equity = equity_after_cost * (1.0 + self._position * ret)

        # Reward: log equity growth (robust to scale), penalize bankruptcy
        if self._equity <= 0.0:
            reward = -100.0
            terminated = True
        else:
            reward = float(np.log(self._equity / max(1e-12, equity_before))) * self.cfg.reward_scale
            terminated = False

        self._t += 1
        truncated = self._t >= (len(self.df) - 2)

        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "position": self._position,
            "ret": ret,
            "cost": cost,
            "timestamp": self.df.loc[self._t, "timestamp"],
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        w = self.cfg.window_size
        frame = self.df.loc[self._t - w : self._t - 1, self.feature_cols].to_numpy(dtype=np.float32)

        pos_col = np.full((w, 1), self._position, dtype=np.float32)
        eq_col = np.full((w, 1), self._equity / self.cfg.initial_equity, dtype=np.float32)

        obs = np.concatenate([frame, pos_col, eq_col], axis=1)
        return obs
