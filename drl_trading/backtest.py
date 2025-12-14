from __future__ import annotations

import numpy as np


def run_backtest(env, model, deterministic: bool = True):
    """Runs a single backtest episode and returns equity curve + actions."""
    obs, info = env.reset()
    equities = [info.get("equity", np.nan)]
    positions = [info.get("position", np.nan)]
    actions = []
    rewards = []

    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        actions.append(float(np.asarray(action).reshape(-1)[0]))
        rewards.append(float(reward))
        equities.append(float(info.get("equity", np.nan)))
        positions.append(float(info.get("position", np.nan)))

    return {
        "equity": np.asarray(equities, dtype=float),
        "position": np.asarray(positions, dtype=float),
        "action": np.asarray(actions, dtype=float),
        "reward": np.asarray(rewards, dtype=float),
    }
