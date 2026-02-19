"""MemeExitEnv — Gymnasium environment for meme coin exit timing.

The agent decides when and how much to sell from a position it entered
at the start of a price sequence.  Actions: hold, exit 25%, exit 50%,
exit 100%.  Observations encode price dynamics, volume, and unrealized PnL.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class MemeExitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sequences: list[list[dict]],
        augmenter=None,
        max_hold_minutes: int = 120,
    ) -> None:
        super().__init__()
        self.sequences = sequences
        self.augmenter = augmenter
        self.max_hold_minutes = max_hold_minutes

        # 0=hold, 1=exit25%, 2=exit50%, 3=exit100%
        self.action_space = gym.spaces.Discrete(4)

        # [price_ratio, time_held_norm, vol_ratio, buy_sell_ratio,
        #  unrealized_pnl, momentum_5m]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -2.0, -1.0], dtype=np.float32),
            high=np.array([20.0, 1.0, 10.0, 1.0, 20.0, 5.0], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.sequences))
        self.seq = self.sequences[idx]
        self.step_idx = 0
        self.entry_price = float(self.seq[0].get("price", 1.0))
        if self.entry_price <= 0:
            self.entry_price = 1.0
        self.position_pct = 1.0
        self.realized_pnl = 0.0
        self.time_held = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        current = self.seq[min(self.step_idx, len(self.seq) - 1)]
        price = float(current.get("price", self.entry_price))
        if price <= 0:
            price = self.entry_price

        # Small time penalty encourages decisive action
        time_penalty = 0.0005 * self.position_pct
        reward = -time_penalty

        exit_fractions = {0: 0.0, 1: 0.25, 2: 0.50, 3: 1.0}
        exit_frac = exit_fractions[action]

        if exit_frac > 0 and self.position_pct > 0.01:
            # Realistic slippage
            if self.augmenter and hasattr(self.augmenter, "sample_exit_slippage"):
                slippage = self.augmenter.sample_exit_slippage()
            else:
                slippage = np.random.uniform(0.03, 0.12)
            fill_price = price * (1.0 - slippage)
            sold_pct = min(exit_frac * self.position_pct, self.position_pct)
            pnl = (fill_price / self.entry_price - 1.0) * sold_pct
            self.realized_pnl += pnl
            reward += pnl
            self.position_pct = max(0.0, self.position_pct - sold_pct)

        self.step_idx += 1
        self.time_held += 1

        terminated = (
            self.position_pct <= 0.01
            or self.step_idx >= len(self.seq)
            or self.time_held >= self.max_hold_minutes
        )

        # Force-close any remaining position on termination
        if terminated and self.position_pct > 0.01:
            slippage = np.random.uniform(0.05, 0.15)
            fill_price = price * (1.0 - slippage)
            pnl = (fill_price / self.entry_price - 1.0) * self.position_pct
            self.realized_pnl += pnl
            reward += pnl
            self.position_pct = 0.0

        return (
            self._get_obs(),
            float(reward),
            terminated,
            False,
            {"realized_pnl": self.realized_pnl},
        )

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        current = self.seq[min(self.step_idx, len(self.seq) - 1)]
        lookback = self.seq[max(0, self.step_idx - 5)]
        price = float(current.get("price", self.entry_price))
        prev_price = float(lookback.get("price", price))
        if price <= 0:
            price = self.entry_price
        if prev_price <= 0:
            prev_price = price
        momentum = (price / prev_price - 1.0) if prev_price > 0 else 0.0

        return np.array(
            [
                np.clip(price / self.entry_price, 0, 20),
                self.time_held / self.max_hold_minutes,
                np.clip(float(current.get("vol_ratio", 1.0)), 0, 10),
                np.clip(float(current.get("buy_sell_ratio", 0.5)), 0, 1),
                np.clip(self.realized_pnl, -2, 20),
                np.clip(momentum, -1, 5),
            ],
            dtype=np.float32,
        )
