"""Slippage augmentation for RL training data.

Takes historical episodes (price/volume candles + trade actions) and produces
augmented copies with realistic slippage, partial fills, and delayed confirmations.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("ml.augment")

# Slippage profiles: (entry_min%, entry_max%, exit_min%, exit_max%)
SLIPPAGE_PROFILES = {
    "ideal":      (0.005, 0.03, 0.01, 0.05),
    "typical":    (0.03,  0.10, 0.05, 0.15),
    "congested":  (0.08,  0.25, 0.12, 0.30),
    "sandwiched": (0.20,  0.50, 0.20, 0.40),
}


@dataclass
class Episode:
    """A single trading episode: candles + a trade action."""
    candles: list[dict[str, float]]  # each: {open, high, low, close, volume}
    action: dict[str, Any]           # {type: "buy"|"sell", candle_idx, amount_usd}
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentedEpisode:
    """An augmented episode with slippage applied."""
    candles: list[dict[str, float]]
    action: dict[str, Any]
    slippage_profile: str
    entry_slippage_pct: float
    exit_slippage_pct: float
    fill_fraction: float
    confirmation_delay_candles: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SlippageAugmenter:
    """Generates augmented training episodes with realistic Solana DEX slippage.

    Uses LogNormal distributions for slippage sampling to model the heavy right
    tail observed in real meme coin trading (occasional extreme slippage events).
    """

    def __init__(
        self,
        variants_per_episode: int = 5,
        partial_fill_range: tuple[float, float] = (0.80, 0.95),
        confirmation_delay_range: tuple[int, int] = (1, 3),
        seed: int | None = None,
    ) -> None:
        self.variants_per_episode = variants_per_episode
        self.partial_fill_range = partial_fill_range
        self.confirmation_delay_range = confirmation_delay_range
        self.rng = np.random.default_rng(seed)

    def _sample_lognormal_slippage(self, lo: float, hi: float) -> float:
        """Sample slippage from a LogNormal distribution clamped to [lo, hi].

        We set the LogNormal mu/sigma so the median falls at the midpoint of
        the range and the distribution has a realistic right tail.
        """
        mid = (lo + hi) / 2
        mu = np.log(max(mid, 1e-6))
        sigma = 0.4
        sample = float(self.rng.lognormal(mu, sigma))
        return max(lo, min(sample, hi))

    def _pick_profile(self) -> str:
        """Pick a slippage profile with weighted probability."""
        profiles = list(SLIPPAGE_PROFILES.keys())
        weights = [0.15, 0.45, 0.25, 0.15]  # ideal, typical, congested, sandwiched
        return self.rng.choice(profiles, p=weights)

    def augment_episode(self, episode: Episode) -> list[AugmentedEpisode]:
        """Generate augmented variants of a single episode."""
        augmented: list[AugmentedEpisode] = []

        for _ in range(self.variants_per_episode):
            profile_name = self._pick_profile()
            entry_lo, entry_hi, exit_lo, exit_hi = SLIPPAGE_PROFILES[profile_name]

            entry_slip = self._sample_lognormal_slippage(entry_lo, entry_hi)
            exit_slip = self._sample_lognormal_slippage(exit_lo, exit_hi)

            # Partial fill: 80-95% of intended amount
            fill_frac = float(self.rng.uniform(*self.partial_fill_range))

            # Delayed confirmation: shift fill price 1-3 candles forward
            delay = int(self.rng.integers(
                self.confirmation_delay_range[0],
                self.confirmation_delay_range[1] + 1,
            ))

            # Deep copy candles and apply slippage to the action
            new_candles = copy.deepcopy(episode.candles)
            new_action = copy.deepcopy(episode.action)

            action_idx = new_action.get("candle_idx", 0)
            action_type = new_action.get("type", "buy")

            # Apply confirmation delay: shift fill to a later candle
            delayed_idx = min(action_idx + delay, len(new_candles) - 1)

            # Get the fill price from the delayed candle
            if delayed_idx < len(new_candles):
                ref_candle = new_candles[delayed_idx]
                base_price = ref_candle.get("close", ref_candle.get("open", 0))
            else:
                base_price = new_action.get("price", 0)

            # Apply slippage to fill price
            if action_type == "buy":
                fill_price = base_price * (1 + entry_slip)
            else:
                fill_price = base_price * (1 - exit_slip)

            # Apply partial fill
            original_amount = new_action.get("amount_usd", 0)
            new_action["amount_usd"] = original_amount * fill_frac
            new_action["fill_price"] = fill_price
            new_action["fill_candle_idx"] = delayed_idx

            augmented.append(AugmentedEpisode(
                candles=new_candles,
                action=new_action,
                slippage_profile=profile_name,
                entry_slippage_pct=entry_slip * 100,
                exit_slippage_pct=exit_slip * 100,
                fill_fraction=fill_frac,
                confirmation_delay_candles=delay,
                metadata={
                    **episode.metadata,
                    "original_amount_usd": original_amount,
                    "original_candle_idx": action_idx,
                },
            ))

        return augmented

    def augment_batch(self, episodes: list[Episode]) -> list[AugmentedEpisode]:
        """Augment a batch of episodes."""
        all_augmented: list[AugmentedEpisode] = []
        for ep in episodes:
            all_augmented.extend(self.augment_episode(ep))
        log.info(
            "Augmented %d episodes -> %d variants",
            len(episodes), len(all_augmented),
        )
        return all_augmented
