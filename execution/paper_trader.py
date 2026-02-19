"""Paper trading mode — simulates fills using live DexScreener prices."""

from __future__ import annotations

import logging
import random

import requests

import config as cfg
from execution.position_manager import PositionManager

log = logging.getLogger("execution.paper_trader")

DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"


def _fetch_price(token_mint: str) -> float | None:
    """Fetch current price from DexScreener."""
    try:
        resp = requests.get(
            DEXSCREENER_TOKEN_URL.format(address=token_mint), timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        pairs = data.get("pairs") or []
        # Pick the pair with the highest liquidity
        if not pairs:
            return None
        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0))
        price_str = best.get("priceUsd")
        return float(price_str) if price_str else None
    except (requests.RequestException, ValueError, TypeError) as exc:
        log.error("Failed to fetch price for %s: %s", token_mint[:8], exc)
        return None


class PaperTrader:
    """Simulates buys and sells using live prices with simulated slippage."""

    def __init__(self) -> None:
        self.pm = PositionManager()

    def simulate_buy(
        self,
        symbol: str,
        token_mint: str,
        pool_address: str,
        amount_usd: float,
        score: int,
        legit_verdict: str,
    ) -> dict:
        """Simulate a buy with live price and random slippage."""
        price = _fetch_price(token_mint)
        if price is None or price <= 0:
            log.error("Cannot paper buy %s — no price available", symbol)
            return {"success": False, "error": "No price available"}

        # Apply simulated slippage: 2-15% worse (higher buy price)
        slippage = random.uniform(0.02, 0.15)
        fill_price = price * (1 + slippage)
        tokens = amount_usd / fill_price

        pos_id = self.pm.open_position(
            symbol=symbol,
            token_mint=token_mint,
            pool_address=pool_address,
            entry_price=fill_price,
            amount_usd=amount_usd,
            tokens_held=tokens,
            score=score,
            legit_verdict=legit_verdict,
            tx_sig=f"paper_buy_{token_mint[:8]}",
            paper=True,
        )

        log.info(
            "PAPER BUY %s: $%.2f @ $%s (slippage %.1f%%) → %.4f tokens [pos #%d]",
            symbol, amount_usd, fill_price, slippage * 100, tokens, pos_id,
        )
        return {
            "success": True,
            "position_id": pos_id,
            "fill_price": fill_price,
            "tokens": tokens,
            "slippage_pct": slippage * 100,
        }

    def simulate_sell(
        self, position_id: int, pct: float = 100.0, reason: str = "manual"
    ) -> dict:
        """Simulate a sell with live price and random slippage."""
        positions = self.pm.get_open_positions()
        pos = next((p for p in positions if p["id"] == position_id), None)
        if not pos:
            return {"success": False, "error": f"Position {position_id} not found or not open"}

        price = _fetch_price(pos["token_mint"])
        if price is None or price <= 0:
            log.error("Cannot paper sell #%d — no price available", position_id)
            return {"success": False, "error": "No price available"}

        # Apply simulated slippage: 2-15% worse (lower sell price)
        slippage = random.uniform(0.02, 0.15)
        fill_price = price * (1 - slippage)

        # Update price before closing so PnL calc is accurate
        self.pm.update_price(position_id, fill_price)
        result = self.pm.close_position(position_id, fill_price, reason, f"paper_sell_{position_id}")

        log.info(
            "PAPER SELL #%d %s: @ $%s (slippage %.1f%%) PnL $%.2f (%.1f%%)",
            position_id, pos["symbol"], fill_price, slippage * 100,
            result["pnl_usd"], result["pnl_pct"],
        )
        return {
            "success": True,
            "fill_price": fill_price,
            "slippage_pct": slippage * 100,
            **result,
        }

    def update_all_prices(self) -> None:
        """Refresh current prices for all open positions."""
        for pos in self.pm.get_open_positions():
            price = _fetch_price(pos["token_mint"])
            if price and price > 0:
                self.pm.update_price(pos["id"], price)
