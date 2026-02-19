"""Tracks open positions and enforces risk limits."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

import config as cfg

try:
    import numpy as np
except ImportError:
    np = None

log = logging.getLogger("execution.positions")

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "trades.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    token_mint TEXT,
    pool_address TEXT,
    entry_price REAL,
    entry_time TEXT,
    entry_amount_usd REAL,
    tokens_held REAL,
    current_price REAL,
    last_updated TEXT,
    status TEXT,
    screener_score INTEGER,
    legit_verdict TEXT,
    paper INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    action TEXT,
    amount_usd REAL,
    price REAL,
    tokens REAL,
    pnl_usd REAL,
    pnl_pct REAL,
    timestamp TEXT,
    tx_signature TEXT,
    exit_reason TEXT
);
"""


class PositionManager:
    """SQLite-backed position tracker with risk limit enforcement."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def can_open_position(self, amount_usd: float) -> tuple[bool, str]:
        """Check risk limits before opening a new position."""
        if amount_usd > cfg.MAX_POSITION_SIZE_USD:
            return False, f"Amount ${amount_usd:.2f} exceeds max ${cfg.MAX_POSITION_SIZE_USD:.2f}"

        open_count = self.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE status = 'open'"
        ).fetchone()[0]
        if open_count >= cfg.MAX_CONCURRENT_POSITIONS:
            return False, f"Already at max {cfg.MAX_CONCURRENT_POSITIONS} open positions"

        daily_pnl = self.get_daily_pnl()
        if daily_pnl < cfg.MAX_DAILY_LOSS_USD:  # strict < for correct boundary
            return False, f"Daily loss limit hit: ${daily_pnl:.2f} (limit ${cfg.MAX_DAILY_LOSS_USD:.2f})"

        return True, "OK"

    def open_position(
        self,
        symbol: str,
        token_mint: str,
        pool_address: str,
        entry_price: float,
        amount_usd: float,
        tokens_held: float,
        score: int,
        legit_verdict: str,
        tx_sig: str,
        paper: bool = False,
    ) -> int:
        """Record a new open position and its buy trade. Returns position ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                cur = self.conn.execute(
                    """INSERT INTO positions
                       (symbol, token_mint, pool_address, entry_price, entry_time,
                        entry_amount_usd, tokens_held, current_price, last_updated,
                        status, screener_score, legit_verdict, paper)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                    (symbol, token_mint, pool_address, entry_price, now,
                     amount_usd, tokens_held, entry_price, now,
                     score, legit_verdict, 1 if paper else 0),
                )
                pos_id = cur.lastrowid
                self.conn.execute(
                    """INSERT INTO trades
                       (position_id, action, amount_usd, price, tokens, pnl_usd, pnl_pct,
                        timestamp, tx_signature, exit_reason)
                       VALUES (?, 'buy', ?, ?, ?, 0, 0, ?, ?, NULL)""",
                    (pos_id, amount_usd, entry_price, tokens_held, now, tx_sig),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        log.info("Opened position #%d: %s $%.2f @ $%s", pos_id, symbol, amount_usd, entry_price)
        return pos_id

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str,
        tx_sig: str,
        partial_pct: float = 100.0,
    ) -> dict:
        """Close (or partially close) a position. Returns PnL info."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM positions WHERE id = ? AND status = 'open'",
                (position_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Position {position_id} not found or not open")

            now = datetime.now(timezone.utc).isoformat()
            sell_fraction = min(partial_pct, 100.0) / 100.0
            tokens_to_sell = row["tokens_held"] * sell_fraction
            sell_usd = tokens_to_sell * exit_price
            # Cost basis proportional to tokens being sold, not original entry
            cost_basis = row["entry_amount_usd"] * sell_fraction
            pnl_usd = sell_usd - cost_basis
            pnl_pct = (pnl_usd / cost_basis * 100) if cost_basis else 0

            action = "sell_all" if partial_pct >= 100 else "sell_partial"
            if exit_reason == "stop_loss":
                action = "stop_loss"

            remaining_tokens = row["tokens_held"] - tokens_to_sell
            remaining_amount = row["entry_amount_usd"] * (1.0 - sell_fraction)
            new_status = "closed" if partial_pct >= 100 else "open"
            if exit_reason == "stop_loss":
                new_status = "stopped"

            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """INSERT INTO trades
                       (position_id, action, amount_usd, price, tokens, pnl_usd, pnl_pct,
                        timestamp, tx_signature, exit_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (position_id, action, sell_usd, exit_price, tokens_to_sell,
                     pnl_usd, pnl_pct, now, tx_sig, exit_reason),
                )
                self.conn.execute(
                    """UPDATE positions
                       SET tokens_held = ?, entry_amount_usd = ?,
                           current_price = ?, last_updated = ?, status = ?
                       WHERE id = ?""",
                    (remaining_tokens, remaining_amount,
                     exit_price, now, new_status, position_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

        log.info(
            "Closed position #%d (%s): PnL $%.2f (%.1f%%) reason=%s",
            position_id, row["symbol"], pnl_usd, pnl_pct, exit_reason,
        )
        return {"pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "action": action}

    def get_open_positions(self) -> list[dict]:
        """Return all open positions as dicts."""
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self) -> float:
        """Sum of today's realized PnL from closed trades."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE timestamp >= ? AND action != 'buy'",
            (today,),
        ).fetchone()
        return row[0]

    def update_price(self, position_id: int, current_price: float) -> None:
        """Update the mark price for an open position."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE positions SET current_price = ?, last_updated = ? WHERE id = ?",
            (current_price, now, position_id),
        )
        self.conn.commit()

    def check_stop_losses(self) -> list[dict]:
        """Check all open positions against the hard stop loss. Returns triggered positions."""
        triggered = []
        for pos in self.get_open_positions():
            if pos["current_price"] and pos["entry_price"]:
                pnl_pct = ((pos["current_price"] - pos["entry_price"]) / pos["entry_price"]) * 100
                if pnl_pct <= cfg.HARD_STOP_LOSS_PCT:
                    triggered.append(pos)
                    log.warning(
                        "STOP LOSS triggered for #%d %s: %.1f%% (limit %.1f%%)",
                        pos["id"], pos["symbol"], pnl_pct, cfg.HARD_STOP_LOSS_PCT,
                    )
        return triggered

    # ------------------------------------------------------------------
    # Position lookup
    # ------------------------------------------------------------------

    def get_position(self, position_id: int) -> dict | None:
        """Return a single position as a dict, or None if not found."""
        row = self.conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # RL agent exit decisions
    # ------------------------------------------------------------------

    def get_rl_exit_action(self, position_id: int, current_price: float) -> str:
        """
        Query RL agent for exit decision on an open position.
        Returns: 'hold' | 'exit_25' | 'exit_50' | 'exit_all'
        Falls back to rule-based if RL model not available.
        """
        pos = self.get_position(position_id)
        if pos is None or pos.get("status") != "open":
            return "hold"

        entry_price = pos["entry_price"]
        entry_time = datetime.fromisoformat(pos["entry_time"])
        time_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60.0

        # Hard max hold time — force exit regardless of RL/rules
        max_hold = getattr(cfg, "MAX_HOLD_MINUTES", 120.0)
        if time_held >= max_hold:
            log.warning(
                "Position #%d exceeded max hold time (%.0fm) — forcing exit",
                position_id, time_held,
            )
            return "exit_all"

        try:
            from stable_baselines3 import PPO

            if np is None:
                return self._rule_based_exit(position_id, current_price)

            ml_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ml", "models",
            )
            # Prefer best-validated model over final training checkpoint
            best_path = os.path.join(ml_dir, "best_model.zip")
            final_path = os.path.join(ml_dir, "ppo_exit_agent.zip")
            model_path = best_path if os.path.exists(best_path) else final_path
            if not os.path.exists(model_path):
                return self._rule_based_exit(position_id, current_price)

            # Lazy load model with freshness check (reload if file changes)
            model_mtime = os.path.getmtime(model_path)
            if (
                not hasattr(self, "_ppo_model")
                or self._ppo_model is None
                or getattr(self, "_ppo_model_mtime", 0) != model_mtime
            ):
                self._ppo_model = PPO.load(model_path)
                self._ppo_model_mtime = model_mtime
                log.info("Loaded PPO model from %s", model_path)

            price_ratio = current_price / entry_price if entry_price > 0 else 1.0

            # Approximate per-step momentum from last price update
            prev_price = pos.get("current_price") or entry_price
            step_momentum = (current_price / prev_price - 1.0) if prev_price > 0 else 0.0

            obs = np.array([
                np.clip(price_ratio, 0.0, 20.0),
                np.clip(time_held / 120.0, 0.0, 1.0),
                1.0,                    # vol_ratio placeholder
                0.5,                    # buy_sell_ratio neutral default
                0.0,                    # realized_pnl placeholder
                np.clip(step_momentum, -1.0, 5.0),  # per-step momentum, not total return
            ], dtype=np.float32)

            action, _ = self._ppo_model.predict(obs, deterministic=True)
            action_map = {0: "hold", 1: "exit_25", 2: "exit_50", 3: "exit_all"}
            return action_map.get(int(action), "hold")

        except Exception as exc:
            log.debug("RL exit fallback for pos #%d: %s", position_id, exc)
            return self._rule_based_exit(position_id, current_price)

    def _rule_based_exit(self, position_id: int, current_price: float) -> str:
        """Fallback rule-based exit logic."""
        pos = self.get_position(position_id)
        if pos is None:
            return "hold"
        entry_price = pos["entry_price"]
        pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

        if pct <= cfg.HARD_STOP_LOSS_PCT:
            return "exit_all"   # stop loss
        if pct >= 500:
            return "exit_all"   # full exit at 6x
        if pct >= 200:
            return "exit_50"    # take half at 3x
        return "hold"
