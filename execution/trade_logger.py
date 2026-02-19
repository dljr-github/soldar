"""Exports trade history for RL training and reporting."""

from __future__ import annotations

import csv
import logging
import os
import sqlite3

from execution.position_manager import DB_PATH

log = logging.getLogger("execution.trade_logger")


def _get_conn(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def export_to_csv(output_path: str, db_path: str | None = None) -> None:
    """Export the trades table to CSV."""
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT t.*, p.symbol, p.token_mint, p.pool_address, p.entry_price,
                  p.entry_amount_usd, p.screener_score, p.legit_verdict, p.paper
           FROM trades t
           JOIN positions p ON t.position_id = p.id
           ORDER BY t.timestamp"""
    ).fetchall()
    conn.close()

    if not rows:
        log.warning("No trades to export")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    keys = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    log.info("Exported %d trades to %s", len(rows), output_path)


def export_to_parquet(output_path: str, db_path: str | None = None) -> None:
    """Export the trades table to parquet for ML pipelines."""
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas required for parquet export — pip install pandas pyarrow")
        return

    conn = _get_conn(db_path)
    df = pd.read_sql_query(
        """SELECT t.*, p.symbol, p.token_mint, p.pool_address, p.entry_price,
                  p.entry_amount_usd, p.screener_score, p.legit_verdict, p.paper
           FROM trades t
           JOIN positions p ON t.position_id = p.id
           ORDER BY t.timestamp""",
        conn,
    )
    conn.close()

    if df.empty:
        log.warning("No trades to export")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Exported %d trades to %s", len(df), output_path)


def get_stats(db_path: str | None = None) -> dict:
    """Return summary stats: win_rate, avg_pnl, total_trades, best/worst trade."""
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT pnl_usd, pnl_pct FROM trades WHERE action != 'buy'"
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_usd": 0.0,
            "avg_pnl_pct": 0.0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0,
            "total_pnl_usd": 0.0,
        }

    pnls = [r["pnl_usd"] for r in rows]
    pcts = [r["pnl_pct"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)

    return {
        "total_trades": len(pnls),
        "win_rate": (wins / len(pnls) * 100) if pnls else 0.0,
        "avg_pnl_usd": sum(pnls) / len(pnls),
        "avg_pnl_pct": sum(pcts) / len(pcts),
        "best_trade_usd": max(pnls),
        "worst_trade_usd": min(pnls),
        "total_pnl_usd": sum(pnls),
    }


def print_summary(db_path: str | None = None) -> None:
    """Pretty print trade stats to console."""
    stats = get_stats(db_path)
    print("\n" + "=" * 50)
    print("  TRADE SUMMARY")
    print("=" * 50)
    print(f"  Total Trades:   {stats['total_trades']}")
    print(f"  Win Rate:       {stats['win_rate']:.1f}%")
    print(f"  Avg PnL:        ${stats['avg_pnl_usd']:+.2f} ({stats['avg_pnl_pct']:+.1f}%)")
    print(f"  Total PnL:      ${stats['total_pnl_usd']:+.2f}")
    print(f"  Best Trade:     ${stats['best_trade_usd']:+.2f}")
    print(f"  Worst Trade:    ${stats['worst_trade_usd']:+.2f}")
    print("=" * 50 + "\n")
