#!/usr/bin/env python3
"""
Backtesting suite for Soldar — simulate paper trading on held-out test set.
Compares: rule-based scorer vs XGBoost vs LSTM (if available).

Usage:
  python ml/backtest.py                    # run all strategies
  python ml/backtest.py --strategy xgb     # single strategy
  python ml/backtest.py --threshold 0.7    # custom entry threshold
  python ml/backtest.py --save-report      # save full HTML report
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")

DROP_COLS = {"label", "source"}

# ---------------------------------------------------------------------------
# PnL simulation parameters
# ---------------------------------------------------------------------------
POSITION_SIZE_USD = 50.0
STARTING_CAPITAL = 1000.0
MAX_CONCURRENT = 3
ENTRY_SLIPPAGE = 0.05   # 5%
EXIT_SLIPPAGE = 0.08    # 8%
TOTAL_SLIPPAGE = 1 - (1 - ENTRY_SLIPPAGE) * (1 - EXIT_SLIPPAGE)  # ~12.6%


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data() -> pd.DataFrame:
    """Load test.parquet with features and labels."""
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")
    df = pd.read_parquet(TEST_PATH)
    print(f"Loaded test set: {len(df)} samples "
          f"({(df['label'] == 1).sum()} pumps, {(df['label'] == 0).sum()} non-pumps)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    return X


# ---------------------------------------------------------------------------
# Rule-based scoring (approximate screener logic from available features)
# ---------------------------------------------------------------------------
def _step_score(values: np.ndarray, thresholds: list[tuple[float, int]]) -> np.ndarray:
    """Apply step-function scoring (like config.py threshold lists)."""
    result = np.zeros(len(values))
    for thresh, pts in thresholds:
        result = np.where((values >= thresh) & (result < pts), pts, result)
    return result


def compute_rule_based_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Approximate the 100-point screener score from test.parquet features.
    Features are normalized 0-1; step-function thresholds mirror config.py.

    Mapping:
      momentum          -> Price Momentum      (max 20 pts)
      buy_sell_pressure  -> Buy Pressure        (max 15 pts)
      volume_intensity   -> Volume/Liq ratio    (max 20 pts)
      price_range        -> Volatility / Age    (max 15 pts)
      group2_top1_pct    -> Holder concentration (max 10 pts, lower = better)
      group3_wash_ratio  -> Wash signal         (max 10 pts, lower = better)
      group3_trader_num  -> Market participation (max 10 pts)
    """
    scores = np.zeros(len(df))

    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0).values
        return np.zeros(len(df))

    # Momentum (max 20 pts) — step thresholds on normalized 0-1
    scores += _step_score(_col("momentum"), [(0.05, 4), (0.10, 8), (0.20, 12), (0.40, 16), (0.60, 20)])

    # Buy pressure (max 15 pts)
    scores += _step_score(_col("buy_sell_pressure"), [(0.30, 5), (0.50, 10), (0.70, 15)])

    # Volume intensity (max 20 pts)
    scores += _step_score(_col("volume_intensity"), [(0.05, 4), (0.10, 8), (0.20, 14), (0.40, 20)])

    # Price range / volatility (max 15 pts)
    scores += _step_score(_col("price_range"), [(0.20, 5), (0.50, 10), (0.80, 15)])

    # Holder concentration — lower top1% = healthier (max 10 pts)
    t1 = _col("group2_top1_pct")
    scores += np.where(t1 <= 0.05, 10, np.where(t1 <= 0.10, 7, np.where(t1 <= 0.20, 4, 0)))

    # Wash ratio — lower = more legit (max 10 pts)
    wr = _col("group3_wash_ratio")
    scores += np.where(wr <= 0.05, 10, np.where(wr <= 0.15, 7, np.where(wr <= 0.30, 4, 0)))

    # Trader count — normalized 0-1, more = better (max 10 pts)
    scores += _step_score(_col("group3_trader_num"), [(0.01, 3), (0.05, 6), (0.10, 10)])

    return scores


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
def strategy_rule_based(
    df: pd.DataFrame, threshold: float = 80.0
) -> np.ndarray:
    """Flag tokens scoring >= threshold on rule-based composite."""
    scores = compute_rule_based_scores(df)
    return (scores >= threshold).astype(int)


def strategy_model(
    df: pd.DataFrame, model_name: str, threshold: float = 0.5
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a pickled model and return (predictions, probabilities) or (None, None)."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None, None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X = prepare_X(df, feature_cols)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return preds, proba


def strategy_combined_vote(
    probas: dict[str, np.ndarray], threshold: float = 0.5
) -> np.ndarray:
    """Soft voting: average probabilities of available models."""
    if not probas:
        return np.zeros(0)
    stacked = np.column_stack(list(probas.values()))
    avg_proba = stacked.mean(axis=1)
    return (avg_proba >= threshold).astype(int)


# ---------------------------------------------------------------------------
# PnL simulation
# TODO(audit-HIGH): Backtest does not enforce MAX_CONCURRENT_POSITIONS or
# starting capital constraints. Equity curves overstate realistic PnL.
# Also: each strategy uses a different RNG seed, making cross-strategy
# comparison unreliable. Use the same seed for fair comparison.
# ---------------------------------------------------------------------------
def simulate_pnl(
    predictions: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> dict:
    """
    Simulate realistic PnL for a set of trading decisions.

    Returns dict with per-trade returns and aggregate metrics.
    """
    rng = np.random.RandomState(seed)
    n_total = len(labels)
    flagged_mask = predictions == 1

    n_trades = int(flagged_mask.sum())
    if n_trades == 0:
        return {
            "n_trades": 0,
            "n_total": n_total,
            "trade_returns": [],
            "equity_curve": [STARTING_CAPITAL],
        }

    flagged_labels = labels[flagged_mask]
    trade_returns = np.zeros(n_trades)

    for i, lbl in enumerate(flagged_labels):
        if lbl == 1:
            # True positive: pump — LogNormal gain
            raw_gain = rng.lognormal(mean=0.8, sigma=0.9)
            # Clip to reasonable range (20% to 500%)
            raw_gain = np.clip(raw_gain, 0.20, 5.0)
            trade_returns[i] = raw_gain - TOTAL_SLIPPAGE
        else:
            # False positive: not a pump — Normal loss/flat
            raw_return = rng.normal(loc=-0.05, scale=0.10)
            raw_return = np.clip(raw_return, -0.20, 0.10)
            trade_returns[i] = raw_return - TOTAL_SLIPPAGE

    # Build equity curve
    equity = [STARTING_CAPITAL]
    for ret in trade_returns:
        pnl = POSITION_SIZE_USD * ret
        equity.append(equity[-1] + pnl)

    return {
        "n_trades": n_trades,
        "n_total": n_total,
        "trade_returns": trade_returns.tolist(),
        "equity_curve": equity,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    pnl_result: dict,
    strategy_name: str,
) -> dict:
    """Compute full metrics dict for a strategy."""
    n_total = len(labels)
    n_trades = pnl_result["n_trades"]
    trade_returns = np.array(pnl_result["trade_returns"])
    equity = pnl_result["equity_curve"]

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # PnL
    trade_pnls = trade_returns * POSITION_SIZE_USD
    total_pnl = float(trade_pnls.sum()) if n_trades > 0 else 0.0
    wins = trade_pnls[trade_pnls > 0] if n_trades > 0 else np.array([])
    losses = trade_pnls[trade_pnls <= 0] if n_trades > 0 else np.array([])

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak * 100
    max_dd = float(drawdown.min())

    # Sharpe ratio (per-trade, annualized assuming ~250 trades/year)
    if n_trades > 1 and trade_returns.std() > 0:
        sharpe = float(trade_returns.mean() / trade_returns.std() * np.sqrt(250))
    else:
        sharpe = 0.0

    total_non_pumps = int((labels == 0).sum())

    return {
        "strategy": strategy_name,
        "n_trades": n_trades,
        "n_total": n_total,
        "trade_rate_pct": round(n_trades / n_total * 100, 1) if n_total > 0 else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "total_pnl_usd": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / STARTING_CAPITAL * 100, 1),
        "win_rate": round(win_rate, 3),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "max_drawdown_pct": round(max_dd, 1),
        "sharpe_ratio": round(sharpe, 2),
        "rugs_avoided": tn,
        "rugs_total": total_non_pumps,
        "starting_capital": STARTING_CAPITAL,
        "final_capital": round(STARTING_CAPITAL + total_pnl, 2),
    }


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------
def run_backtest(
    strategy_filter: str | None = None,
    threshold: float = 0.5,
    save_report: bool = False,
) -> list[dict]:
    """Run backtest across all strategies. Return list of metrics dicts."""
    df = load_test_data()
    labels = df["label"].values.astype(int)

    all_results = []
    all_equity_curves = {}
    model_probas: dict[str, np.ndarray] = {}

    # ── Strategy 1: rule_based ────────────────────────────────────────────
    if strategy_filter is None or strategy_filter == "rule_based":
        print("\n" + "=" * 60)
        print("Strategy: rule_based (score >= 80)")
        print("=" * 60)
        preds = strategy_rule_based(df, threshold=80.0)
        pnl = simulate_pnl(preds, labels, seed=42)
        metrics = compute_metrics(preds, labels, pnl, "rule_based")
        all_results.append(metrics)
        all_equity_curves["rule_based"] = pnl["equity_curve"]

    # ── Strategy 2: xgb_baseline ──────────────────────────────────────────
    if strategy_filter is None or strategy_filter == "xgb":
        print("\n" + "=" * 60)
        print("Strategy: xgb_baseline")
        print("=" * 60)
        preds, proba = strategy_model(df, "xgb_baseline", threshold=threshold)
        if preds is not None:
            pnl = simulate_pnl(preds, labels, seed=43)
            metrics = compute_metrics(preds, labels, pnl, "xgb_baseline")
            all_results.append(metrics)
            all_equity_curves["xgb_baseline"] = pnl["equity_curve"]
            model_probas["xgb_baseline"] = proba
        else:
            print("  [SKIP] xgb_baseline.pkl not found")

    # ── Strategy 3: xgb_tuned ─────────────────────────────────────────────
    if strategy_filter is None or strategy_filter == "xgb":
        print("\n" + "=" * 60)
        print("Strategy: xgb_tuned")
        print("=" * 60)
        preds, proba = strategy_model(df, "xgb_tuned", threshold=threshold)
        if preds is not None:
            pnl = simulate_pnl(preds, labels, seed=44)
            metrics = compute_metrics(preds, labels, pnl, "xgb_tuned")
            all_results.append(metrics)
            all_equity_curves["xgb_tuned"] = pnl["equity_curve"]
            model_probas["xgb_tuned"] = proba
        else:
            print("  [SKIP] xgb_tuned.pkl not found")

    # ── Strategy 4: lgbm ──────────────────────────────────────────────────
    if strategy_filter is None or strategy_filter == "lgbm":
        print("\n" + "=" * 60)
        print("Strategy: lgbm_baseline")
        print("=" * 60)
        preds, proba = strategy_model(df, "lgbm_baseline", threshold=threshold)
        if preds is not None:
            pnl = simulate_pnl(preds, labels, seed=45)
            metrics = compute_metrics(preds, labels, pnl, "lgbm_baseline")
            all_results.append(metrics)
            all_equity_curves["lgbm_baseline"] = pnl["equity_curve"]
            model_probas["lgbm_baseline"] = proba
        else:
            print("  [SKIP] lgbm_baseline.pkl not found")

    # ── Strategy 5: combined_vote ─────────────────────────────────────────
    if (strategy_filter is None or strategy_filter == "combined") and len(model_probas) >= 2:
        print("\n" + "=" * 60)
        print("Strategy: combined_vote (soft voting)")
        print("=" * 60)
        preds = strategy_combined_vote(model_probas, threshold=threshold)
        if len(preds) > 0:
            pnl = simulate_pnl(preds, labels, seed=46)
            metrics = compute_metrics(preds, labels, pnl, "combined_vote")
            all_results.append(metrics)
            all_equity_curves["combined_vote"] = pnl["equity_curve"]

    # ── Print comparison table ────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("BACKTEST RESULTS — STRATEGY COMPARISON")
    print("=" * 100)
    header = (
        f"{'Strategy':<16} {'Trades':>7} {'Prec':>7} {'Recall':>7} "
        f"{'F1':>7} {'WinRate':>8} {'PnL $':>9} {'PnL %':>8} "
        f"{'PF':>7} {'MaxDD':>8} {'Sharpe':>7} {'Rugs Avoided':>14}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        line = (
            f"{r['strategy']:<16} {r['n_trades']:>7} {r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} {r['f1']:>7.3f} {r['win_rate']:>7.1%} "
            f"{r['total_pnl_usd']:>+9.2f} {r['total_pnl_pct']:>+7.1f}% "
            f"{r['profit_factor']:>7.2f} {r['max_drawdown_pct']:>+7.1f}% "
            f"{r['sharpe_ratio']:>7.2f} {r['rugs_avoided']:>6}/{r['rugs_total']}"
        )
        print(line)
    print("=" * 100)

    # Find best strategy
    if all_results:
        best = max(all_results, key=lambda r: r["total_pnl_usd"])
        print(f"\nBest strategy: {best['strategy']} "
              f"(+{best['total_pnl_pct']:.1f}% on test set, "
              f"${best['total_pnl_usd']:+.2f})")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    results_path = os.path.join(MODEL_DIR, "backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    curves_path = os.path.join(MODEL_DIR, "backtest_equity_curves.json")
    with open(curves_path, "w") as f:
        json.dump(all_equity_curves, f)
    print(f"Equity curves saved to {curves_path}")

    # ── HTML report ───────────────────────────────────────────────────────
    if save_report:
        _generate_html_report(all_results, all_equity_curves)

    return all_results


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------
def _generate_html_report(
    results: list[dict],
    equity_curves: dict[str, list[float]],
) -> None:
    """Generate an interactive HTML report with Plotly charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Colors for strategies
    colors = {
        "rule_based": "#7d8590",
        "xgb_baseline": "#58a6ff",
        "xgb_tuned": "#3fb950",
        "lgbm_baseline": "#d29922",
        "combined_vote": "#f778ba",
    }

    # Equity curves chart
    fig_equity = go.Figure()
    best_strategy = max(results, key=lambda r: r["total_pnl_usd"])["strategy"] if results else ""
    for name, curve in equity_curves.items():
        width = 3 if name == best_strategy else 1.5
        fig_equity.add_trace(go.Scatter(
            y=curve,
            mode="lines",
            name=name,
            line=dict(color=colors.get(name, "#58a6ff"), width=width),
        ))
    fig_equity.add_hline(
        y=STARTING_CAPITAL, line_dash="dot",
        line_color="rgba(125,133,144,0.4)",
        annotation_text="Starting Capital",
    )
    fig_equity.update_layout(
        title="Equity Curves — Cumulative PnL by Strategy",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#7d8590", family="Courier New, monospace", size=11),
        xaxis=dict(title="Trade #", gridcolor="#21262d"),
        yaxis=dict(title="Portfolio Value ($)", gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=60, b=40),
        height=500,
    )

    # Metrics bar charts
    fig_bars = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Precision", "Win Rate", "Total PnL %", "Sharpe Ratio"],
    )
    names = [r["strategy"] for r in results]
    for i, (metric, col) in enumerate([
        ("precision", 1), ("win_rate", 2),
        ("total_pnl_pct", 3), ("sharpe_ratio", 4),
    ]):
        vals = [r[metric] for r in results]
        bar_colors = [colors.get(n, "#58a6ff") for n in names]
        fig_bars.add_trace(
            go.Bar(x=names, y=vals, marker_color=bar_colors, showlegend=False),
            row=1, col=col,
        )
    fig_bars.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#7d8590", family="Courier New, monospace", size=10),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig_bars.update_xaxes(gridcolor="#21262d", tickangle=30)
    fig_bars.update_yaxes(gridcolor="#21262d")

    # Build metrics table HTML
    table_rows = ""
    for r in results:
        is_best = r["strategy"] == best_strategy
        row_style = "background:rgba(63,185,80,0.08);" if is_best else ""
        table_rows += f"""
        <tr style="{row_style}">
            <td><strong>{r['strategy']}</strong>{'  ★' if is_best else ''}</td>
            <td>{r['n_trades']}</td>
            <td>{r['precision']:.3f}</td>
            <td>{r['recall']:.3f}</td>
            <td>{r['win_rate']:.1%}</td>
            <td style="color:{'#3fb950' if r['total_pnl_usd']>0 else '#f85149'}">${r['total_pnl_usd']:+.2f}</td>
            <td>{r['profit_factor']:.2f}</td>
            <td>{r['max_drawdown_pct']:+.1f}%</td>
            <td>{r['sharpe_ratio']:.2f}</td>
        </tr>"""

    # Key insights
    if results:
        best = max(results, key=lambda r: r["total_pnl_usd"])
        total_non_pumps = best["rugs_total"]
        insights_html = f"""
        <div style="margin:20px 0;padding:16px;background:#161b22;border:1px solid #21262d;border-radius:8px;">
            <h3 style="color:#3fb950;margin-top:0;">Key Insights</h3>
            <ul style="color:#e6edf3;line-height:2;">
                <li>Best strategy: <strong>{best['strategy']}</strong> (+{best['total_pnl_pct']:.1f}% on test set)</li>
                <li>Rugs avoided: <strong>{best['rugs_avoided']:,}</strong> out of {total_non_pumps:,} non-pumps</li>
                <li>Trade selectivity: <strong>{best['trade_rate_pct']:.1f}%</strong> of tokens flagged</li>
                <li>Final capital: <strong>${best['final_capital']:,.2f}</strong> from ${STARTING_CAPITAL:,.2f} starting</li>
            </ul>
            <div style="margin-top:12px;padding:10px;background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);border-radius:6px;color:#d29922;font-size:13px;">
                &#9888; Past performance on test set does not guarantee live results. Always paper trade first.
            </div>
        </div>"""
    else:
        insights_html = ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Soldar Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background:#0d1117; color:#e6edf3; font-family:'Courier New',monospace; padding:20px; }}
        h1 {{ color:#58a6ff; letter-spacing:3px; }}
        h2 {{ color:#7d8590; font-size:14px; text-transform:uppercase; letter-spacing:1px; margin-top:30px; }}
        table {{ width:100%; border-collapse:collapse; margin:10px 0; }}
        th {{ background:#161b22; color:#7d8590; text-transform:uppercase; font-size:10px;
             letter-spacing:0.5px; padding:8px 12px; text-align:left; border-bottom:1px solid #21262d; }}
        td {{ padding:8px 12px; border-bottom:1px solid #1c2128; font-size:13px; }}
    </style>
</head>
<body>
    <h1>SOLDAR BACKTEST REPORT</h1>
    <p style="color:#7d8590;">Test set: {results[0]['n_total'] if results else 0:,} samples |
       Starting capital: ${STARTING_CAPITAL:,.2f} |
       Position size: ${POSITION_SIZE_USD:.0f}</p>

    <h2>Strategy Comparison</h2>
    <table>
        <tr><th>Strategy</th><th>Trades</th><th>Precision</th><th>Recall</th>
            <th>Win Rate</th><th>Total PnL</th><th>Profit Factor</th>
            <th>Max DD</th><th>Sharpe</th></tr>
        {table_rows}
    </table>

    <h2>Equity Curves</h2>
    <div id="equity-chart"></div>

    <h2>Metrics Comparison</h2>
    <div id="bars-chart"></div>

    {insights_html}

    <script>
        Plotly.newPlot('equity-chart', {fig_equity.to_json()}.data, {fig_equity.to_json()}.layout);
        Plotly.newPlot('bars-chart', {fig_bars.to_json()}.data, {fig_bars.to_json()}.layout);
    </script>
</body>
</html>"""

    report_path = os.path.join(MODEL_DIR, "backtest_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Soldar backtesting suite")
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["rule_based", "xgb", "lgbm", "combined"],
        help="Run only this strategy (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Prediction probability threshold for ML models (default: 0.5)",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Generate HTML report with Plotly charts",
    )
    args = parser.parse_args()

    t0 = time.time()
    run_backtest(
        strategy_filter=args.strategy,
        threshold=args.threshold,
        save_report=args.save_report,
    )
    print(f"\nBacktest completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
