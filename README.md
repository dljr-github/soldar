# Solana Meme Coin Screener

Scans for early meme coin pumps on Solana using the DEX Screener API and fires Telegram alerts when it finds something hot.

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate meme-screener
```

Or with pip directly:

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Copy/edit `.env` with your Telegram bot token and chat ID:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Usage

### Live mode (sends Telegram alerts)

```bash
python screener.py
```

### Dry run (prints alerts to console, no Telegram)

```bash
python screener.py --dry-run
```

### Single cycle (run once and exit)

```bash
python screener.py --dry-run --once
```

## How It Works

1. **Fetch** – Polls multiple DEX Screener endpoints every 90s for new Solana pairs
2. **Filter** – Rejects tokens with low liquidity, old age, or scam-pattern names
3. **Score** – Assigns 0-100 score based on 6 weighted signals:
   - Token Age (25 pts) – newer is better
   - Volume/Liquidity Ratio (20 pts) – high volume relative to liquidity
   - Price Momentum (20 pts) – 5-minute price surge
   - Buy Pressure (15 pts) – buy-to-sell ratio
   - Liquidity Sweet Spot (10 pts) – not a rug, not already huge
   - Market Cap (10 pts) – small cap = more upside
4. **Alert** – Sends Telegram message for tokens scoring 55+

### Alert Levels

| Score | Level | Emoji |
|-------|-------|-------|
| 85+   | HOT   | 🔴    |
| 70-84 | WARM  | 🟠    |
| 55-69 | WATCH | 🟡    |
| < 55  | —     | silent |

## Tuning

All thresholds live in `config.py`. Adjust scoring weights, filter cutoffs, alert levels, and polling interval without touching any logic.

## Files

| File | Purpose |
|------|---------|
| `screener.py` | Main polling loop |
| `signals.py` | Scoring engine (0-100) |
| `filters.py` | Hard rejection filters + deduplication |
| `alerts.py` | Telegram alert formatting & sending |
| `config.py` | All tunable thresholds |
| `seen.json` | Auto-created deduplication state |
| `screener.log` | Rolling log output |
