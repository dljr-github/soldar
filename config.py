"""All tunable thresholds for the meme coin screener."""

# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# Filters (hard rejections)
# ---------------------------------------------------------------------------
MIN_LIQUIDITY_USD = 5_000
MAX_AGE_HOURS = 6
SCAM_PATTERNS = [
    r"(?i)\brug\b",
    r"(?i)\bscam\b",
    r"(?i)\bhoneypot\b",
    r"(?i)\bfree\s*money\b",
    r"(?i)\bairdrop\b",
    r"(?i)\bsafe\s*moon\b",
    r"(?i)\belon\b",
    r"(?i)\b100x\b",
    r"(?i)\b1000x\b",
    r"(?i)\btest\b",
]

# ---------------------------------------------------------------------------
# Seen / deduplication
# ---------------------------------------------------------------------------
SEEN_FILE = "seen.json"
SEEN_EXPIRY_HOURS = 4

# ---------------------------------------------------------------------------
# Scoring weights  (each section sums to its max)
# ---------------------------------------------------------------------------
# Token Age (25 pts max)
AGE_THRESHOLDS = [
    (15,  25),   # < 15 min
    (60,  18),   # 15-60 min
    (240, 10),   # 1-4 hr
]
AGE_DEFAULT_POINTS = 0

# Volume / Liquidity Ratio (20 pts max)
VOL_LIQ_THRESHOLDS = [
    (2.0, 20),
    (1.0, 14),
    (0.5,  8),
]
VOL_LIQ_DEFAULT_POINTS = 0

# Price Momentum (20 pts max total: 12 from 5m + 8 from 1h capped at 500%)
MOMENTUM_5M_THRESHOLDS = [
    (50, 12),
    (20, 8),
    (10,  4),
]
MOMENTUM_1H_THRESHOLDS = [
    (100, 8),   # 1h > 100% (capped at 500 internally)
    (50,  5),
    (20,  3),
    (10,  1),
]
MOMENTUM_DEFAULT_POINTS = 0
MOMENTUM_BONUS = 3  # bonus if both 5m AND 1h are positive

# Buy Pressure (15 pts max)
BUY_PRESSURE_THRESHOLDS = [
    (0.75, 15),
    (0.60, 10),
    (0.50,  5),
]
BUY_PRESSURE_DEFAULT_POINTS = 0

# Liquidity Sweet Spot (10 pts max)
LIQ_SWEET_SPOT = [
    (20_000, 200_000, 10),
    (10_000,  20_000,  6),
    (200_000, 500_000, 6),
]
LIQ_SWEET_DEFAULT_POINTS = 0

# Market Cap (10 pts max)
MCAP_THRESHOLDS = [
    (500_000,    10),
    (2_000_000,   7),
    (10_000_000,  4),
]
MCAP_DEFAULT_POINTS = 0

# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------
ALERT_LEVELS = [
    (80, "HOT",   "\U0001f534"),   # 🔴
    (65, "WARM",  "\U0001f7e0"),   # 🟠
    (50, "WATCH", "\U0001f7e1"),   # 🟡
]

# Set to False to disable auto Telegram alerts (on-demand reports only)
AUTO_ALERTS_ENABLED = False

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"
DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?q=solana"
DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"

# ---------------------------------------------------------------------------
# Legitimacy / RugCheck thresholds
# ---------------------------------------------------------------------------
# Top single holder % — above REJECT = hard fail, above WARN = warning
LEGIT_TOP1_HOLDER_REJECT = 50.0   # > 50% owned by one wallet → reject
LEGIT_TOP1_HOLDER_WARN   = 20.0   # 20-50% → warn

# LP lock % — below REJECT = hard fail, below WARN = warning
LEGIT_LP_LOCKED_REJECT = 50.0     # < 50% locked → reject
LEGIT_LP_LOCKED_WARN   = 80.0     # 50-80% locked → warn

# RugCheck risk score — higher is riskier
LEGIT_RUGCHECK_SCORE_WARN = 800   # above this → warn (not hard reject)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "screener.log"
CONSOLE_TOP_N = 10
