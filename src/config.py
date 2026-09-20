"""All tunable settings in one place."""
import datetime as dt
import os
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.environ.get("DATA_DIR", "data")

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
EOD_CUTOFF = dt.time(16, 30)      # after this, runs do nothing
HORIZONS = (15, 30, 60)           # minutes

# How much each signal family counts, per horizon.
WEIGHTS = {
    15: {"technical": 0.35, "options": 0.27, "global": 0.13, "news": 0.13, "breadth": 0.12},
    30: {"technical": 0.31, "options": 0.27, "global": 0.17, "news": 0.13, "breadth": 0.12},
    60: {"technical": 0.27, "options": 0.25, "global": 0.20, "news": 0.18, "breadth": 0.10},
}

# A direction is only called when the blended bias is strong AND enough signal families agree.
# Otherwise the model says SIDEWAYS ("no clear edge"). Fewer calls, but hopefully better ones.
DIRECTION_THRESHOLD = 0.18
MIN_AGREE = 0.65             # share of signal weight that must point the same way
MIN_CONF = 22.0
DRIFT_K = 0.5                # bias=1 shifts the range centre by 0.5 sigma
Z68, Z90 = 1.0, 1.645        # range widths (in sigmas)
TRADING_MINUTES_PER_DAY = 375

# Global cues: yahoo symbol -> (label, sign, weight, pct-move that counts as "full" signal)
# sign +1 means "up is good for Nifty", -1 means "up is bad for Nifty"
GLOBAL_CUES = {
    "ES=F":      ("S&P 500 futures", +1, 0.22, 1.0),
    "NQ=F":      ("Nasdaq futures", +1, 0.12, 1.5),
    "^N225":     ("Nikkei 225", +1, 0.08, 1.5),
    "^HSI":      ("Hang Seng", +1, 0.10, 1.5),
    "^GDAXI":    ("DAX", +1, 0.08, 1.2),
    "^INDIAVIX": ("India VIX", -1, 0.20, 5.0),
    "BZ=F":      ("Brent crude", -1, 0.08, 2.0),
    "DX-Y.NYB":  ("US Dollar Index", -1, 0.05, 0.5),
    "INR=X":     ("USD/INR", -1, 0.05, 0.4),
    "^TNX":      ("US 10Y yield", -1, 0.05, 2.0),
}

# Alert thresholds
OI_MIN_Z = 4.0               # robust z-score of the 15-min OI change across strikes
OI_MIN_PCT = 0.20            # and at least +20% vs previous OI at that strike
OI_MIN_SHARE = 0.005         # and at least 0.5% of that side's total OI
PCR_SHIFT = 0.08             # PCR move between runs that triggers an alert
VIX_SPIKE_PCT = 4.0
PRICE_SHOCK_SIGMA = 1.6
ALERT_COOLDOWN_MIN = 45

GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS",
    "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]
USE_GROUNDED_SEARCH = os.environ.get("USE_GROUNDED_SEARCH", "1") == "1"

HISTORY_DAYS = 30            # detailed prediction records kept

# Nifty moves more at the open and the close. Multipliers applied to the implied-volatility part of the range.
TOD_BUCKETS = [((9, 15), (9, 45), 1.45), ((9, 45), (10, 30), 1.15), ((10, 30), (13, 30), 0.85),
               ((13, 30), (14, 30), 1.00), ((14, 30), (15, 30), 1.20)]

# Event days: ranges get wider and direction calls need stronger evidence.
EVENT_SIGMA = {"none": 1.0, "moderate": 1.12, "high": 1.25}
EVENT_EDGE = {"none": 1.0, "moderate": 1.2, "high": 1.5}

# Index heavyweights (approximate Nifty weights) and Bank Nifty, used for the "market breadth" signal.
HEAVYWEIGHTS = {
    "HDFCBANK.NS": ("HDFC Bank", 0.13), "ICICIBANK.NS": ("ICICI Bank", 0.09), "RELIANCE.NS": ("Reliance", 0.08),
    "INFY.NS": ("Infosys", 0.05), "BHARTIARTL.NS": ("Bharti Airtel", 0.045), "LT.NS": ("L&T", 0.04),
    "ITC.NS": ("ITC", 0.035), "TCS.NS": ("TCS", 0.03), "AXISBANK.NS": ("Axis Bank", 0.03), "SBIN.NS": ("SBI", 0.03),
}
BANK_NIFTY = "^NSEBANK"
