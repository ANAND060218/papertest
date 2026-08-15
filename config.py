"""
Central Configuration for the Intraday Trading Research Platform.
All constants, symbols, timeframes, cost assumptions, and risk parameters live here.

RULE: Change parameters HERE, not in individual modules.
"""

# ============================================================
# UNIVERSE
# ============================================================
# Start with RELIANCE only (V2). Expand in V6.
PRIMARY_SYMBOL = "RELIANCE.NS"

# V6 expansion universe
UNIVERSE = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
]

# Index tickers for regime detection
INDICES = ["^NSEI", "^NSEBANK"]

# ============================================================
# DATA SETTINGS
# ============================================================
INTRADAY_INTERVAL = "5m"        # 5-minute candles
DAILY_INTERVAL = "1d"
YFINANCE_MAX_5M_DAYS = 59       # yfinance limit for 5m data

# Market session (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ============================================================
# V2 SETUP PARAMETERS: Previous-Day High Breakout
# ============================================================
# These are FIXED before looking at results.
# Do NOT change after backtesting.
BREAKOUT_VOLUME_RATIO_MIN = 1.5     # Volume must be 1.5x 20-bar average
BREAKOUT_TARGET_PCT = 1.0           # +1.0% from entry
BREAKOUT_STOP_PCT = 0.5             # -0.5% from entry
BREAKOUT_MAX_HOLDING_BARS = 30      # Max 30 x 5min = 2.5 hours
BREAKOUT_MIN_TIME_HOUR = 9          # Earliest entry time (hour)
BREAKOUT_MIN_TIME_MINUTE = 30       # Earliest entry time (minute) -- skip first 15min
BREAKOUT_MAX_TIME_HOUR = 14         # Latest entry time (hour)
BREAKOUT_MAX_TIME_MINUTE = 30       # Latest entry time (minute) -- no new trades after 14:30

# ============================================================
# COST MODEL (Indian Intraday Equity)
# ============================================================
BROKERAGE_PCT = 0.03 / 100         # 0.03% per side (Zerodha-like)
STT_SELL_PCT = 0.025 / 100         # 0.025% on sell side
STAMP_DUTY_BUY_PCT = 0.003 / 100   # 0.003% on buy side
GST_ON_BROKERAGE_PCT = 18 / 100    # 18% GST on brokerage
EXCHANGE_TXN_PCT = 0.00345 / 100   # NSE transaction charge
SEBI_TURNOVER_PCT = 0.0001 / 100   # SEBI turnover fee

SLIPPAGE_PCT = 0.05 / 100          # 0.05% adverse movement on market orders
SPREAD_PCT = 0.02 / 100            # 0.02% for liquid NIFTY stocks

# ============================================================
# RISK PARAMETERS
# ============================================================
INITIAL_CAPITAL = 100_000           # Rs. 1,00,000 paper trading capital
MAX_POSITION_PCT = 0.10            # Max 10% of capital per trade
MAX_DAILY_LOSS_PCT = 0.02          # Stop trading if down 2% in a day
MAX_DRAWDOWN_PCT = 0.10            # Stop system if drawdown hits 10%
MAX_CONCURRENT_POSITIONS = 3       # Max simultaneous open positions

# ============================================================
# DATABASE PATHS
# ============================================================
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
DB_PATH_DAILY = os.path.join(DATA_DIR, "nifty_10year_stock_market.db")
DB_PATH_INTRADAY = os.path.join(DATA_DIR, "intraday_5m.db")
PARQUET_INTRADAY = os.path.join(DATA_DIR, "intraday_5m.parquet")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(_BASE_DIR, "models", "saved"), exist_ok=True)
