"""
V1 -- Data Manager
Unified data loading, validation, and storage for intraday research.

Responsibilities:
1. Fetch 5-minute intraday data from yfinance (max ~60 days)
2. Fetch daily data from existing SQLite DB (10 years)
3. Store intraday data in SQLite + Parquet for persistence
4. Validate: no gaps during market hours, correct timestamps, volume present
5. Build daily context (previous day's OHLCV) needed for V2 setups
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DataManager:
    """
    Unified data loader for the trading research platform.
    Handles both daily (10yr DB) and intraday (5m yfinance) data.
    """

    def __init__(self):
        self.daily_db = config.DB_PATH_DAILY
        self.intraday_db = config.DB_PATH_INTRADAY

    # ----------------------------------------------------------
    # DAILY DATA (from existing 10-year DB)
    # ----------------------------------------------------------
    def load_daily(self, symbol, start_date=None, end_date=None):
        """
        Load daily OHLCV from the existing 10-year SQLite database.
        Returns a clean DataFrame sorted by Date.
        """
        if not os.path.exists(self.daily_db):
            print(f"[ERROR] Daily DB not found: {self.daily_db}")
            return None

        conn = sqlite3.connect(self.daily_db)
        query = "SELECT * FROM stock_daily_10y WHERE Symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND Date >= ?"
            params.append(str(start_date))
        if end_date:
            query += " AND Date <= ?"
            params.append(str(end_date))

        query += " ORDER BY Date ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            print(f"[WARNING] No daily data found for {symbol}")
            return None

        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)
        return df

    # ----------------------------------------------------------
    # INTRADAY DATA (5-minute from yfinance)
    # ----------------------------------------------------------
    def fetch_intraday_from_yfinance(self, symbol, days_back=59):
        """
        Fetch 5-minute intraday data from yfinance.
        yfinance limit: max ~60 days for 5m interval.
        Returns raw DataFrame with IST timestamps.
        """
        if not YFINANCE_AVAILABLE:
            print("[ERROR] yfinance not installed. Run: pip install yfinance")
            return None

        print(f"[V1] Fetching {days_back} days of 5m data for {symbol}...")
        ticker = yf.Ticker(symbol)

        # yfinance uses period or start/end
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        df = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=config.INTRADAY_INTERVAL,
            auto_adjust=False
        )

        if df is None or df.empty:
            print(f"[ERROR] No intraday data returned for {symbol}")
            return None

        # Reset index to get Datetime as column
        df = df.reset_index()

        # Rename 'Datetime' to 'Date' if present
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})

        # Ensure timezone-aware -> convert to IST then make naive
        if df['Date'].dt.tz is not None:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            df['Date'] = df['Date'].dt.tz_convert(ist).dt.tz_localize(None)

        df['Symbol'] = symbol

        # Keep only OHLCV + Symbol
        cols_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
        available = [c for c in cols_keep if c in df.columns]
        df = df[available]

        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"[V1] Fetched {len(df)} rows for {symbol}")
        print(f"     Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df

    def validate_intraday(self, df, symbol=None):
        """
        Validate intraday 5-minute data:
        1. Timestamps are within market hours (09:15 - 15:30 IST)
        2. No duplicate timestamps
        3. Volume is present and non-zero for most bars
        4. OHLC relationships are valid (Low <= Open/Close <= High)
        5. Report gaps and quality metrics

        Returns: (clean_df, validation_report dict)
        """
        report = {
            'symbol': symbol or 'UNKNOWN',
            'total_raw_rows': len(df),
            'issues': []
        }

        if df.empty:
            report['issues'].append("Empty DataFrame")
            return df, report

        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)

        # 1. Remove rows outside market hours
        market_open = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
        market_close = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)

        df['time'] = df['Date'].dt.time
        outside_hours = df[(df['time'] < market_open) | (df['time'] > market_close)]
        if len(outside_hours) > 0:
            report['issues'].append(f"Removed {len(outside_hours)} rows outside market hours")
            df = df[(df['time'] >= market_open) & (df['time'] <= market_close)]

        df = df.drop(columns=['time'])

        # 2. Remove duplicates
        dups = df.duplicated(subset=['Date'], keep='first')
        if dups.sum() > 0:
            report['issues'].append(f"Removed {dups.sum()} duplicate timestamps")
            df = df[~dups]

        # 3. Check volume
        zero_vol = (df['Volume'] == 0).sum()
        null_vol = df['Volume'].isna().sum()
        report['zero_volume_bars'] = int(zero_vol)
        report['null_volume_bars'] = int(null_vol)
        if zero_vol > len(df) * 0.1:
            report['issues'].append(f"WARNING: {zero_vol} bars ({zero_vol/len(df)*100:.1f}%) have zero volume")

        # 4. OHLC sanity
        invalid_ohlc = df[
            (df['Low'] > df['Open']) | (df['Low'] > df['Close']) |
            (df['High'] < df['Open']) | (df['High'] < df['Close']) |
            (df['Low'] > df['High'])
        ]
        if len(invalid_ohlc) > 0:
            report['issues'].append(f"Removed {len(invalid_ohlc)} rows with invalid OHLC (Low > High etc)")
            df = df.drop(invalid_ohlc.index)

        # 5. Per-day bar count analysis
        df['trade_date'] = df['Date'].dt.date
        bars_per_day = df.groupby('trade_date').size()
        expected_bars = 75  # 6.25 hours x 12 bars/hour = 75 bars per day for 5m

        report['trading_days'] = len(bars_per_day)
        report['avg_bars_per_day'] = float(bars_per_day.mean())
        report['min_bars_per_day'] = int(bars_per_day.min())
        report['max_bars_per_day'] = int(bars_per_day.max())

        short_days = bars_per_day[bars_per_day < expected_bars * 0.7]
        if len(short_days) > 0:
            report['issues'].append(
                f"{len(short_days)} days have <70% expected bars (may be partial/holiday sessions)"
            )

        df = df.drop(columns=['trade_date'])

        # Final stats
        report['clean_rows'] = len(df)
        report['date_range_start'] = str(df['Date'].min())
        report['date_range_end'] = str(df['Date'].max())

        return df.reset_index(drop=True), report

    def save_intraday_to_db(self, df, table_name="intraday_5m"):
        """
        Save validated intraday data to SQLite for persistence.
        """
        conn = sqlite3.connect(self.intraday_db)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"[V1] Saved {len(df)} rows to {self.intraday_db} (table: {table_name})")

    def save_intraday_to_parquet(self, df):
        """
        Save validated intraday data to Parquet for fast loading.
        """
        df.to_parquet(config.PARQUET_INTRADAY, index=False)
        print(f"[V1] Saved {len(df)} rows to {config.PARQUET_INTRADAY}")

    def load_intraday(self, symbol=None):
        """
        Load previously saved intraday data.
        Tries Parquet first (faster), falls back to SQLite.
        """
        if os.path.exists(config.PARQUET_INTRADAY):
            df = pd.read_parquet(config.PARQUET_INTRADAY)
            df['Date'] = pd.to_datetime(df['Date'])
            if symbol and 'Symbol' in df.columns:
                df = df[df['Symbol'] == symbol]
            return df

        if os.path.exists(self.intraday_db):
            conn = sqlite3.connect(self.intraday_db)
            query = "SELECT * FROM intraday_5m"
            if symbol:
                query += f" WHERE Symbol = '{symbol}'"
            query += " ORDER BY Date ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()
            df['Date'] = pd.to_datetime(df['Date'])
            return df

        return None

    # ----------------------------------------------------------
    # DAILY CONTEXT (needed for V2: previous day's High/Low/Close)
    # ----------------------------------------------------------
    def build_daily_context(self, intraday_df):
        """
        From intraday data, compute the previous trading day's OHLCV.
        This is needed for the Previous-Day High Breakout setup.

        Returns DataFrame with columns:
        trade_date, prev_day_open, prev_day_high, prev_day_low, prev_day_close, prev_day_volume
        """
        df = intraday_df.copy()
        df['trade_date'] = df['Date'].dt.date

        # Group by trade_date to get daily OHLCV from intraday bars
        daily = df.groupby('trade_date').agg(
            day_open=('Open', 'first'),
            day_high=('High', 'max'),
            day_low=('Low', 'min'),
            day_close=('Close', 'last'),
            day_volume=('Volume', 'sum')
        ).reset_index()

        daily = daily.sort_values('trade_date').reset_index(drop=True)

        # Shift to get PREVIOUS day values
        daily['prev_day_open'] = daily['day_open'].shift(1)
        daily['prev_day_high'] = daily['day_high'].shift(1)
        daily['prev_day_low'] = daily['day_low'].shift(1)
        daily['prev_day_close'] = daily['day_close'].shift(1)
        daily['prev_day_volume'] = daily['day_volume'].shift(1)

        # Drop first row (no previous day)
        daily = daily.dropna().reset_index(drop=True)

        return daily


# ============================================================
# V1.2 -- Fetch and validate 5-minute data
# ============================================================
if __name__ == "__main__":
    dm = DataManager()

    symbol = config.PRIMARY_SYMBOL
    print("=" * 70)
    print(f"V1.2 -- Fetching 5-minute intraday data for {symbol}")
    print("=" * 70)

    # Step 1: Fetch from yfinance
    raw_df = dm.fetch_intraday_from_yfinance(symbol, days_back=config.YFINANCE_MAX_5M_DAYS)
    if raw_df is None:
        print("[FATAL] Could not fetch intraday data. Exiting.")
        sys.exit(1)

    # Step 2: Validate
    clean_df, report = dm.validate_intraday(raw_df, symbol=symbol)

    print(f"\n{'=' * 70}")
    print("V1 VALIDATION REPORT")
    print(f"{'=' * 70}")
    for key, val in report.items():
        if key == 'issues':
            print(f"\n  ISSUES ({len(val)}):")
            for issue in val:
                print(f"    - {issue}")
        else:
            print(f"  {key}: {val}")

    # Step 3: Save
    if len(clean_df) > 0:
        dm.save_intraday_to_db(clean_df)
        dm.save_intraday_to_parquet(clean_df)

        # Step 4: Show sample
        print(f"\n{'=' * 70}")
        print("SAMPLE CLEAN DATA (first 10 rows)")
        print(f"{'=' * 70}")
        print(clean_df.head(10).to_string(index=False))

        print(f"\n{'=' * 70}")
        print("SAMPLE CLEAN DATA (last 10 rows)")
        print(f"{'=' * 70}")
        print(clean_df.tail(10).to_string(index=False))

        # Step 5: Build and show daily context
        daily_ctx = dm.build_daily_context(clean_df)
        print(f"\n{'=' * 70}")
        print("DAILY CONTEXT (prev-day OHLCV for V2 setup detection)")
        print(f"{'=' * 70}")
        print(daily_ctx.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("V1 COMPLETE -- Data is ready for V2")
    print(f"{'=' * 70}")
