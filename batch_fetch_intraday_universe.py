"""
V6.5.1 -- Multi-Symbol Intraday Universe Data Ingestion
Fetches the maximum available 5-minute intraday historical data for the full trading universe:
  1. RELIANCE.NS
  2. TCS.NS
  3. INFY.NS
  4. HDFCBANK.NS
  5. ICICIBANK.NS
  6. SBIN.NS
  7. ^NSEI (Benchmark Nifty 50 Index)

Features:
  - Validates market hours (09:15 - 15:30 IST)
  - Timezone normalization (UTC -> Asia/Kolkata -> naive)
  - Removes zero-volume and broken candles
  - Stores into high-performance SQLite database & Parquet file
"""
import sys
import os
import time
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.data_manager import DataManager

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


INTRADAY_UNIVERSE_DB = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")
INTRADAY_UNIVERSE_PARQUET = os.path.join(config.DATA_DIR, "intraday_universe_5m.parquet")


def fetch_and_store_intraday_universe(symbols=None, days_back=59):
    if not YFINANCE_AVAILABLE:
        print("[ERROR] yfinance not available. Run: pip install yfinance")
        return None

    target_symbols = symbols or (config.UNIVERSE + ["^NSEI"])
    dm = DataManager()

    print("=" * 90)
    print(f"V6.5.1 INGESTING 5-MINUTE INTRADAY UNIVERSE ({len(target_symbols)} SYMBOLS)")
    print(f"Timeframe: 5-minute | Period: Last {days_back} days")
    print("=" * 90)

    all_dfs = []
    summary_report = []

    conn = sqlite3.connect(INTRADAY_UNIVERSE_DB)

    for sym in target_symbols:
        print(f"\nFetching {sym} (5m candles)...")
        try:
            raw_df = dm.fetch_intraday_from_yfinance(sym, days_back=days_back)
            if raw_df is None or raw_df.empty:
                print(f"  [WARNING] No data returned for {sym}")
                continue

            clean_df, report = dm.validate_intraday(raw_df, symbol=sym)

            # Store in symbol-specific table and collect for master table
            table_name = f"bars_5m_{sym.replace('^', 'IDX_').replace('.NS', '')}"
            clean_df.to_sql(table_name, conn, if_exists='replace', index=False)

            all_dfs.append(clean_df)

            summary_report.append({
                'Symbol': sym,
                'RawRows': report['total_raw_rows'],
                'CleanRows': report['clean_rows'],
                'TradingDays': report['trading_days'],
                'AvgBarsPerDay': round(report['avg_bars_per_day'], 1),
                'StartDate': str(report['date_range_start'])[:10],
                'EndDate': str(report['date_range_end'])[:10],
            })

            print(f"  Saved {len(clean_df)} bars for {sym} to table {table_name}")

        except Exception as e:
            print(f"  [ERROR] Failed to fetch {sym}: {str(e)}")

        time.sleep(0.5)

    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.sort_values(['Symbol', 'Date'], inplace=True)
        master_df.to_sql("universe_intraday_5m", conn, if_exists='replace', index=False)
        master_df.to_parquet(INTRADAY_UNIVERSE_PARQUET, index=False)
        print(f"\n[SUCCESS] Master universe table created: {len(master_df):,} total 5m bars.")
        print(f"Saved to SQLite: {INTRADAY_UNIVERSE_DB}")
        print(f"Saved to Parquet: {INTRADAY_UNIVERSE_PARQUET}")

    conn.close()

    summary_df = pd.DataFrame(summary_report)
    print("\n" + "=" * 90)
    print("V6.5.1 INTRADAY DATASET INGESTION SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    fetch_and_store_intraday_universe()
