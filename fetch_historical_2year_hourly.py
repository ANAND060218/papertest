"""
V8.4 Data Ingestion: 2-Year Historical Hourly Dataset (2023 - 2026)
Fetches 730 days of 1-hour candles for the core universe (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, and ^NSEI).

Enables long-horizon temporal stress testing on market data that pre-dates the 5m training set.
"""
import sys
import os
import sqlite3
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config


HOURLY_DB_PATH = os.path.join(config.DATA_DIR, "historical_2year_hourly.db")
HOURLY_SYMBOLS = config.UNIVERSE + ["^NSEI"]


def fetch_2year_hourly_dataset():
    print("=" * 80)
    print("V8.4: FETCHING 2-YEAR HISTORICAL HOURLY INTRADAY DATASET (2023-2026)")
    print(f"Symbols: {HOURLY_SYMBOLS}")
    print(f"Target DB: {HOURLY_DB_PATH}")
    print("=" * 80)

    if os.path.exists(HOURLY_DB_PATH):
        os.remove(HOURLY_DB_PATH)

    conn = sqlite3.connect(HOURLY_DB_PATH)
    all_dfs = []

    for sym in HOURLY_SYMBOLS:
        print(f"Fetching 1h data for {sym} (period=730d)...")
        ticker = yf.Ticker(sym)
        df = ticker.history(period="730d", interval="1h")

        if df.empty:
            print(f"  [WARNING] No hourly data for {sym}")
            continue

        df.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['symbol'] = sym

        clean_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
        df_clean = df[[c for c in clean_cols if c in df.columns]].dropna()

        tbl_name = f"bars_1h_{sym.replace('^', 'IDX_').replace('.NS', '')}"
        df_clean.to_sql(tbl_name, conn, if_exists='replace', index=False)
        all_dfs.append(df_clean)
        print(f"  -> Saved {len(df_clean)} hourly bars to table {tbl_name} ({df_clean['Date'].min()[:10]} to {df_clean['Date'].max()[:10]})")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql("universe_hourly_2year", conn, if_exists='replace', index=False)
        print(f"\n[SUCCESS] Ingested total {len(combined_df)} hourly bars across {len(HOURLY_SYMBOLS)} assets.")

    conn.close()


if __name__ == "__main__":
    fetch_2year_hourly_dataset()
