"""
V8.3 -- Ingest Unseen Universe 5-Minute Intraday Dataset
Fetches 5-minute intraday data for 6 completely unseen liquid Indian stocks:
  1. AXISBANK.NS (Banking)
  2. KOTAKBANK.NS (Banking)
  3. BHARTIARTL.NS (Telecom)
  4. LT.NS (Capital Goods / Infra)
  5. WIPRO.NS (IT)
  6. ITC.NS (FMCG)

Stores in data/unseen_universe_5m.db for pure generalization testing with frozen model.
"""
import sys
import os
import sqlite3
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config


UNSEEN_UNIVERSE = [
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "WIPRO.NS",
    "ITC.NS"
]

UNSEEN_DB_PATH = os.path.join(config.DATA_DIR, "unseen_universe_5m.db")


def fetch_unseen_universe():
    print("=" * 80)
    print("V8.3: FETCHING UNSEEN 5-MINUTE INTRADAY UNIVERSE DATASET")
    print(f"Symbols: {UNSEEN_UNIVERSE}")
    print(f"Target DB: {UNSEEN_DB_PATH}")
    print("=" * 80)

    if os.path.exists(UNSEEN_DB_PATH):
        os.remove(UNSEEN_DB_PATH)

    conn = sqlite3.connect(UNSEEN_DB_PATH)
    all_dfs = []

    for sym in UNSEEN_UNIVERSE:
        print(f"Fetching 5m data for {sym}...")
        ticker = yf.Ticker(sym)
        df = ticker.history(period="max", interval="5m")

        if df.empty:
            print(f"  [WARNING] No data returned for {sym}")
            continue

        df.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['symbol'] = sym

        clean_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
        df_clean = df[[c for c in clean_cols if c in df.columns]].dropna()

        tbl_name = f"bars_5m_{sym.replace('.NS', '')}"
        df_clean.to_sql(tbl_name, conn, if_exists='replace', index=False)
        all_dfs.append(df_clean)
        print(f"  -> Saved {len(df_clean)} bars to table {tbl_name}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql("universe_intraday_5m", conn, if_exists='replace', index=False)
        print(f"\n[SUCCESS] Ingested total {len(combined_df)} 5-minute bars across {len(UNSEEN_UNIVERSE)} unseen stocks.")

    conn.close()


if __name__ == "__main__":
    fetch_unseen_universe()
