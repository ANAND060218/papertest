import os
import time
import sqlite3
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


# Sample list of major NSE NIFTY Companies
NIFTY_TOP_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "ULTRACEMCO.NS", "TATASTEEL.NS",
    "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "GRASIM.NS", "HCLTECH.NS",
    "CIPLA.NS", "SBILIFE.NS", "DRREDDY.NS", "WIPRO.NS", "EICHERMOT.NS",
    "BPCL.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "^NSEI", "^NSEBANK"
]


def download_10_years_data(ticker_list=NIFTY_TOP_TICKERS, period="10y"):
    """
    Downloads 10+ years of historical daily data in parallel for a list of stocks.
    Stores data in Parquet, CSV, and SQLite database formats for ML models.
    """
    ensure_data_dir()
    start_time = time.time()

    print("==================================================================================")
    print(f" BATCH DOWNLOADING 10+ YEARS HISTORICAL DATA ({len(ticker_list)} STOCKS)")
    print("==================================================================================\n")

    # Fast multi-threaded batch download using yfinance
    print(f"[1/3] Downloading 10 years of daily OHLCV data via multi-threading...")
    df_raw = yf.download(ticker_list, period=period, interval="1d", threads=True)
    elapsed_download = time.time() - start_time
    print(f"[SUCCESS] Download Completed in {elapsed_download:.2f} seconds!\n")

    print("[2/3] Processing and Storing Data...")

    # Process each ticker into clean flat DataFrame
    records = []
    for ticker in ticker_list:
        try:
            if isinstance(df_raw.columns, pd.MultiIndex):
                # Extract columns for specific ticker
                sub_df = pd.DataFrame()
                for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if (col_type, ticker) in df_raw.columns:
                        sub_df[col_type] = df_raw[(col_type, ticker)]
            else:
                sub_df = df_raw.copy()

            if not sub_df.empty:
                sub_df.dropna(subset=['Close'], inplace=True)
                sub_df['Symbol'] = ticker
                sub_df.reset_index(inplace=True)
                records.append(sub_df)
        except Exception:
            pass

    full_stacked_df = pd.concat(records, ignore_index=True)
    full_stacked_df['Date'] = pd.to_datetime(full_stacked_df['Date']).dt.tz_localize(None)

    # 1. Save Compressed Parquet
    parquet_path = os.path.join(DATA_DIR, "nifty_10year_historical.parquet")
    full_stacked_df.to_parquet(parquet_path, index=False)
    parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)

    # 2. Save SQLite DB
    db_path = os.path.join(DATA_DIR, "nifty_10year_stock_market.db")
    conn = sqlite3.connect(db_path)
    full_stacked_df.to_sql("stock_daily_10y", conn, if_exists="replace", index=False)
    conn.close()
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # 3. Save CSV
    csv_path = os.path.join(DATA_DIR, "nifty_10year_stacked.csv")
    full_stacked_df.to_csv(csv_path, index=False)
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    total_time = time.time() - start_time

    print("==================================================================================")
    print(" STORAGE & TIMING SUMMARY")
    print("==================================================================================")
    print(f"Total Companies Downloaded : {len(ticker_list)}")
    print(f"Time Range                 : 10+ Years (2016 - 2026)")
    print(f"Total Clean Rows Generated : {len(full_stacked_df):,}")
    print(f"Download & Processing Time : {total_time:.2f} seconds")
    print("-" * 60)
    print(f"Storage Option 1 (Parquet) : {parquet_path} ({parquet_size_mb:.2f} MB)")
    print(f"Storage Option 2 (SQLite DB): {db_path} ({db_size_mb:.2f} MB)")
    print(f"Storage Option 3 (CSV File) : {csv_path} ({csv_size_mb:.2f} MB)")
    print("==================================================================================")


if __name__ == "__main__":
    download_10_years_data()
