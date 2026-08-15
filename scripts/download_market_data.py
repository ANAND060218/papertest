import os
import sys
import pandas as pd
import yfinance as yf
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    print("Initializing Market Data Updater...")
    universe_file = os.path.join(DATA_DIR, "universe.txt")
    
    if not os.path.exists(universe_file):
        print("Error: universe.txt not found. Cannot download data.")
        sys.exit(1)
        
    with open(universe_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
        
    # Remove index symbols for yfinance downloading (we just want stocks)
    symbols = [s for s in symbols if not s.startswith('^')]
    
    print(f"Fetching 3 years of daily data for {len(symbols)} symbols...")
    
    # Download data
    data = yf.download(symbols, period="3y", interval="1d", group_by="ticker", auto_adjust=True, threads=True)
    
    # Convert multi-index to stacked format
    stacked_records = []
    
    # yfinance output format varies depending on number of tickers. 
    # If multiple tickers, it's a MultiIndex column: (Ticker, PriceType)
    for sym in symbols:
        if sym in data:
            df_sym = data[sym]
        else:
            print(f"Warning: No data for {sym}")
            continue
            
        if 'Close' not in df_sym.columns:
            continue
            
        df_sym = df_sym[['Close']].dropna()
        for date, row in df_sym.iterrows():
            stacked_records.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Symbol': sym,
                'Close': row['Close']
            })
            
    df_stacked = pd.DataFrame(stacked_records)
    df_stacked = df_stacked.sort_values(by=['Date', 'Symbol']).reset_index(drop=True)
    
    # Save as CSV
    csv_path = os.path.join(DATA_DIR, "nifty_10year_stacked.csv")
    df_stacked.to_csv(csv_path, index=False)
    print(f"Saved CSV with {len(df_stacked)} rows to {csv_path}")
    
    # Save to SQLite DB
    db_path = os.path.join(DATA_DIR, "nifty_10year_stock_market.db")
    conn = sqlite3.connect(db_path)
    df_stacked.to_sql("stock_daily_10y", conn, if_exists="replace", index=False)
    
    # Create indices for faster queries
    c = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_daily_10y (Date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON stock_daily_10y (Symbol)")
    conn.commit()
    conn.close()
    
    print(f"Saved Database to {db_path}")
    print("Market Data Update Complete!")

if __name__ == "__main__":
    main()
