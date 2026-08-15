import os
import sqlite3
import pandas as pd
import glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "v17_multi_year.db")
KAGGLE_DATA_DIR = r"C:\Users\anand\.cache\kagglehub\datasets\debashis74017\stock-market-data-nifty-100-stocks-5-min-data\versions\13"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intraday_5m (
            symbol TEXT,
            datetime TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, datetime)
        )
    ''')
    
    # Create index for fast time-series queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol_time ON intraday_5m(symbol, datetime)
    ''')
    conn.commit()
    return conn

def process_csv(file_path, conn):
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return
            
        # Extract symbol from filename (e.g., RELIANCE_5minute.csv -> RELIANCE.NS)
        filename = os.path.basename(file_path)
        base_symbol = filename.replace("_5minute.csv", "")
        
        # We need the '.NS' suffix for our system (except maybe for indices, but we'll add it to all for consistency, 
        # or map NIFTY 50 to ^NSEI)
        if base_symbol == "NIFTY 50":
            symbol = "^NSEI"
        elif base_symbol == "NIFTY BANK":
            symbol = "^NSEBANK"
        else:
            symbol = base_symbol + ".NS"
            
        # Add symbol column
        df['symbol'] = symbol
        
        # Rename date to datetime to match schema
        df = df.rename(columns={'date': 'datetime'})
        
        # Ensure correct column order
        cols = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[cols]
        
        # Insert into SQLite
        df.to_sql('intraday_5m', conn, if_exists='append', index=False)
        print(f"Inserted {len(df)} rows for {symbol}.")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    print(f"Initializing database at {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = init_db()
    print("Database initialized.")
    
    # Process all CSVs
    csv_files = glob.glob(os.path.join(KAGGLE_DATA_DIR, "*_5minute.csv"))
    
    # To save time for this test run, we can just process Nifty and a few top stocks
    # The user asked for: NIFTY 50, RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, BAJFINANCE, HCLTECH
    target_files = [
        "NIFTY 50_5minute.csv",
        "RELIANCE_5minute.csv",
        "TCS_5minute.csv",
        "INFY_5minute.csv",
        "HDFCBANK_5minute.csv",
        "ICICIBANK_5minute.csv",
        "SBIN_5minute.csv",
        "BHARTIARTL_5minute.csv",
        "BAJFINANCE_5minute.csv",
        "HCLTECH_5minute.csv"
    ]
    
    for filename in target_files:
        file_path = os.path.join(KAGGLE_DATA_DIR, filename)
        if os.path.exists(file_path):
            process_csv(file_path, conn)
        else:
            print(f"File not found: {file_path}")
            
    print("Ingestion complete.")
