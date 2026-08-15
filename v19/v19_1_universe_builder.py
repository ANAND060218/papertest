import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# A representative sample of the Nifty 100 universe
NIFTY_100_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", 
    "ITC.NS", "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", 
    "AXISBANK.NS", "M&M.NS", "MARUTI.NS", "HCLTECH.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "BAJAJFINSV.NS",
    "TATAMOTORS.NS", "WIPRO.NS", "NESTLEIND.NS", "GRASIM.NS", "TECHM.NS", "ONGC.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "HINDALCO.NS", "JSWSTEEL.NS", "DRREDDY.NS", "CIPLA.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "EICHERMOT.NS", 
    "HEROMOTOCO.NS", "COALINDIA.NS", "BPCL.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "LTIM.NS", "PIDILITIND.NS", "AMBUJACEM.NS", "SHREECEM.NS", "SIEMENS.NS", "TVSMOTOR.NS",
    "INDIGO.NS", "CHOLAFIN.NS", "GAIL.NS", "HAVELLS.NS", "DABUR.NS", "GODREJCP.NS",
    "TORNTPHARM.NS", "ZOMATO.NS", "TATAPOWER.NS", "ICICIPRULI.NS", "INDUSINDBK.NS", "PNB.NS",
    "BANKBARODA.NS", "CANBK.NS", "DLF.NS", "LODHA.NS", "TRENT.NS", "BOSCHLTD.NS",
    "CUMMINSIND.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "MUTHOOTFIN.NS",
    "PIIND.NS", "UPL.NS", "SRF.NS", "VOLTAS.NS", "COLPAL.NS", "MARICO.NS", "PAGEIND.NS",
    "MOTHERSUMI.NS", "JUBLFOOD.NS", "HAL.NS", "BEL.NS", "MCDOWELL-N.NS", "IRCTC.NS",
    "PFC.NS", "RECLTD.NS", "IOC.NS", "IGL.NS", "MGL.NS", "LICHSGFIN.NS", "TATACOMM.NS",
    "BIOCON.NS", "AUROPHARMA.NS", "LUPIN.NS", "BANDHANBNK.NS"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def build_dynamic_universe():
    print(f"Starting Dynamic Universe Builder for {len(NIFTY_100_SYMBOLS)} symbols...")
    
    # 1. Download last 10 years of data
    print("Downloading 10 years of data...")
    data = yf.download(NIFTY_100_SYMBOLS, start="2016-01-01", end="2026-08-15", group_by="ticker", auto_adjust=True)
    
    universe_metadata = {}
    valid_symbols = []
    
    stacked_data = []
    
    for sym in NIFTY_100_SYMBOLS:
        try:
            if sym not in data:
                print(f"  [WARN] No data for {sym}. Skipping.")
                continue
                
            df = data[sym].dropna(subset=['Close'])
            if len(df) < 500: # Need at least 2 years of history
                print(f"  [WARN] Insufficient history for {sym} ({len(df)} days). Skipping.")
                continue
                
            # Calculate 30-day Average Daily Volume in INR (Crores)
            # Volume is shares, Close is INR. 1 Crore = 10,000,000
            last_30 = df.tail(30)
            avg_daily_turnover_cr = (last_30['Volume'] * last_30['Close']).mean() / 10000000.0
            
            # Liquidity Filter: Must trade > ₹50 Crores per day
            if avg_daily_turnover_cr < 50.0:
                print(f"  [WARN] Illiquid: {sym} trades {avg_daily_turnover_cr:.1f} Cr/day. Skipping.")
                continue
                
            # Fetch Sector & Industry dynamically
            print(f"  Fetching metadata for {sym} (Turnover: {avg_daily_turnover_cr:.1f} Cr)...")
            info = yf.Ticker(sym).info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            universe_metadata[sym] = {
                'sector': sector,
                'industry': industry,
                'adv_cr': round(avg_daily_turnover_cr, 2)
            }
            
            valid_symbols.append(sym)
            
            # Prepare for stacking
            df_reset = df.reset_index()
            df_reset['Symbol'] = sym
            df_reset['Volume'] = df_reset['Volume'].fillna(0)
            stacked_data.append(df_reset[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']])
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {sym}: {e}")
            
    print(f"\n[SUCCESS] Extracted {len(valid_symbols)} highly liquid stocks.")
    
    # Save Metadata to JSON
    meta_path = os.path.join(DATA_DIR, "v19_1_dynamic_sectors.json")
    with open(meta_path, "w") as f:
        json.dump(universe_metadata, f, indent=4)
        
    # Save stacked dataset
    print("Stacking and saving historical dataset...")
    full_df = pd.concat(stacked_data, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv")
    full_df.to_csv(out_path, index=False)
    
    print(f"Saved Metadata -> {meta_path}")
    print(f"Saved History  -> {out_path}")
    
    # Print groupings
    sectors = {}
    for sym, meta in universe_metadata.items():
        s = meta['sector']
        if s not in sectors: sectors[s] = []
        sectors[s].append(sym)
        
    print("\nDynamic Sector Groupings:")
    for s, syms in sectors.items():
        print(f"  {s} ({len(syms)}): {syms}")

if __name__ == "__main__":
    build_dynamic_universe()
