import pandas as pd
import numpy as np
import os
import json
import itertools
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def run_dynamic_cointegration_scan():
    print("="*100)
    print("V19.1 DYNAMIC COINTEGRATION SCANNER (ENGLE-GRANGER)")
    print("Universe: Liquid Nifty 100 Stocks (Turnover > 50Cr/day)")
    print("="*100)

    # 1. Load Data
    print("Loading dynamic sector mappings and historical data...")
    with open(os.path.join(DATA_DIR, "v19_1_dynamic_sectors.json"), "r") as f:
        metadata = json.load(f)
        
    df_all = pd.read_csv(os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv"))
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    # Filter for In-Sample period (2016-2020) for initial discovery
    df_train = df_all[(df_all['Date'] >= '2016-01-01') & (df_all['Date'] <= '2020-12-31')]
    price_matrix = df_train.pivot(index='Date', columns='Symbol', values='Close').ffill()
    
    # Group by Sector
    sectors = {}
    for sym, meta in metadata.items():
        s = meta['sector']
        if s not in sectors: sectors[s] = []
        sectors[s].append(sym)
        
    print(f"Loaded {len(metadata)} stocks across {len(sectors)} dynamic sectors.")
    
    # 2. Build Combinations & Run Scan
    all_pairs = []
    total_combinations = 0
    
    print("\nScanning sectors for cointegration (p-value < 0.05)...")
    for sector, symbols in sectors.items():
        if len(symbols) < 2: continue
        
        combinations = list(itertools.combinations(symbols, 2))
        total_combinations += len(combinations)
        
        valid_in_sector = 0
        for sym_a, sym_b in combinations:
            if sym_a not in price_matrix.columns or sym_b not in price_matrix.columns: continue
            
            p_a = price_matrix[sym_a].dropna()
            p_b = price_matrix[sym_b].dropna()
            
            # Align dates
            common_idx = p_a.index.intersection(p_b.index)
            if len(common_idx) < 500: continue # Need at least 2 years overlap
            
            p_a = p_a.loc[common_idx]
            p_b = p_b.loc[common_idx]
            
            # Engle-Granger Test
            score, pvalue, _ = coint(p_a, p_b)
            
            if pvalue < 0.05:
                # Calculate Hedge Ratio & Half Life
                X = sm.add_constant(p_b)
                model = sm.OLS(p_a, X).fit()
                hedge_ratio = model.params[sym_b]
                spread = p_a - (hedge_ratio * p_b)
                
                # Half-life using Ornstein-Uhlenbeck
                spread_lag = spread.shift(1).dropna()
                spread_diff = spread.diff().dropna()
                if len(spread_lag) == len(spread_diff):
                    X_hl = sm.add_constant(spread_lag)
                    model_hl = sm.OLS(spread_diff, X_hl).fit()
                    lam = model_hl.params.iloc[1]
                    half_life = -np.log(2) / lam if lam < 0 else np.nan
                else:
                    half_life = np.nan
                    
                if pd.notna(half_life) and 1 <= half_life <= 90:
                    valid_in_sector += 1
                    all_pairs.append({
                        'pair_id': f"{sym_a}_{sym_b}",
                        'stock_a': sym_a,
                        'stock_b': sym_b,
                        'sector': sector,
                        'p_value': round(pvalue, 4),
                        'hedge_ratio': round(hedge_ratio, 4),
                        'half_life': round(half_life, 1)
                    })
                    
        print(f"  [{sector}] Scanned {len(combinations)} pairs -> Found {valid_in_sector} highly cointegrated pairs")
        
    print(f"\n[SCAN COMPLETE] Tested {total_combinations} total pairs.")
    print(f"Discovered {len(all_pairs)} statistically valid pairs for V19.1 Universe.")
    
    df_pairs = pd.DataFrame(all_pairs).sort_values('p_value')
    out_path = os.path.join(RESULTS_DIR, "v19_1_discovered_pairs.csv")
    df_pairs.to_csv(out_path, index=False)
    print(f"Saved initial pairs -> {out_path}")
    
if __name__ == "__main__":
    run_dynamic_cointegration_scan()
