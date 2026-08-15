import pandas as pd
import numpy as np
import os
import json
import itertools
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_fdr_and_robustness():
    print("="*80)
    print("V19.1 ROBUSTNESS SUITE: FDR, Z-SCORE STABILITY, SLIPPAGE")
    print("="*80)

    # 1. Load Data
    with open(os.path.join(DATA_DIR, "v19_1_dynamic_sectors.json"), "r") as f:
        metadata = json.load(f)
        
    df_all = pd.read_csv(os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv"))
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    # IS: 2016-2020, OOS: 2021-2025
    df_train = df_all[(df_all['Date'] >= '2016-01-01') & (df_all['Date'] <= '2020-12-31')]
    df_oos = df_all[(df_all['Date'] >= '2021-01-01') & (df_all['Date'] <= '2026-12-31')]
    
    price_matrix_is = df_train.pivot(index='Date', columns='Symbol', values='Close').ffill()
    price_matrix_oos = df_oos.pivot(index='Date', columns='Symbol', values='Close').ffill()
    
    sectors = {}
    for sym, meta in metadata.items():
        s = meta['sector']
        if s not in sectors: sectors[s] = []
        sectors[s].append(sym)

    # ---------------------------------------------------------
    # TEST 1: FDR Multiple Testing Correction
    # ---------------------------------------------------------
    print("\n--- TEST 1: Benjamini-Hochberg FDR Correction (IS 2016-2020) ---")
    p_values = []
    pair_meta = []
    
    for sector, symbols in sectors.items():
        if len(symbols) < 2: continue
        combinations = list(itertools.combinations(symbols, 2))
        
        for sym_a, sym_b in combinations:
            if sym_a not in price_matrix_is.columns or sym_b not in price_matrix_is.columns: continue
            
            p_a = price_matrix_is[sym_a].dropna()
            p_b = price_matrix_is[sym_b].dropna()
            common_idx = p_a.index.intersection(p_b.index)
            if len(common_idx) < 500: continue
            
            p_a = p_a.loc[common_idx]
            p_b = p_b.loc[common_idx]
            
            score, pvalue, _ = coint(p_a, p_b)
            p_values.append(pvalue)
            pair_meta.append((sym_a, sym_b, sector, p_a, p_b))

    # Apply FDR correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    surviving_pairs = []
    for i, is_rejected in enumerate(reject):
        if is_rejected: # Null hypothesis rejected -> cointegrated
            sym_a, sym_b, sector, p_a, p_b = pair_meta[i]
            
            # Calculate Hedge Ratio & Half Life
            X = sm.add_constant(p_b)
            model = sm.OLS(p_a, X).fit()
            hedge_ratio = model.params[sym_b]
            spread = p_a - (hedge_ratio * p_b)
            
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
                surviving_pairs.append({
                    'stock_a': sym_a, 'stock_b': sym_b, 
                    'hedge_ratio': hedge_ratio, 'half_life': half_life
                })
                
    raw_significant = sum([1 for p in p_values if p < 0.05])
    print(f"Total pairs tested: {len(p_values)}")
    print(f"Pairs significant at raw p < 0.05: {raw_significant}")
    print(f"Pairs surviving FDR correction (alpha 0.05): {len(surviving_pairs)}")
    
    # ---------------------------------------------------------
    # TEST 2 & 3: Z-Score Stability & Slippage Stress (OOS)
    # ---------------------------------------------------------
    print("\n--- TEST 2 & 3: Z-Score Stability & Slippage Stress (OOS 2021-2025) ---")
    
    z_thresholds = [1.5, 2.0, 2.5, 3.0]
    slippage_levels = [0.0, 0.0005, 0.001, 0.0015] # 0bps, 5bps, 10bps, 15bps
    
    results = []
    
    for z_thresh in z_thresholds:
        for slip in slippage_levels:
            total_net_pnl = 0
            for row in surviving_pairs:
                sym_a, sym_b, hr = row['stock_a'], row['stock_b'], row['hedge_ratio']
                if sym_a not in price_matrix_oos.columns or sym_b not in price_matrix_oos.columns: continue
                
                p_a = price_matrix_oos[sym_a]
                p_b = price_matrix_oos[sym_b]
                spread = p_a - (hr * p_b)
                
                roll_mean = spread.rolling(window=60).mean()
                roll_std = spread.rolling(window=60).std()
                z_score = (spread - roll_mean) / roll_std
                
                position = 0
                gross_pnl = 0
                trades = 0
                entry_a = entry_b = 0
                
                for i in range(1, len(z_score)):
                    z = z_score.iloc[i]
                    if pd.isna(z): continue
                    
                    ca = p_a.iloc[i]
                    cb = p_b.iloc[i]
                    
                    if z > z_thresh and position == 0:
                        position = -1
                        entry_a = ca * (1 - slip) # sell bid
                        entry_b = cb * (1 + slip) # buy ask
                        trades += 2
                    elif z < -z_thresh and position == 0:
                        position = 1
                        entry_a = ca * (1 + slip)
                        entry_b = cb * (1 - slip)
                        trades += 2
                    elif (position == -1 and z <= 0) or (position == 1 and z >= 0):
                        if position == -1:
                            exit_a = ca * (1 + slip)
                            exit_b = cb * (1 - slip)
                            pnl_a = entry_a - exit_a
                            pnl_b = (exit_b - entry_b) * hr
                        else:
                            exit_a = ca * (1 - slip)
                            exit_b = cb * (1 + slip)
                            pnl_a = exit_a - entry_a
                            pnl_b = (entry_b - exit_b) * hr
                            
                        gross_pnl += (pnl_a + pnl_b) * (100000 / ca)
                        position = 0
                        trades += 2
                        
                costs = trades * 100000 * 0.0003
                total_net_pnl += (gross_pnl - costs)
                
            results.append({'Z': z_thresh, 'Slippage': slip*10000, 'Net_PnL': total_net_pnl})
            
    df_res = pd.DataFrame(results)
    print("\nOOS Net P&L (₹) across Z-Scores and Slippage (bps):")
    pivot = df_res.pivot(index='Z', columns='Slippage', values='Net_PnL')
    print(pivot.applymap(lambda x: f"{x:,.0f}"))
    
    # Save the FDR surviving pairs for production
    if len(surviving_pairs) > 0:
        df_survive = pd.DataFrame(surviving_pairs)
        out_path = os.path.join(RESULTS_DIR, "v19_1_fdr_surviving_pairs.csv")
        df_survive.to_csv(out_path, index=False)
        print(f"\nSaved {len(surviving_pairs)} FDR-corrected pairs to {out_path}")

if __name__ == "__main__":
    run_fdr_and_robustness()
