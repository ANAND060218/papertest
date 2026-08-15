import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_oos_backtest():
    print("="*80)
    print("V19.1 DYNAMIC UNIVERSE OOS BACKTEST (2021-2025)")
    print("="*80)
    
    pairs_df = pd.read_csv(os.path.join(RESULTS_DIR, "v19_1_discovered_pairs.csv"))
    print(f"Loaded {len(pairs_df)} dynamically discovered pairs.")
    
    df_all = pd.read_csv(os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv"))
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    # OOS Period
    df_oos = df_all[(df_all['Date'] >= '2021-01-01') & (df_all['Date'] <= '2026-12-31')]
    price_matrix = df_oos.pivot(index='Date', columns='Symbol', values='Close').ffill()
    
    total_gross_pnl = 0
    total_costs = 0
    winning_pairs = 0
    
    print("\nRunning OOS vector backtest with 0.05% slippage...")
    
    for _, row in pairs_df.iterrows():
        sym_a = row['stock_a']
        sym_b = row['stock_b']
        hr = row['hedge_ratio']
        hl = row['half_life']
        
        if sym_a not in price_matrix.columns or sym_b not in price_matrix.columns:
            continue
            
        p_a = price_matrix[sym_a]
        p_b = price_matrix[sym_b]
        
        spread = p_a - (hr * p_b)
        
        # Rolling Z-Score (60 days)
        roll_mean = spread.rolling(window=60).mean()
        roll_std = spread.rolling(window=60).std()
        z_score = (spread - roll_mean) / roll_std
        
        # Trading Logic: Entry at +/- 2.0, Exit at 0
        position = 0
        gross_pnl = 0
        trades = 0
        
        entry_price_a = 0
        entry_price_b = 0
        
        for i in range(1, len(z_score)):
            z = z_score.iloc[i]
            prev_z = z_score.iloc[i-1]
            
            if pd.isna(z) or pd.isna(prev_z): continue
            
            curr_pa = p_a.iloc[i]
            curr_pb = p_b.iloc[i]
            
            # Entry Short Spread
            if z > 2.0 and position == 0:
                position = -1
                entry_price_a = curr_pa * 0.9995
                entry_price_b = curr_pb * 1.0005
                trades += 2
            # Entry Long Spread
            elif z < -2.0 and position == 0:
                position = 1
                entry_price_a = curr_pa * 1.0005
                entry_price_b = curr_pb * 0.9995
                trades += 2
            # Exit
            elif (position == -1 and z <= 0) or (position == 1 and z >= 0):
                if position == -1:
                    exit_price_a = curr_pa * 1.0005 # buy back a
                    exit_price_b = curr_pb * 0.9995 # sell back b
                    pnl_a = entry_price_a - exit_price_a
                    pnl_b = (exit_price_b - entry_price_b) * hr
                else:
                    exit_price_a = curr_pa * 0.9995 # sell a
                    exit_price_b = curr_pb * 1.0005 # buy back b
                    pnl_a = exit_price_a - entry_price_a
                    pnl_b = (entry_price_b - exit_price_b) * hr
                    
                gross_pnl += (pnl_a + pnl_b) * (100000 / curr_pa) # Normalized to 1L capital per leg
                position = 0
                trades += 2
                
        # Costs ~ 0.03% per leg on trades
        costs = trades * 100000 * 0.0003
        net_pnl = gross_pnl - costs
        
        total_gross_pnl += gross_pnl
        total_costs += costs
        
        if net_pnl > 0:
            winning_pairs += 1
            
        print(f"  {sym_a} / {sym_b}: Net Rs {net_pnl:,.0f} | Trades: {trades}")
        
    print("\n" + "="*80)
    print(f"OOS RESULTS (2021-2025):")
    print(f"Winning Pairs: {winning_pairs} / {len(pairs_df)}")
    print(f"Gross P&L: Rs {total_gross_pnl:,.0f}")
    print(f"Costs & Slippage: Rs {total_costs:,.0f}")
    print(f"Net Portfolio P&L: Rs {(total_gross_pnl - total_costs):,.0f}")
    
    if (total_gross_pnl - total_costs) > 0:
        print("\nVERDICT: ✅ SUCCESS. Edge holds in dynamically scaled universe.")
    else:
        print("\nVERDICT: ❌ FAILED. The edge decayed or costs consumed the spread in small caps.")

if __name__ == "__main__":
    run_oos_backtest()
