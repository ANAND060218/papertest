import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def build_ml_dataset():
    print("="*80)
    print("V21-A MACHINE LEARNING: FEATURE ENGINEERING")
    print("="*80)
    
    pairs_df = pd.read_csv(os.path.join(RESULTS_DIR, "v19_1_discovered_pairs.csv"))
    print(f"Loading data for {len(pairs_df)} dynamic pairs...")
    
    df_all = pd.read_csv(os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv"))
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    price_matrix = df_all.pivot(index='Date', columns='Symbol', values='Close').ffill().dropna(how='all')
    
    ml_records = []
    
    for idx, row in pairs_df.iterrows():
        sym_a = row['stock_a']
        sym_b = row['stock_b']
        
        print(f"  Processing features for {sym_a} / {sym_b}...")
        
        if sym_a not in price_matrix.columns or sym_b not in price_matrix.columns:
            continue
            
        p_a = price_matrix[sym_a]
        p_b = price_matrix[sym_b]
        
        # Calculate Rolling Features
        # We need rolling Hedge Ratio. This is computationally heavy to do daily with OLS over 10 years (2500 days).
        # We will approximate by recalculating every 5 days for speed in feature engineering.
        
        spreads = pd.Series(index=p_a.index, dtype=float)
        hedge_ratios = pd.Series(index=p_a.index, dtype=float)
        
        # 1. Rolling Hedge Ratio (250 day window)
        valid_mask = p_a.notna() & p_b.notna()
        p_a_valid = p_a[valid_mask]
        p_b_valid = p_b[valid_mask]
        
        for i in range(250, len(p_a_valid), 5):
            window_a = p_a_valid.iloc[i-250:i]
            window_b = p_b_valid.iloc[i-250:i]
            if window_a.isna().any() or window_b.isna().any(): continue
            X = sm.add_constant(window_b)
            model = sm.OLS(window_a, X).fit()
            hr = model.params.iloc[1]
            
            # Map back to original index roughly
            idx_date = p_a_valid.index[i]
            loc_idx = p_a.index.get_loc(idx_date)
            hedge_ratios.iloc[loc_idx:min(loc_idx+5, len(p_a))] = hr
            
        hedge_ratios.ffill(inplace=True)
        spread = p_a - (hedge_ratios * p_b)
        
        # 2. Rolling Z-Score (60 day window)
        roll_mean = spread.rolling(window=60).mean()
        roll_std = spread.rolling(window=60).std()
        z_score = (spread - roll_mean) / roll_std
        
        # 3. Market Regime (NIFTY trend)
        # We approximate market trend using RELIANCE.NS since it's highly correlated to NIFTY, 
        # or we just use the price of A + B as a proxy for sector trend.
        sector_trend = (p_a + p_b).pct_change(20) # 20-day momentum
        
        # 4. Target Variable (Y): Forward 10-day P&L of the spread
        # If we short the spread today, what is the return in 10 days?
        # Short Spread = Sell A, Buy B.
        # Long Spread = Buy A, Sell B.
        # We define Y as: if we act on the Z-score, what is the net return?
        # Target = (Spread_t - Spread_{t+10}) for short, (Spread_{t+10} - Spread_t) for long.
        # To make it universally predictable, we predict the continuous variable: Forward 10-day Delta of Spread
        fwd_10d_spread = spread.shift(-10)
        fwd_10d_delta = fwd_10d_spread - spread 
        
        # Assemble DataFrame
        pair_df = pd.DataFrame({
            'Date': p_a.index,
            'Pair': f"{sym_a}_{sym_b}",
            'Stock_A': sym_a,
            'Stock_B': sym_b,
            'Price_A': p_a,
            'Price_B': p_b,
            'Hedge_Ratio': hedge_ratios,
            'Spread': spread,
            'Z_Score': z_score,
            'Sector_Mom_20d': sector_trend,
            'Spread_Vol_60d': roll_std,
            'Target_Fwd_10d_Delta': fwd_10d_delta
        })
        
        # Drop rows with NaNs (e.g., initial 250 days, last 10 days)
        pair_df = pair_df.dropna()
        ml_records.append(pair_df)
        
    print(f"\nCombining datasets...")
    final_dataset = pd.concat(ml_records, ignore_index=True)
    
    # Save Dataset
    v21_dir = os.path.join(DATA_DIR, "v21")
    if not os.path.exists(v21_dir): os.makedirs(v21_dir)
    
    out_path = os.path.join(v21_dir, "v21_ml_dataset.csv")
    final_dataset.to_csv(out_path, index=False)
    
    print(f"Feature Engineering Complete! Extracted {len(final_dataset)} total daily observations.")
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    build_ml_dataset()
