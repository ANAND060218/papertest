import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_ml_engine():
    print("="*80)
    print("V21-A XGBOOST SPREAD PREDICTOR (SUPERVISED ML)")
    print("="*80)
    
    dataset_path = os.path.join(DATA_DIR, "v21", "v21_ml_dataset.csv")
    if not os.path.exists(dataset_path):
        print("Dataset not found. Run feature engineering first.")
        return
        
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Features & Target
    features = ['Z_Score', 'Hedge_Ratio', 'Sector_Mom_20d', 'Spread_Vol_60d']
    target = 'Target_Fwd_10d_Delta'
    
    df = df.dropna(subset=features + [target])
    
    # Train / Test Split
    train_df = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2020-12-31')]
    test_df = df[(df['Date'] >= '2021-01-01') & (df['Date'] <= '2025-12-31')]
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Training on {len(train_df)} samples (2016-2020)...")
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"OOS MSE: {mse:.4f}")
    
    # Feature Importance
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for f, imp in zip(features, importances):
        print(f"  {f}: {imp:.4f}")
        
    # ML-Driven Backtest
    print("\nRunning ML-Driven OOS Backtest (2021-2025)...")
    
    test_df['ML_Prediction'] = preds
    
    total_gross_pnl = 0
    total_costs = 0
    winning_pairs = 0
    
    # Group by pair to simulate
    for pair_id, pair_data in test_df.groupby('Pair'):
        pair_data = pair_data.sort_values('Date').reset_index(drop=True)
        
        position = 0
        gross_pnl = 0
        trades = 0
        entry_price_a = 0
        entry_price_b = 0
        
        hr = pair_data['Hedge_Ratio'].mean() # approx for cost sizing
        
        for i in range(1, len(pair_data)):
            row = pair_data.iloc[i]
            z = row['Z_Score']
            pred_delta = row['ML_Prediction']
            
            curr_pa = row['Price_A']
            curr_pb = row['Price_B']
            curr_hr = row['Hedge_Ratio']
            
            # ML Rule: We only short if Z > 1.5 AND ML predicts spread will drop by > 1.0 (covering costs)
            if position == 0 and z > 1.5 and pred_delta < -1.0:
                position = -1
                entry_price_a = curr_pa * 0.9995
                entry_price_b = curr_pb * 1.0005
                trades += 2
            # ML Rule: We only long if Z < -1.5 AND ML predicts spread will rise by > 1.0
            elif position == 0 and z < -1.5 and pred_delta > 1.0:
                position = 1
                entry_price_a = curr_pa * 1.0005
                entry_price_b = curr_pb * 0.9995
                trades += 2
            # Exit on mean reversion (Z crosses 0)
            elif (position == -1 and z <= 0) or (position == 1 and z >= 0):
                if position == -1:
                    exit_price_a = curr_pa * 1.0005 
                    exit_price_b = curr_pb * 0.9995 
                    pnl_a = entry_price_a - exit_price_a
                    pnl_b = (exit_price_b - entry_price_b) * curr_hr
                else:
                    exit_price_a = curr_pa * 0.9995 
                    exit_price_b = curr_pb * 1.0005 
                    pnl_a = exit_price_a - entry_price_a
                    pnl_b = (entry_price_b - exit_price_b) * curr_hr
                    
                gross_pnl += (pnl_a + pnl_b) * (100000 / curr_pa) # 1L per leg
                position = 0
                trades += 2
                
        costs = trades * 100000 * 0.0003
        net_pnl = gross_pnl - costs
        
        total_gross_pnl += gross_pnl
        total_costs += costs
        
        if net_pnl > 0:
            winning_pairs += 1
            
    print("\n" + "="*80)
    print(f"ML-ENHANCED OOS RESULTS (2021-2025):")
    print(f"Winning Pairs: {winning_pairs} / {len(test_df['Pair'].unique())}")
    print(f"Gross P&L: Rs {total_gross_pnl:,.0f}")
    print(f"Costs & Slippage: Rs {total_costs:,.0f}")
    print(f"Net Portfolio P&L: Rs {(total_gross_pnl - total_costs):,.0f}")
    
if __name__ == "__main__":
    run_ml_engine()
