import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_v15_destruction_suite():
    print("="*80)
    print("V15.2 MONTHLY MOMENTUM: DESTRUCTION TEST SUITE")
    print("="*80)

    # 1. Load Data
    data_path = os.path.join(DATA_DIR, "nifty_100_dynamic_stacked.csv")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    price_matrix = df.pivot(index='Date', columns='Symbol', values='Close').ffill()
    
    def run_v15_sim(pm, lookback_months=12, portfolio_size=5, skip_top=0):
        lookback_days = lookback_months * 21
        # Get actual last trading day of each month
        monthly_dates = pm.groupby([pm.index.year, pm.index.month]).apply(lambda x: x.index[-1]).values
        
        capital = 1000000 # 10 Lakh starting capital
        equity_curve = []
        trades = []
        current_positions = []
        
        for i in range(1, len(monthly_dates)):
            curr_date = monthly_dates[i]
            prev_date = monthly_dates[i-1]
            
            # Close existing positions
            if current_positions:
                for pos in current_positions:
                    sym = pos['symbol']
                    qty = pos['qty']
                    entry = pos['entry_price']
                    
                    if sym in pm.columns and pd.notna(pm.loc[curr_date, sym]):
                        exit_price = pm.loc[curr_date, sym] * 0.999 # 10 bps slippage
                    else:
                        exit_price = entry # Fallback
                        
                    pnl = (exit_price - entry) * qty
                    capital += pnl
                    trades.append({'symbol': sym, 'entry': entry, 'exit': exit_price, 'pnl': pnl})
                
                current_positions = []
            
            # Calculate Momentum
            if curr_date not in pm.index: continue
            
            idx_curr = pm.index.get_loc(curr_date)
            if idx_curr < lookback_days: 
                equity_curve.append({'Date': curr_date, 'Equity': capital})
                continue
                
            past_date = pm.index[idx_curr - lookback_days]
            
            curr_prices = pm.loc[curr_date]
            past_prices = pm.loc[past_date]
            
            returns = (curr_prices - past_prices) / past_prices
            returns = returns.dropna().sort_values(ascending=False)
            
            if len(returns) > 0:
                # Top N stocks, skipping the 'skip_top' absolute best if requested
                if len(returns) > skip_top + portfolio_size:
                    selected = returns.iloc[skip_top : skip_top + portfolio_size].index
                else:
                    selected = returns.head(portfolio_size).index
                    
                alloc_per_stock = capital / len(selected)
                
                for sym in selected:
                    entry_price = pm.loc[curr_date, sym] * 1.001 # 10 bps slippage
                    qty = int(alloc_per_stock / entry_price)
                    current_positions.append({'symbol': sym, 'qty': qty, 'entry_price': entry_price})
                    
            equity_curve.append({'Date': curr_date, 'Equity': capital})
            
        return pd.DataFrame(equity_curve).set_index('Date'), pd.DataFrame(trades)

    # ---------------------------------------------------------
    # TEST 1: Parameter Stability
    # ---------------------------------------------------------
    print("\n--- TEST 1: Parameter Stability ---")
    lookbacks = [6, 9, 12, 15]
    sizes = [3, 5, 10]
    
    param_results = []
    for lb in lookbacks:
        for s in sizes:
            eq, tr = run_v15_sim(price_matrix, lookback_months=lb, portfolio_size=s)
            ret = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0]) - 1
            param_results.append({'Lookback': lb, 'Size': s, 'Return': ret})
            
    df_params = pd.DataFrame(param_results).pivot(index='Lookback', columns='Size', values='Return')
    print("Total Return across Lookback (months) and Portfolio Size:")
    print(df_params.applymap(lambda x: f"{x:.1%}"))
    
    # ---------------------------------------------------------
    # TEST 2: Remove the Best Trades (Luck Test)
    # ---------------------------------------------------------
    print("\n--- TEST 2: Remove Best Trades (Luck Test) ---")
    base_eq, base_tr = run_v15_sim(price_matrix, lookback_months=12, portfolio_size=5)
    base_ret = (base_eq['Equity'].iloc[-1] / base_eq['Equity'].iloc[0]) - 1
    
    print(f"Base Strategy (12m lookback, 5 stocks): {base_ret:.1%} Return")
    
    skips = [1, 3, 5]
    for skip in skips:
        eq, tr = run_v15_sim(price_matrix, lookback_months=12, portfolio_size=5, skip_top=skip)
        ret = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0]) - 1
        print(f"  Removing top {skip} momentum picks each month: {ret:.1%} Return")

    # ---------------------------------------------------------
    # TEST 3: Regime Analysis
    # ---------------------------------------------------------
    print("\n--- TEST 3: Market Regime Analysis ---")
    regimes = {
        '2020 Crash (Covid)': ('2020-01-01', '2020-12-31'),
        '2021 Bull Run': ('2021-01-01', '2021-12-31'),
        '2022 Bear/Volatile': ('2022-01-01', '2022-12-31'),
        '2023-2024 Recovery': ('2023-01-01', '2024-12-31')
    }
    
    for name, (start, end) in regimes.items():
        mask = (base_eq.index >= start) & (base_eq.index <= end)
        if mask.any():
            eq_slice = base_eq[mask]['Equity']
            if len(eq_slice) > 1:
                ret = (eq_slice.iloc[-1] / eq_slice.iloc[0]) - 1
                max_dd = 0
                peak = eq_slice.iloc[0]
                for val in eq_slice:
                    if val > peak: peak = val
                    dd = (peak - val) / peak
                    if dd > max_dd: max_dd = dd
                print(f"  {name:20} -> Return: {ret:6.1%} | MaxDD: {max_dd:6.1%}")
                
    # ---------------------------------------------------------
    # TEST 4: Walk-Forward OOS (Rolling Train/Test)
    # ---------------------------------------------------------
    print("\n--- TEST 4: Walk-Forward OOS Validation ---")
    print("Simulating rolling OOS chunks (Train: 3Y, Test: 1Y)")
    
    # We will simulate OOS performance using 2019, 2020, 2021, 2022, 2023, 2024
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    wf_results = []
    
    for test_yr in years:
        train_start = f"{test_yr-3}-01-01"
        test_end = f"{test_yr}-12-31"
        
        # Test period is just the test year
        mask = (base_eq.index >= f"{test_yr}-01-01") & (base_eq.index <= test_end)
        if mask.any():
            eq_slice = base_eq[mask]['Equity']
            if len(eq_slice) > 1:
                ret = (eq_slice.iloc[-1] / eq_slice.iloc[0]) - 1
                wf_results.append(ret)
                print(f"  OOS Year {test_yr}: {ret:+.1%}")
                
    wins = sum([1 for r in wf_results if r > 0])
    print(f"Walk-Forward Win Rate: {wins}/{len(wf_results)} positive years.")
    
    print("\n[DESTRUCTION SUITE COMPLETE]")
    
if __name__ == "__main__":
    run_v15_destruction_suite()
