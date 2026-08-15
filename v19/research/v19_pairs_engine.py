"""
V19 Statistical Arbitrage (Pairs Trading) Engine
=================================================
Fundamentally different from V15-V18:
  - NOT predicting market direction
  - Profiting from temporary mispricing between correlated stocks
  - Market-neutral: can profit whether NIFTY goes up or down

Edge hypothesis: historically cointegrated stock pairs temporarily diverge,
then revert to their equilibrium spread. We buy the cheap one, short the
expensive one, and profit from convergence.

Uses the same 10-year NSE dataset as V15.2.
"""
import sys, os
import pandas as pd
import numpy as np
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ─── 2026 Equity Delivery/Intraday Cost Model ───────────────────────────
def calculate_equity_trade_cost(price, qty, is_buy, is_intraday=False):
    """
    2026 Indian equity transaction costs.
    For pairs trading we use delivery (CNC) since holding > 1 day.
    """
    turnover = price * qty
    brokerage = 20.0  # flat per order
    
    # Exchange charges
    exchange = turnover * 0.00035
    
    # GST on brokerage + exchange
    gst = (brokerage + exchange) * 0.18
    
    if is_intraday:
        # Intraday: STT 0.025% on sell side
        stt = turnover * 0.00025 if not is_buy else 0.0
    else:
        # Delivery: STT 0.1% on both sides
        stt = turnover * 0.001
    
    # Stamp duty: 0.015% on buy side
    stamp = turnover * 0.00015 if is_buy else 0.0
    
    # SEBI fee
    sebi = turnover * 0.000001
    
    return brokerage + exchange + gst + stt + stamp + sebi

def calculate_pair_trade_costs(price_a, price_b, qty_a, qty_b, is_intraday=False):
    """
    Full round-trip cost for a pairs trade (4 orders: buy A, sell B, then reverse).
    """
    # Entry
    entry_cost_a = calculate_equity_trade_cost(price_a, qty_a, is_buy=True, is_intraday=is_intraday)
    entry_cost_b = calculate_equity_trade_cost(price_b, qty_b, is_buy=False, is_intraday=is_intraday)  # short
    
    # Exit (reverse)
    exit_cost_a = calculate_equity_trade_cost(price_a, qty_a, is_buy=False, is_intraday=is_intraday)
    exit_cost_b = calculate_equity_trade_cost(price_b, qty_b, is_buy=True, is_intraday=is_intraday)
    
    return entry_cost_a + entry_cost_b + exit_cost_a + exit_cost_b


# ─── Cointegration Testing ──────────────────────────────────────────────
def test_cointegration(series_a, series_b):
    """
    Engle-Granger two-step cointegration test.
    Returns (is_cointegrated, p_value, hedge_ratio, half_life).
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    # Step 1: OLS regression to find hedge ratio
    X = add_constant(series_b.values)
    model = OLS(series_a.values, X).fit()
    hedge_ratio = model.params[1]
    intercept = model.params[0]
    
    # Step 2: Test residuals (spread) for stationarity
    spread = series_a.values - hedge_ratio * series_b.values - intercept
    adf_result = adfuller(spread, maxlag=20, regression='c')
    p_value = adf_result[1]
    
    # Step 3: Estimate mean-reversion half-life
    spread_lag = pd.Series(spread[:-1])
    spread_diff = pd.Series(np.diff(spread))
    X_hl = add_constant(spread_lag.values)
    model_hl = OLS(spread_diff.values, X_hl).fit()
    lambda_param = model_hl.params[1]
    half_life = -np.log(2) / lambda_param if lambda_param < 0 else float('inf')
    
    is_cointegrated = p_value < 0.05
    
    return is_cointegrated, p_value, hedge_ratio, intercept, half_life


# ─── Pairs Trading Backtest ─────────────────────────────────────────────
def backtest_pair(df_a, df_b, name_a, name_b, hedge_ratio, intercept,
                  lookback=60, z_entry=2.0, z_exit=0.5, z_stop=4.0,
                  capital_per_leg=100000):
    """
    Walk-forward pairs trading backtest.
    
    - Recalculates rolling spread mean/std using only past data (no look-ahead)
    - Enters when z-score > z_entry or < -z_entry
    - Exits when z-score crosses z_exit (mean reversion)
    - Stop-loss if z-score > z_stop (divergence acceleration)
    """
    # Align dates
    merged = pd.merge(df_a[['Date', 'Close']], df_b[['Date', 'Close']], 
                       on='Date', suffixes=('_a', '_b'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    # Compute spread
    merged['spread'] = merged['Close_a'] - hedge_ratio * merged['Close_b'] - intercept
    
    # Rolling z-score (lookback window, using ONLY past data)
    merged['spread_mean'] = merged['spread'].rolling(lookback, min_periods=lookback).mean()
    merged['spread_std'] = merged['spread'].rolling(lookback, min_periods=lookback).std()
    merged['z_score'] = (merged['spread'] - merged['spread_mean']) / merged['spread_std']
    
    trades = []
    position = 0  # 0 = flat, 1 = long spread, -1 = short spread
    entry_price_a = 0
    entry_price_b = 0
    entry_date = None
    
    for i in range(lookback, len(merged)):
        row = merged.iloc[i]
        z = row['z_score']
        
        if np.isnan(z):
            continue
        
        price_a = row['Close_a']
        price_b = row['Close_b']
        
        # Position sizing: equal capital per leg
        qty_a = max(1, int(capital_per_leg / price_a))
        qty_b = max(1, int(capital_per_leg / price_b))
        
        if position == 0:
            # ENTRY: spread is abnormally high → short spread (sell A, buy B)
            if z > z_entry:
                position = -1
                entry_price_a = price_a
                entry_price_b = price_b
                entry_date = row['Date']
                entry_qty_a = qty_a
                entry_qty_b = qty_b
            
            # ENTRY: spread is abnormally low → long spread (buy A, sell B)
            elif z < -z_entry:
                position = 1
                entry_price_a = price_a
                entry_price_b = price_b
                entry_date = row['Date']
                entry_qty_a = qty_a
                entry_qty_b = qty_b
        
        elif position == 1:  # long spread (bought A, shorted B)
            # EXIT: mean reversion
            if z >= -z_exit or z > z_stop:
                pnl_a = (price_a - entry_price_a) * entry_qty_a
                pnl_b = (entry_price_b - price_b) * entry_qty_b  # short B
                gross_pnl = pnl_a + pnl_b
                
                costs = calculate_pair_trade_costs(
                    entry_price_a, entry_price_b, entry_qty_a, entry_qty_b)
                # Add exit costs at current prices
                costs += calculate_pair_trade_costs(
                    price_a, price_b, entry_qty_a, entry_qty_b)
                # Simplify: just use entry+exit
                costs = calculate_pair_trade_costs(
                    (entry_price_a + price_a) / 2, (entry_price_b + price_b) / 2,
                    entry_qty_a, entry_qty_b)
                
                net_pnl = gross_pnl - costs
                holding_days = (pd.to_datetime(row['Date']) - pd.to_datetime(entry_date)).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['Date'],
                    'direction': 'LONG_SPREAD',
                    'z_entry': z,
                    'gross_pnl': gross_pnl,
                    'costs': costs,
                    'net_pnl': net_pnl,
                    'holding_days': holding_days,
                    'exit_reason': 'REVERSION' if z >= -z_exit else 'STOP'
                })
                position = 0
        
        elif position == -1:  # short spread (sold A, bought B)
            if z <= z_exit or z < -z_stop:
                pnl_a = (entry_price_a - price_a) * entry_qty_a  # short A
                pnl_b = (price_b - entry_price_b) * entry_qty_b  # long B
                gross_pnl = pnl_a + pnl_b
                
                costs = calculate_pair_trade_costs(
                    (entry_price_a + price_a) / 2, (entry_price_b + price_b) / 2,
                    entry_qty_a, entry_qty_b)
                
                net_pnl = gross_pnl - costs
                holding_days = (pd.to_datetime(row['Date']) - pd.to_datetime(entry_date)).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['Date'],
                    'direction': 'SHORT_SPREAD',
                    'z_entry': z,
                    'gross_pnl': gross_pnl,
                    'costs': costs,
                    'net_pnl': net_pnl,
                    'holding_days': holding_days,
                    'exit_reason': 'REVERSION' if z <= z_exit else 'STOP'
                })
                position = 0
    
    return pd.DataFrame(trades)


# ─── Main Runner ────────────────────────────────────────────────────────
def run_v19():
    print("=" * 100)
    print("V19 STATISTICAL ARBITRAGE (PAIRS TRADING) ENGINE")
    print("=" * 100)
    
    # Load all stock data
    df_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'nifty_10year_stacked.csv'))
    
    # Filter to stocks only (exclude indices)
    stocks = [s for s in df_all['Symbol'].unique() if not s.startswith('^')]
    print(f"Stocks available: {len(stocks)}")
    
    # ─── PHASE 1: Cointegration Screening ────────────────────────────
    # Use 2016-2020 as training period for cointegration
    train_end = '2021-01-01'
    test_start = '2021-01-01'
    
    print(f"\nPhase 1: Cointegration Screening (training period: 2016 - {train_end})")
    print("-" * 80)
    
    # Pre-filter: only test pairs within same sector
    sector_groups = {
        'Banks': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
        'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS'],
        'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS'],
        'Energy': ['RELIANCE.NS', 'BPCL.NS', 'NTPC.NS', 'POWERGRID.NS'],
        'Auto': ['M&M.NS', 'MARUTI.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'],
        'Finance': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS', 'HDFCLIFE.NS'],
        'Infra': ['LT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'ADANIENT.NS', 'ADANIPORTS.NS'],
        'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'ASIANPAINT.NS', 'TITAN.NS'],
    }
    
    cointegrated_pairs = []
    
    for sector, tickers in sector_groups.items():
        available = [t for t in tickers if t in stocks]
        for t1, t2 in combinations(available, 2):
            df_1 = df_all[(df_all['Symbol'] == t1) & (df_all['Date'] < train_end)][['Date', 'Close']].dropna()
            df_2 = df_all[(df_all['Symbol'] == t2) & (df_all['Date'] < train_end)][['Date', 'Close']].dropna()
            
            # Merge on date
            merged = pd.merge(df_1, df_2, on='Date', suffixes=('_1', '_2'))
            if len(merged) < 200:
                continue
            
            try:
                is_coint, p_val, hr, intercept, hl = test_cointegration(
                    merged['Close_1'], merged['Close_2'])
            except Exception:
                continue
            
            if is_coint and 1 < hl < 60:
                cointegrated_pairs.append({
                    'sector': sector,
                    'stock_a': t1,
                    'stock_b': t2,
                    'p_value': p_val,
                    'hedge_ratio': hr,
                    'intercept': intercept,
                    'half_life': hl,
                })
                print(f"  [COINT] {sector:>10}: {t1:>16} / {t2:>16} | p={p_val:.4f} | HR={hr:.3f} | HL={hl:.1f}d")
    
    print(f"\nCointegrated pairs found: {len(cointegrated_pairs)}")
    
    if not cointegrated_pairs:
        print("NO COINTEGRATED PAIRS FOUND. V19 CANNOT PROCEED.")
        return
    
    # ─── PHASE 2: Out-of-Sample Backtest ─────────────────────────────
    print(f"\nPhase 2: Out-of-Sample Backtest ({test_start} - present)")
    print("-" * 80)
    
    all_results = []
    
    for pair in cointegrated_pairs:
        df_a = df_all[(df_all['Symbol'] == pair['stock_a']) & (df_all['Date'] >= test_start)][['Date', 'Close']].dropna()
        df_b = df_all[(df_all['Symbol'] == pair['stock_b']) & (df_all['Date'] >= test_start)][['Date', 'Close']].dropna()
        
        if len(df_a) < 100 or len(df_b) < 100:
            continue
        
        trades_df = backtest_pair(
            df_a, df_b, pair['stock_a'], pair['stock_b'],
            pair['hedge_ratio'], pair['intercept'],
            lookback=60, z_entry=2.0, z_exit=0.5, z_stop=4.0,
            capital_per_leg=100000
        )
        
        if trades_df.empty or len(trades_df) < 5:
            continue
        
        n = len(trades_df)
        wins = (trades_df['net_pnl'] > 0).sum()
        gross = trades_df['gross_pnl'].sum()
        costs = trades_df['costs'].sum()
        net = trades_df['net_pnl'].sum()
        
        gp = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gl = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
        
        cum = trades_df['net_pnl'].cumsum()
        maxdd = (cum - cum.cummax()).min()
        
        avg_hold = trades_df['holding_days'].mean()
        reversions = (trades_df['exit_reason'] == 'REVERSION').sum()
        
        result = {
            'pair': f"{pair['stock_a'].replace('.NS','')} / {pair['stock_b'].replace('.NS','')}",
            'sector': pair['sector'],
            'trades': n,
            'win_rate': wins / n * 100,
            'gross_pnl': gross,
            'costs': costs,
            'net_pnl': net,
            'net_exp': net / n,
            'pf': pf,
            'max_dd': maxdd,
            'avg_hold': avg_hold,
            'reversions': reversions,
            'stops': n - reversions,
            'half_life': pair['half_life'],
        }
        all_results.append(result)
        
        status = "+" if net > 0 else "-"
        print(f"  [{status}] {result['pair']:>30} | {n:>3} trades | WR {result['win_rate']:.0f}% | "
              f"Net Rs {net:>8.0f} | PF {pf:.2f} | DD Rs {maxdd:.0f} | Avg Hold {avg_hold:.0f}d")
    
    # ─── PHASE 3: Summary ────────────────────────────────────────────
    if not all_results:
        print("\nNO PAIRS PRODUCED ENOUGH TRADES FOR EVALUATION.")
        return
    
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "=" * 100)
    print("V19 STATISTICAL ARBITRAGE — OOS SUMMARY (2021-2025)")
    print("=" * 100)
    
    profitable = df_results[df_results['net_pnl'] > 0]
    unprofitable = df_results[df_results['net_pnl'] <= 0]
    
    print(f"Total pairs tested:     {len(df_results)}")
    print(f"Profitable pairs:       {len(profitable)}")
    print(f"Unprofitable pairs:     {len(unprofitable)}")
    print(f"Total trades:           {df_results['trades'].sum()}")
    print(f"Total Net P&L:          Rs {df_results['net_pnl'].sum():.0f}")
    print(f"Total Costs:            Rs {df_results['costs'].sum():.0f}")
    
    if len(profitable) > 0:
        print(f"\n--- Top Profitable Pairs ---")
        top = profitable.sort_values('net_pnl', ascending=False)
        for _, row in top.iterrows():
            print(f"  {row['pair']:>30} | {row['trades']:>3} trades | WR {row['win_rate']:.0f}% | "
                  f"Net Rs {row['net_pnl']:>8.0f} | PF {row['pf']:.2f} | HL {row['half_life']:.0f}d")
    
    # Aggregate portfolio if we traded ALL profitable pairs
    if len(profitable) > 0:
        total_net = profitable['net_pnl'].sum()
        total_trades = profitable['trades'].sum()
        total_costs = profitable['costs'].sum()
        avg_pf = profitable['pf'].mean()
        print(f"\n  PORTFOLIO (all profitable pairs):")
        print(f"    Total Net P&L:  Rs {total_net:.0f}")
        print(f"    Total Trades:   {total_trades}")
        print(f"    Avg PF:         {avg_pf:.2f}")
        print(f"    Total Costs:    Rs {total_costs:.0f}")
    
    print("=" * 100)


if __name__ == "__main__":
    run_v19()
