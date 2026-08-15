import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import statsmodels exactly as backtest does
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from core.paper_broker import VirtualBroker
from core.database import get_db_connection

# Sector groups from backtest
SECTOR_GROUPS = {
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

def fetch_data_for_pair(symbol_a, symbol_b, lookback_days=300):
    """
    Fetches the historical data needed for training + rolling Z-score.
    In real life, this fetches from yfinance or an API. 
    Here we simulate using the local stacked CSV, up to a specified "current" date.
    """
    df_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'nifty_10year_stacked.csv'))
    
    # We want 300 calendar days of history to ensure we have ~200 trading days
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    df_a = df_all[df_all['Symbol'] == symbol_a].copy()
    df_b = df_all[df_all['Symbol'] == symbol_b].copy()
    
    merged = pd.merge(df_a[['Date', 'Close']], df_b[['Date', 'Close']], on='Date', suffixes=('_a', '_b'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged.tail(lookback_days)

def test_cointegration(series_a, series_b):
    """Engle-Granger two-step, perfectly mirroring V19 backtest."""
    X = add_constant(series_b.values)
    model = OLS(series_a.values, X).fit()
    hedge_ratio = model.params[1]
    intercept = model.params[0]
    
    spread = series_a.values - hedge_ratio * series_b.values - intercept
    
    # Needs at least 20 samples to not crash adf
    if len(spread) < 30:
        return False, 1.0, 0, 0, 0
        
    try:
        adf_result = adfuller(spread, maxlag=20, regression='c')
        p_value = adf_result[1]
    except:
        return False, 1.0, 0, 0, 0
    
    spread_lag = pd.Series(spread[:-1])
    spread_diff = pd.Series(np.diff(spread))
    X_hl = add_constant(spread_lag.values)
    model_hl = OLS(spread_diff.values, X_hl).fit()
    lambda_param = model_hl.params[1]
    half_life = -np.log(2) / lambda_param if lambda_param < 0 else float('inf')
    
    is_cointegrated = p_value < 0.05
    return is_cointegrated, p_value, hedge_ratio, intercept, half_life

class V19PaperEngine:
    """
    Executes the frozen V19 Statistical Arbitrage model.
    """
    def __init__(self, broker: VirtualBroker, strategy_id="V19_STATARB", capital_per_leg=100000.0):
        self.broker = broker
        self.strategy_id = strategy_id
        self.z_entry = 2.0
        self.z_exit = 0.5
        self.z_stop = 4.0
        self.lookback = 60
        self.capital_per_leg = capital_per_leg
        
    def run_daily_cycle(self, current_date_str):
        print(f"\n[V19 PAPER ENGINE] Running Daily Cycle for {current_date_str}")
        
        # 1. Update Pair Cointegration Health
        # We re-evaluate health over the last ~200 trading days.
        # If a pair breaks, we disable it.
        self._monitor_pair_health(current_date_str)
        
        # 2. Get active pairs
        conn = get_db_connection()
        active_pairs = pd.read_sql_query("SELECT * FROM v19_pairs WHERE status = 'ACTIVE'", conn)
        
        # 3. Process each active pair for trading signals
        current_prices = {}
        
        for _, row in active_pairs.iterrows():
            sym_a = row['stock_a']
            sym_b = row['stock_b']
            
            merged = fetch_data_for_pair(sym_a, sym_b, lookback_days=250)
            if len(merged) < self.lookback + 5:
                continue
                
            hr = row['hedge_ratio']
            intercept = 0 # Backtest assumes rolling mean captures intercept dynamically in Z-score calculation.
            
            merged['spread'] = merged['Close_a'] - hr * merged['Close_b']
            merged['spread_mean'] = merged['spread'].rolling(self.lookback).mean()
            merged['spread_std'] = merged['spread'].rolling(self.lookback).std()
            merged['z_score'] = (merged['spread'] - merged['spread_mean']) / merged['spread_std']
            
            current_row = merged.iloc[-1]
            z = current_row['z_score']
            price_a = current_row['Close_a']
            price_b = current_row['Close_b']
            
            current_prices[sym_a] = price_a
            current_prices[sym_b] = price_b
            
            # Update DB with latest Z-Score for dashboard
            c = conn.cursor()
            c.execute("UPDATE v19_pairs SET last_z_score = ?, updated_at = ? WHERE pair_id = ?",
                      (z, current_date_str, row['pair_id']))
            conn.commit()
            
            # --- Check Current Portfolio Position for this pair ---
            pos_a = pd.read_sql_query("SELECT quantity FROM positions WHERE strategy_id=? AND symbol=?", 
                                      conn, params=(self.strategy_id, sym_a))
            
            current_qty_a = pos_a['quantity'].iloc[0] if not pos_a.empty else 0
            
            # Trading Logic
            if current_qty_a == 0:
                # We are flat. Look for entry.
                if z > self.z_entry:
                    print(f"  [SIGNAL] {sym_a}/{sym_b} Z={z:.2f}. Spread abnormally HIGH. Short A, Buy B.")
                    qty_a = max(1, int(self.capital_per_leg / price_a))
                    qty_b = max(1, int(self.capital_per_leg / price_b))
                    
                    self.broker.submit_order(self.strategy_id, sym_a, 'SELL', qty_a, price_a, notes=f"Entry Z={z:.2f}")
                    self.broker.submit_order(self.strategy_id, sym_b, 'BUY', qty_b, price_b, notes=f"Entry Z={z:.2f}")
                    
                elif z < -self.z_entry:
                    print(f"  [SIGNAL] {sym_a}/{sym_b} Z={z:.2f}. Spread abnormally LOW. Buy A, Short B.")
                    qty_a = max(1, int(self.capital_per_leg / price_a))
                    qty_b = max(1, int(self.capital_per_leg / price_b))
                    
                    self.broker.submit_order(self.strategy_id, sym_a, 'BUY', qty_a, price_a, notes=f"Entry Z={z:.2f}")
                    self.broker.submit_order(self.strategy_id, sym_b, 'SELL', qty_b, price_b, notes=f"Entry Z={z:.2f}")
            else:
                # We are in a position. Look for exit.
                pos_b = pd.read_sql_query("SELECT quantity FROM positions WHERE strategy_id=? AND symbol=?", 
                                          conn, params=(self.strategy_id, sym_b))
                current_qty_b = pos_b['quantity'].iloc[0] if not pos_b.empty else 0
                
                # Are we Long Spread (Long A, Short B) or Short Spread (Short A, Long B)?
                is_long_spread = current_qty_a > 0
                
                should_exit = False
                reason = ""
                
                if is_long_spread:
                    # Entered when z < -z_entry. Exit when z crosses back over -z_exit, or stops at -z_stop
                    if z > -self.z_exit:
                        should_exit = True
                        reason = "Mean Reversion Reached"
                    elif z < -self.z_stop:
                        should_exit = True
                        reason = "Stop Loss"
                else:
                    # Entered when z > z_entry. Exit when z crosses back under z_exit, or stops at z_stop
                    if z < self.z_exit:
                        should_exit = True
                        reason = "Mean Reversion Reached"
                    elif z > self.z_stop:
                        should_exit = True
                        reason = "Stop Loss"
                        
                if should_exit:
                    print(f"  [SIGNAL] {sym_a}/{sym_b} Z={z:.2f}. EXITING. Reason: {reason}")
                    
                    if is_long_spread:
                        self.broker.submit_order(self.strategy_id, sym_a, 'SELL', abs(current_qty_a), price_a, notes=reason)
                        self.broker.submit_order(self.strategy_id, sym_b, 'BUY', abs(current_qty_b), price_b, notes=reason)
                    else:
                        self.broker.submit_order(self.strategy_id, sym_a, 'BUY', abs(current_qty_a), price_a, notes=reason)
                        self.broker.submit_order(self.strategy_id, sym_b, 'SELL', abs(current_qty_b), price_b, notes=reason)
        
        # 4. Save EOD snapshot
        self.broker.save_eod_equity(self.strategy_id, current_prices, current_date_str)
        conn.close()
        
    def _monitor_pair_health(self, current_date_str):
        """
        Re-tests all combinations within sectors to find currently cointegrated pairs.
        Registers new ones, disables broken ones.
        """
        conn = get_db_connection()
        c = conn.cursor()
        
        new_active = []
        
        for sector, tickers in SECTOR_GROUPS.items():
            from itertools import combinations
            for t1, t2 in combinations(tickers, 2):
                merged = fetch_data_for_pair(t1, t2, lookback_days=250)
                if len(merged) < 150:
                    continue
                    
                is_coint, p_val, hr, intercept, hl = test_cointegration(merged['Close_a'], merged['Close_b'])
                
                pair_id = f"{t1}/{t2}"
                
                if is_coint and 1 < hl < 60:
                    new_active.append(pair_id)
                    # Insert or Update as active
                    c.execute('''
                        INSERT INTO v19_pairs (pair_id, stock_a, stock_b, hedge_ratio, half_life, status, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
                        ON CONFLICT(pair_id) DO UPDATE SET 
                            hedge_ratio=excluded.hedge_ratio,
                            half_life=excluded.half_life,
                            status='ACTIVE',
                            updated_at=excluded.updated_at
                    ''', (pair_id, t1, t2, hr, hl, current_date_str))
                else:
                    # Update to DISABLED if it exists
                    c.execute('''
                        UPDATE v19_pairs SET status='DISABLED', updated_at=? WHERE pair_id=?
                    ''', (current_date_str, pair_id))
                    
        conn.commit()
        conn.close()
        print(f"  [HEALTH] Pair Cointegration Matrix updated. Active pairs: {len(new_active)}")

if __name__ == "__main__":
    from core.database import init_db, register_strategy
    
    init_db()
    register_strategy("V19_STATARB", 1000000.0) # 10L start
    
    broker = VirtualBroker(default_slippage_pct=0.0005)
    engine = V19PaperEngine(broker)
    
    # Simulate today's run
    engine.run_daily_cycle("2026-08-15")
