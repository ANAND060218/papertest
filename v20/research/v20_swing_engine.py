"""
V20 Swing Trading Engine (2-10 Day Holding Period)
====================================================
Bridges the gap between V15.2 (monthly) and V16/V17 (intraday).

Strategy:
  Daily data -> Market regime -> Sector strength -> Relative strength
  -> Top stocks -> Hold 2-10 days

Edge hypothesis: Multi-day momentum capture can outrun transaction costs
while maintaining higher frequency than monthly rebalancing.

Uses the same 10-year NSE dataset. Walk-forward OOS validation.
"""
import sys, os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ─── 2026 Equity Cost Model ─────────────────────────────────────────────
def calculate_delivery_costs(buy_price, sell_price, qty):
    """
    Full round-trip delivery (CNC) costs for Indian equity.
    """
    buy_turnover = buy_price * qty
    sell_turnover = sell_price * qty
    
    # Brokerage: Rs 20 flat per order (2 orders: buy + sell)
    brokerage = 40.0
    
    # Exchange charges: 0.035% on both sides
    exchange = (buy_turnover + sell_turnover) * 0.00035
    
    # GST: 18% on brokerage + exchange
    gst = (brokerage + exchange) * 0.18
    
    # STT: 0.1% on both buy and sell for delivery
    stt = (buy_turnover + sell_turnover) * 0.001
    
    # Stamp duty: 0.015% on buy side
    stamp = buy_turnover * 0.00015
    
    # SEBI fee
    sebi = (buy_turnover + sell_turnover) * 0.000001
    
    return brokerage + exchange + gst + stt + stamp + sebi


# ─── Market Regime ──────────────────────────────────────────────────────
def compute_market_regime(nifty_close, sma_period=50):
    """
    Simple regime filter: NIFTY above 50-day SMA = Risk-On, below = Risk-Off.
    """
    sma = nifty_close.rolling(sma_period).mean()
    return nifty_close > sma


# ─── Relative Strength ──────────────────────────────────────────────────
def compute_relative_strength(df_all, date, lookback=20):
    """
    Ranks all stocks by their N-day return. Returns top stocks.
    """
    rankings = []
    stocks = [s for s in df_all['Symbol'].unique() if not s.startswith('^')]
    
    for sym in stocks:
        df_sym = df_all[(df_all['Symbol'] == sym) & (df_all['Date'] <= date)].tail(lookback + 1)
        if len(df_sym) < lookback + 1:
            continue
        
        ret = (df_sym['Close'].iloc[-1] / df_sym['Close'].iloc[0]) - 1
        vol = df_sym['Close'].pct_change().std() * np.sqrt(252)
        
        # Risk-adjusted momentum (return / volatility)
        risk_adj = ret / vol if vol > 0 else 0
        
        rankings.append({
            'symbol': sym,
            'return': ret,
            'volatility': vol,
            'risk_adj_momentum': risk_adj,
            'last_price': df_sym['Close'].iloc[-1],
            'volume': df_sym['Volume'].iloc[-5:].mean(),
        })
    
    df_rank = pd.DataFrame(rankings)
    
    # Filter: must have positive momentum and reasonable volume
    df_rank = df_rank[df_rank['return'] > 0]
    df_rank = df_rank[df_rank['volume'] > 50000]
    
    # Sort by risk-adjusted momentum
    df_rank = df_rank.sort_values('risk_adj_momentum', ascending=False)
    
    return df_rank


# ─── Swing Trading Backtest ─────────────────────────────────────────────
def run_swing_backtest(df_all, start_date, end_date, 
                       top_n=3, hold_days=5, lookback=20,
                       capital=1000000, rebalance_every=5):
    """
    Walk-forward swing trading backtest.
    
    Every `rebalance_every` trading days:
    1. Check market regime (NIFTY > 50 SMA)
    2. If Risk-On: rank stocks by 20-day risk-adjusted momentum
    3. Buy top N stocks with equal weight
    4. Hold for `hold_days` days, then rebalance
    
    If Risk-Off: stay in cash.
    """
    # Get NIFTY data for regime
    df_nifty = df_all[df_all['Symbol'] == '^NSEI'][['Date', 'Close']].copy()
    df_nifty = df_nifty.sort_values('Date').reset_index(drop=True)
    df_nifty['regime'] = compute_market_regime(df_nifty['Close'], 50)
    
    # Get all trading dates in range
    all_dates = sorted(df_all[(df_all['Date'] >= start_date) & 
                              (df_all['Date'] <= end_date)]['Date'].unique())
    
    trades = []
    portfolio = {}  # {symbol: {qty, entry_price, entry_date}}
    
    for i, date in enumerate(all_dates):
        # Check if it's a rebalance day
        if i % rebalance_every != 0:
            continue
        
        # Get regime
        nifty_row = df_nifty[df_nifty['Date'] <= date].tail(1)
        if nifty_row.empty:
            continue
        is_risk_on = nifty_row['regime'].iloc[0]
        
        # Close existing positions
        for sym, pos in list(portfolio.items()):
            df_sym = df_all[(df_all['Symbol'] == sym) & (df_all['Date'] == date)]
            if df_sym.empty:
                # Use last available price
                df_sym = df_all[(df_all['Symbol'] == sym) & (df_all['Date'] <= date)].tail(1)
            if df_sym.empty:
                continue
            
            exit_price = df_sym['Close'].iloc[0]
            gross_pnl = (exit_price - pos['entry_price']) * pos['qty']
            costs = calculate_delivery_costs(pos['entry_price'], exit_price, pos['qty'])
            net_pnl = gross_pnl - costs
            
            holding = (pd.to_datetime(date) - pd.to_datetime(pos['entry_date'])).days
            
            trades.append({
                'symbol': sym.replace('.NS', ''),
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'qty': pos['qty'],
                'gross_pnl': gross_pnl,
                'costs': costs,
                'net_pnl': net_pnl,
                'holding_days': holding,
                'regime': 'ON' if is_risk_on else 'OFF',
            })
        
        portfolio = {}
        
        # Open new positions if Risk-On
        if is_risk_on:
            rankings = compute_relative_strength(df_all, date, lookback)
            
            if rankings.empty:
                continue
            
            top_stocks = rankings.head(top_n)
            capital_per_stock = capital / top_n
            
            for _, stock in top_stocks.iterrows():
                sym = stock['symbol']
                price = stock['last_price']
                qty = max(1, int(capital_per_stock / price))
                
                portfolio[sym] = {
                    'qty': qty,
                    'entry_price': price,
                    'entry_date': date,
                }
    
    # Close remaining positions at end
    for sym, pos in portfolio.items():
        df_sym = df_all[(df_all['Symbol'] == sym)].tail(1)
        if df_sym.empty:
            continue
        exit_price = df_sym['Close'].iloc[0]
        gross_pnl = (exit_price - pos['entry_price']) * pos['qty']
        costs = calculate_delivery_costs(pos['entry_price'], exit_price, pos['qty'])
        trades.append({
            'symbol': sym.replace('.NS', ''),
            'entry_date': pos['entry_date'],
            'exit_date': df_sym['Date'].iloc[0],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'qty': pos['qty'],
            'gross_pnl': gross_pnl,
            'costs': costs,
            'net_pnl': gross_pnl - costs,
            'holding_days': 0,
            'regime': 'END',
        })
    
    return pd.DataFrame(trades)


# ─── Main Runner ────────────────────────────────────────────────────────
def run_v20():
    print("=" * 100)
    print("V20 SWING TRADING ENGINE (2-10 DAY HOLDING)")
    print("=" * 100)
    
    df_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'nifty_10year_stacked.csv'))
    
    # Walk-forward structure:
    # Train: 2016-2020 (parameter selection)
    # OOS-1: 2021-2022
    # OOS-2: 2023-2025
    
    configs = [
        {'top_n': 3, 'hold_days': 5, 'lookback': 10, 'rebalance': 5, 'label': 'Top3/5d/10L'},
        {'top_n': 3, 'hold_days': 5, 'lookback': 20, 'rebalance': 5, 'label': 'Top3/5d/20L'},
        {'top_n': 5, 'hold_days': 5, 'lookback': 20, 'rebalance': 5, 'label': 'Top5/5d/20L'},
        {'top_n': 3, 'hold_days': 10, 'lookback': 20, 'rebalance': 10, 'label': 'Top3/10d/20L'},
        {'top_n': 5, 'hold_days': 10, 'lookback': 20, 'rebalance': 10, 'label': 'Top5/10d/20L'},
    ]
    
    periods = [
        {'name': 'IN-SAMPLE 2016-2020', 'start': '2016-08-01', 'end': '2020-12-31'},
        {'name': 'OOS-1 2021-2022', 'start': '2021-01-01', 'end': '2022-12-31'},
        {'name': 'OOS-2 2023-2025', 'start': '2023-01-01', 'end': '2025-12-31'},
    ]
    
    for period in periods:
        print(f"\n{'=' * 100}")
        print(f"PERIOD: {period['name']}")
        print(f"{'=' * 100}")
        print(f"{'Config':>15} | {'Trades':>6} | {'WinRate':>7} | {'Gross P&L':>12} | {'Costs':>10} | {'Net P&L':>12} | {'NetExp':>8} | {'PF':>5} | {'MaxDD':>12} | {'AvgHold':>7}")
        print("-" * 120)
        
        for cfg in configs:
            trades_df = run_swing_backtest(
                df_all,
                start_date=period['start'],
                end_date=period['end'],
                top_n=cfg['top_n'],
                hold_days=cfg['hold_days'],
                lookback=cfg['lookback'],
                rebalance_every=cfg['rebalance'],
            )
            
            if trades_df.empty:
                print(f"{cfg['label']:>15} | NO TRADES")
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
            
            print(f"{cfg['label']:>15} | {n:>6} | {wins/n*100:>6.1f}% | {gross:>12.0f} | {costs:>10.0f} | {net:>12.0f} | {net/n:>8.1f} | {pf:>5.2f} | {maxdd:>12.0f} | {avg_hold:>6.1f}d")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_v20()
