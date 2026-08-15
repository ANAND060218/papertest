"""
V19 Stress Validation Suite
============================
Four-gate validation of V19 Statistical Arbitrage before acceptance:

Gate 1: Walk-Forward Hedge Ratio Retraining (6-month windows)
Gate 2: Cointegration Stability (rolling ADF through OOS)
Gate 3: Slippage Stress Test (0% to 0.15%)
Gate 4: Monte Carlo Reshuffling (2000 iterations)
Gate 5: Leave-One-Out Pair Concentration
Gate 6: V15.2 vs V19 Comparative Scorecard
"""
import sys, os
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from v19.research.v19_pairs_engine import (
    test_cointegration, calculate_pair_trade_costs, backtest_pair
)

# Sector groups (same as v19_pairs_engine)
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


def compute_trade_stats(trades_df):
    """Compute standard stats from a trades DataFrame."""
    if trades_df.empty or len(trades_df) == 0:
        return {'trades': 0, 'net_pnl': 0, 'pf': 0, 'win_rate': 0, 'max_dd': 0}
    
    n = len(trades_df)
    wins = (trades_df['net_pnl'] > 0).sum()
    net = trades_df['net_pnl'].sum()
    gp = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
    gl = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    cum = trades_df['net_pnl'].cumsum()
    maxdd = (cum - cum.cummax()).min() if len(cum) > 0 else 0
    
    return {
        'trades': n, 'net_pnl': net, 'pf': pf,
        'win_rate': wins / n * 100 if n > 0 else 0,
        'max_dd': maxdd, 'gross_pnl': trades_df['gross_pnl'].sum(),
        'costs': trades_df['costs'].sum(),
        'avg_hold': trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# GATE 1: Walk-Forward Hedge Ratio Retraining
# ═══════════════════════════════════════════════════════════════════════
def gate1_walk_forward(df_all, stocks):
    """
    Re-estimate hedge ratios every 6 months using a rolling 4-year window.
    Only trade pairs that remain cointegrated at each retraining point.
    """
    print("\n" + "=" * 100)
    print("GATE 1: WALK-FORWARD HEDGE RATIO RETRAINING (6-month windows)")
    print("=" * 100)
    
    # Define 6-month retraining windows
    retrain_dates = [
        ('2017-01-01', '2021-01-01'),  # Train: 2017-2020, Trade: 2021-H1
        ('2017-07-01', '2021-07-01'),  # Train: 2017.5-2021, Trade: 2021-H2
        ('2018-01-01', '2022-01-01'),  # Train: 2018-2021, Trade: 2022-H1
        ('2018-07-01', '2022-07-01'),  # Train: 2018.5-2022, Trade: 2022-H2
        ('2019-01-01', '2023-01-01'),  # Train: 2019-2022, Trade: 2023-H1
        ('2019-07-01', '2023-07-01'),  # Train: 2019.5-2023, Trade: 2023-H2
        ('2020-01-01', '2024-01-01'),  # Train: 2020-2023, Trade: 2024-H1
        ('2020-07-01', '2024-07-01'),  # Train: 2020.5-2024, Trade: 2024-H2
        ('2021-01-01', '2025-01-01'),  # Train: 2021-2024, Trade: 2025-H1
    ]
    
    all_wf_trades = []
    pair_activity = {}  # Track which pairs are active in each window
    
    for train_start, trade_start in retrain_dates:
        train_end = trade_start
        # Trade for 6 months
        trade_end_dt = pd.to_datetime(trade_start) + timedelta(days=183)
        trade_end = trade_end_dt.strftime('%Y-%m-%d')
        
        # Clip to available data
        max_date = df_all['Date'].max()
        if trade_start > max_date:
            break
        if trade_end > max_date:
            trade_end = max_date
        
        print(f"\n  Window: Train [{train_start} -> {train_end}] | Trade [{trade_start} -> {trade_end}]")
        
        # Screen for cointegrated pairs in THIS training window
        active_pairs = []
        for sector, tickers in SECTOR_GROUPS.items():
            available = [t for t in tickers if t in stocks]
            for t1, t2 in combinations(available, 2):
                df_1 = df_all[(df_all['Symbol'] == t1) & 
                              (df_all['Date'] >= train_start) & 
                              (df_all['Date'] < train_end)][['Date', 'Close']].dropna()
                df_2 = df_all[(df_all['Symbol'] == t2) & 
                              (df_all['Date'] >= train_start) & 
                              (df_all['Date'] < train_end)][['Date', 'Close']].dropna()
                
                merged = pd.merge(df_1, df_2, on='Date', suffixes=('_1', '_2'))
                if len(merged) < 200:
                    continue
                
                try:
                    is_coint, p_val, hr, intercept, hl = test_cointegration(
                        merged['Close_1'], merged['Close_2'])
                except Exception:
                    continue
                
                if is_coint and 1 < hl < 60:
                    active_pairs.append({
                        'stock_a': t1, 'stock_b': t2,
                        'hedge_ratio': hr, 'intercept': intercept,
                        'half_life': hl, 'p_value': p_val,
                    })
        
        print(f"    Active pairs this window: {len(active_pairs)}")
        
        # Backtest active pairs on the 6-month OOS window
        for pair in active_pairs:
            pair_key = f"{pair['stock_a']}/{pair['stock_b']}"
            
            df_a = df_all[(df_all['Symbol'] == pair['stock_a']) & 
                          (df_all['Date'] >= trade_start) & 
                          (df_all['Date'] <= trade_end)][['Date', 'Close']].dropna()
            df_b = df_all[(df_all['Symbol'] == pair['stock_b']) & 
                          (df_all['Date'] >= trade_start) & 
                          (df_all['Date'] <= trade_end)][['Date', 'Close']].dropna()
            
            if len(df_a) < 30 or len(df_b) < 30:
                continue
            
            trades_df = backtest_pair(
                df_a, df_b, pair['stock_a'], pair['stock_b'],
                pair['hedge_ratio'], pair['intercept'],
                lookback=60, z_entry=2.0, z_exit=0.5, z_stop=4.0,
                capital_per_leg=100000
            )
            
            if not trades_df.empty:
                trades_df['pair'] = pair_key
                trades_df['window'] = f"{trade_start[:7]}"
                all_wf_trades.append(trades_df)
                
                if pair_key not in pair_activity:
                    pair_activity[pair_key] = 0
                pair_activity[pair_key] += 1
    
    if not all_wf_trades:
        print("  NO TRADES GENERATED IN WALK-FORWARD.")
        return pd.DataFrame()
    
    wf_trades = pd.concat(all_wf_trades, ignore_index=True)
    stats = compute_trade_stats(wf_trades)
    
    print(f"\n  WALK-FORWARD RESULTS:")
    print(f"    Total Trades:  {stats['trades']}")
    print(f"    Net P&L:       Rs {stats['net_pnl']:.0f}")
    print(f"    Profit Factor: {stats['pf']:.2f}")
    print(f"    Win Rate:      {stats['win_rate']:.1f}%")
    print(f"    Max Drawdown:  Rs {stats['max_dd']:.0f}")
    print(f"    Costs:         Rs {stats['costs']:.0f}")
    
    print(f"\n  Pair Activity (windows active out of {len(retrain_dates)}):")
    for pair_key, count in sorted(pair_activity.items(), key=lambda x: -x[1]):
        print(f"    {pair_key:>35}: {count} windows")
    
    return wf_trades


# ═══════════════════════════════════════════════════════════════════════
# GATE 2: Cointegration Stability
# ═══════════════════════════════════════════════════════════════════════
def gate2_cointegration_stability(df_all, stocks):
    """
    Rolling cointegration test: check if each pair remains statistically
    stable through the OOS period using 2-year rolling windows.
    """
    print("\n" + "=" * 100)
    print("GATE 2: COINTEGRATION STABILITY (Rolling ADF through OOS)")
    print("=" * 100)
    
    # The 7 profitable pairs from initial V19 run
    pairs_to_test = [
        ('HDFCBANK.NS', 'KOTAKBANK.NS'),
        ('ICICIBANK.NS', 'KOTAKBANK.NS'),
        ('SBIN.NS', 'AXISBANK.NS'),
        ('INFY.NS', 'HCLTECH.NS'),
        ('BPCL.NS', 'POWERGRID.NS'),
        ('EICHERMOT.NS', 'HEROMOTOCO.NS'),
        ('ULTRACEMCO.NS', 'ADANIPORTS.NS'),
    ]
    
    # Test cointegration at yearly checkpoints
    checkpoints = ['2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01',
                    '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']
    
    stability = {}
    
    for t1, t2 in pairs_to_test:
        pair_key = f"{t1.replace('.NS','')}/{t2.replace('.NS','')}"
        results = []
        
        for cp in checkpoints:
            # Use 2 years of data ending at checkpoint
            cp_dt = pd.to_datetime(cp)
            start = (cp_dt - timedelta(days=730)).strftime('%Y-%m-%d')
            
            df_1 = df_all[(df_all['Symbol'] == t1) & 
                          (df_all['Date'] >= start) & 
                          (df_all['Date'] < cp)][['Date', 'Close']].dropna()
            df_2 = df_all[(df_all['Symbol'] == t2) & 
                          (df_all['Date'] >= start) & 
                          (df_all['Date'] < cp)][['Date', 'Close']].dropna()
            
            merged = pd.merge(df_1, df_2, on='Date', suffixes=('_1', '_2'))
            if len(merged) < 100:
                results.append({'checkpoint': cp[:4], 'coint': False, 'p': 1.0, 'hl': 0})
                continue
            
            try:
                is_coint, p_val, hr, intercept, hl = test_cointegration(
                    merged['Close_1'], merged['Close_2'])
                results.append({'checkpoint': cp[:4], 'coint': is_coint, 'p': p_val, 'hl': hl})
            except Exception:
                results.append({'checkpoint': cp[:4], 'coint': False, 'p': 1.0, 'hl': 0})
        
        stability[pair_key] = results
        
        # Display as timeline
        timeline = ' | '.join([
            f"{r['checkpoint']}:{'Y' if r['coint'] else 'N'}({r['p']:.2f})"
            for r in results
        ])
        stable_count = sum(1 for r in results if r['coint'])
        print(f"  {pair_key:>30}: {timeline} | Stable: {stable_count}/{len(results)}")
    
    return stability


# ═══════════════════════════════════════════════════════════════════════
# GATE 3: Slippage Stress Test
# ═══════════════════════════════════════════════════════════════════════
def gate3_slippage_stress(wf_trades):
    """
    Apply increasing slippage to walk-forward trades and find breakeven.
    """
    print("\n" + "=" * 100)
    print("GATE 3: SLIPPAGE STRESS TEST")
    print("=" * 100)
    
    if wf_trades.empty:
        print("  No trades to stress test.")
        return
    
    slippage_levels = [0.0, 0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003]
    
    print(f"  {'Slippage':>10} | {'Net P&L':>12} | {'PF':>6} | {'WinRate':>8} | {'NetExp/Trd':>10} | {'MaxDD':>12}")
    print("  " + "-" * 75)
    
    breakeven_slippage = None
    
    for slip in slippage_levels:
        # Slippage applied to both legs on both entry and exit (4 legs total)
        # Each leg's slippage = price * slippage_pct * qty
        # Approximate: total slippage per trade = 4 * avg_turnover * slippage_pct
        avg_gross = wf_trades['gross_pnl'].mean()
        avg_cost = wf_trades['costs'].mean()
        
        # Slippage per trade: assume ~Rs 200,000 total turnover (2 legs x Rs 100,000)
        slippage_per_trade = 200000 * slip * 4  # 4 transactions per round trip
        
        adjusted = wf_trades.copy()
        adjusted['net_pnl'] = adjusted['gross_pnl'] - adjusted['costs'] - slippage_per_trade
        
        stats = compute_trade_stats(adjusted)
        
        marker = ""
        if breakeven_slippage is None and stats['net_pnl'] < 0:
            breakeven_slippage = slip
            marker = " <-- BREAKEVEN"
        
        print(f"  {slip*100:>9.2f}% | {stats['net_pnl']:>12.0f} | {stats['pf']:>6.2f} | "
              f"{stats['win_rate']:>7.1f}% | {stats['net_pnl']/max(stats['trades'],1):>10.1f} | "
              f"{stats['max_dd']:>12.0f}{marker}")
    
    if breakeven_slippage is not None:
        print(f"\n  BREAKEVEN SLIPPAGE: {breakeven_slippage*100:.2f}%")
    else:
        print(f"\n  Strategy survives all tested slippage levels up to {slippage_levels[-1]*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════
# GATE 4: Monte Carlo Reshuffling
# ═══════════════════════════════════════════════════════════════════════
def gate4_monte_carlo(wf_trades, n_simulations=2000):
    """
    Reshuffle the trade sequence 2000 times to estimate:
    - Median return
    - 5th/95th percentile
    - Probability of losing money
    - Max drawdown distribution
    - Probability PF < 1
    """
    print("\n" + "=" * 100)
    print(f"GATE 4: MONTE CARLO RESHUFFLING ({n_simulations} simulations)")
    print("=" * 100)
    
    if wf_trades.empty:
        print("  No trades to simulate.")
        return
    
    pnl_array = wf_trades['net_pnl'].values
    n_trades = len(pnl_array)
    
    sim_returns = []
    sim_max_dds = []
    sim_pfs = []
    
    np.random.seed(42)
    
    for _ in range(n_simulations):
        # Reshuffle trade order
        shuffled = np.random.permutation(pnl_array)
        
        total_return = shuffled.sum()
        sim_returns.append(total_return)
        
        # Max drawdown of this shuffled sequence
        cumsum = np.cumsum(shuffled)
        running_max = np.maximum.accumulate(cumsum)
        dd = cumsum - running_max
        sim_max_dds.append(dd.min())
        
        # Profit factor
        gains = shuffled[shuffled > 0].sum()
        losses = abs(shuffled[shuffled < 0].sum())
        pf = gains / losses if losses > 0 else float('inf')
        sim_pfs.append(pf)
    
    sim_returns = np.array(sim_returns)
    sim_max_dds = np.array(sim_max_dds)
    sim_pfs = np.array(sim_pfs)
    
    print(f"  Total Return Distribution:")
    print(f"    Median:          Rs {np.median(sim_returns):>10.0f}")
    print(f"    Mean:            Rs {np.mean(sim_returns):>10.0f}")
    print(f"    5th percentile:  Rs {np.percentile(sim_returns, 5):>10.0f}")
    print(f"    25th percentile: Rs {np.percentile(sim_returns, 25):>10.0f}")
    print(f"    75th percentile: Rs {np.percentile(sim_returns, 75):>10.0f}")
    print(f"    95th percentile: Rs {np.percentile(sim_returns, 95):>10.0f}")
    
    prob_loss = (sim_returns < 0).mean() * 100
    print(f"\n  Probability of losing money: {prob_loss:.1f}%")
    
    print(f"\n  Max Drawdown Distribution:")
    print(f"    Median DD:       Rs {np.median(sim_max_dds):>10.0f}")
    print(f"    5th pctl DD:     Rs {np.percentile(sim_max_dds, 5):>10.0f}")
    print(f"    95th pctl DD:    Rs {np.percentile(sim_max_dds, 95):>10.0f}")
    print(f"    Worst case DD:   Rs {np.min(sim_max_dds):>10.0f}")
    
    prob_pf_below_1 = (sim_pfs < 1.0).mean() * 100
    print(f"\n  Profit Factor Distribution:")
    print(f"    Median PF:       {np.median(sim_pfs):.2f}")
    print(f"    5th pctl PF:     {np.percentile(sim_pfs, 5):.2f}")
    print(f"    95th pctl PF:    {np.percentile(sim_pfs, 95):.2f}")
    print(f"    Probability PF < 1: {prob_pf_below_1:.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# GATE 5: Leave-One-Out Pair Concentration
# ═══════════════════════════════════════════════════════════════════════
def gate5_leave_one_out(wf_trades):
    """
    Remove each pair individually and measure portfolio impact.
    If removing one pair destroys the portfolio, the edge is fragile.
    """
    print("\n" + "=" * 100)
    print("GATE 5: LEAVE-ONE-OUT PAIR CONCENTRATION")
    print("=" * 100)
    
    if wf_trades.empty or 'pair' not in wf_trades.columns:
        print("  No pair-level data available.")
        return
    
    full_stats = compute_trade_stats(wf_trades)
    pairs = wf_trades['pair'].unique()
    
    print(f"  Full portfolio: {full_stats['trades']} trades | Net Rs {full_stats['net_pnl']:.0f} | PF {full_stats['pf']:.2f}")
    print(f"\n  {'Removed Pair':>35} | {'Trades':>6} | {'Net P&L':>12} | {'PF':>6} | {'Impact':>12}")
    print("  " + "-" * 85)
    
    for pair in pairs:
        remaining = wf_trades[wf_trades['pair'] != pair]
        if remaining.empty:
            continue
        stats = compute_trade_stats(remaining)
        impact = stats['net_pnl'] - full_stats['net_pnl']
        
        flag = " *** CRITICAL" if stats['net_pnl'] < 0 else ""
        print(f"  {pair:>35} | {stats['trades']:>6} | {stats['net_pnl']:>12.0f} | "
              f"{stats['pf']:>6.2f} | {impact:>+12.0f}{flag}")


# ═══════════════════════════════════════════════════════════════════════
# GATE 6: V15.2 vs V19 Scorecard
# ═══════════════════════════════════════════════════════════════════════
def gate6_comparative_scorecard(wf_trades, df_all):
    """
    Build a head-to-head comparison between V15.2 and V19.
    """
    print("\n" + "=" * 100)
    print("GATE 6: V15.2 vs V19 COMPARATIVE SCORECARD")
    print("=" * 100)
    
    if wf_trades.empty:
        print("  No V19 data for comparison.")
        return
    
    # V19 metrics (from walk-forward trades)
    v19_stats = compute_trade_stats(wf_trades)
    v19_net = v19_stats['net_pnl']
    
    # Approximate V19 CAGR (assume ~4 years of OOS trading, Rs 700,000 capital for 7 pairs)
    v19_capital = 700000  # 7 pairs x Rs 100,000 per leg
    v19_years = 4.0  # approximate OOS duration
    v19_cagr = ((1 + v19_net / v19_capital) ** (1 / v19_years) - 1) * 100 if v19_capital > 0 else 0
    
    # Annualized Sharpe (approximate)
    if 'window' in wf_trades.columns:
        window_pnl = wf_trades.groupby('window')['net_pnl'].sum()
        if len(window_pnl) > 1:
            v19_sharpe = (window_pnl.mean() / window_pnl.std()) * np.sqrt(2)  # 2 windows/year
        else:
            v19_sharpe = 0
    else:
        v19_sharpe = 0
    
    # V19 Calmar
    v19_calmar = abs(v19_net / v19_years / v19_stats['max_dd']) if v19_stats['max_dd'] < 0 else 0
    
    # V15.2 known metrics (from previous validation)
    v15_cagr = 16.75
    v15_pf = 2.65
    v15_maxdd_pct = -18.5  # approximate from previous results
    v15_winrate = 62.0  # approximate
    v15_avg_hold = 30  # monthly
    v15_sharpe = 0.85  # approximate
    v15_calmar = v15_cagr / abs(v15_maxdd_pct)
    
    print(f"\n  {'Metric':>25} | {'V15.2 Monthly':>15} | {'V19 Stat Arb':>15}")
    print("  " + "-" * 60)
    print(f"  {'OOS CAGR':>25} | {v15_cagr:>14.2f}% | {v19_cagr:>14.2f}%")
    print(f"  {'Profit Factor':>25} | {v15_pf:>15.2f} | {v19_stats['pf']:>15.2f}")
    print(f"  {'Win Rate':>25} | {v15_winrate:>14.1f}% | {v19_stats['win_rate']:>14.1f}%")
    print(f"  {'Sharpe (approx)':>25} | {v15_sharpe:>15.2f} | {v19_sharpe:>15.2f}")
    print(f"  {'Calmar':>25} | {v15_calmar:>15.2f} | {v19_calmar:>15.2f}")
    print(f"  {'Max Drawdown':>25} | {v15_maxdd_pct:>14.1f}% | Rs {v19_stats['max_dd']:>10.0f}")
    print(f"  {'Avg Holding Period':>25} | {'~30 days':>15} | {v19_stats['avg_hold']:>12.1f} days")
    print(f"  {'Total Trades (OOS)':>25} | {'~50':>15} | {v19_stats['trades']:>15}")
    print(f"  {'Total Costs':>25} | {'Low (monthly)':>15} | Rs {v19_stats['costs']:>10.0f}")
    print(f"  {'Edge Source':>25} | {'Momentum':>15} | {'Mean Reversion':>15}")
    print(f"  {'Market Exposure':>25} | {'Long Only':>15} | {'Mkt Neutral':>15}")
    
    # Correlation analysis
    # V15.2 profits when markets trend. V19 profits when spreads revert.
    # These are likely uncorrelated or negatively correlated.
    print(f"\n  PORTFOLIO DIVERSIFICATION POTENTIAL:")
    print(f"    V15.2 edge: Momentum (trends)")
    print(f"    V19 edge:   Mean reversion (spreads)")
    print(f"    Expected correlation: LOW (different alpha sources)")
    print(f"    Combined portfolio could reduce drawdowns significantly")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 100)
    print("V19 STATISTICAL ARBITRAGE — COMPLETE STRESS VALIDATION SUITE")
    print("=" * 100)
    
    df_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'nifty_10year_stacked.csv'))
    stocks = [s for s in df_all['Symbol'].unique() if not s.startswith('^')]
    
    # Gate 1: Walk-Forward
    wf_trades = gate1_walk_forward(df_all, stocks)
    
    # Gate 2: Cointegration Stability
    stability = gate2_cointegration_stability(df_all, stocks)
    
    # Gate 3: Slippage Stress
    gate3_slippage_stress(wf_trades)
    
    # Gate 4: Monte Carlo
    gate4_monte_carlo(wf_trades, n_simulations=2000)
    
    # Gate 5: Leave-One-Out
    gate5_leave_one_out(wf_trades)
    
    # Gate 6: V15.2 vs V19
    gate6_comparative_scorecard(wf_trades, df_all)
    
    # ─── FINAL VERDICT ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    if not wf_trades.empty:
        final_stats = compute_trade_stats(wf_trades)
        if final_stats['pf'] > 1.0 and final_stats['net_pnl'] > 0:
            print(f"  V19 Walk-Forward PF: {final_stats['pf']:.2f}")
            print(f"  V19 Walk-Forward Net: Rs {final_stats['net_pnl']:.0f}")
            print(f"  STATUS: CANDIDATE FOR PAPER TRADING")
        else:
            print(f"  V19 Walk-Forward PF: {final_stats['pf']:.2f}")
            print(f"  V19 Walk-Forward Net: Rs {final_stats['net_pnl']:.0f}")
            print(f"  STATUS: FAILED WALK-FORWARD VALIDATION")
    else:
        print("  STATUS: NO TRADES - REJECTED")
    print("=" * 100)


if __name__ == "__main__":
    main()
