"""
V13 10-Year Multi-Day Swing Momentum Scientific Benchmark
Tests Stage-2 Daily Momentum Breakouts across 40 Top Indian Stocks from 2016 to 2026 (98,000+ daily bars).

Strict Temporal Cohorts:
  - Cohort 1 (In-Sample Development)   : 2016-08-16 to 2022-12-31 (6.5 Years)
  - Cohort 2 (Out-of-Sample Validation): 2023-01-01 to 2024-12-31 (2.0 Years)
  - Cohort 3 (Final Untouched OOS)     : 2025-01-01 to 2026-08-14 (1.6 Years)

Includes full statutory delivery costs:
  - STT: 0.10% on buy & sell turnover
  - Stamp Duty: 0.015% on buy
  - Exchange & SEBI: 0.0035%
  - Brokerage: min(20, 0.03%)
  - Slippage: 0.05%
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.daily_swing_engine import DailySwingEngine
from risk.dynamic_risk import DynamicRiskEngine


def compute_swing_costs(entry_price, exit_price, quantity):
    """Calculates full statutory Indian delivery/swing equity costs."""
    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    total_turnover = buy_turnover + sell_turnover

    brokerage = min(20.0, buy_turnover * config.BROKERAGE_PCT) + min(20.0, sell_turnover * config.BROKERAGE_PCT)
    stt = (buy_turnover + sell_turnover) * 0.0010  # 0.10% delivery STT on both legs
    stamp_duty = buy_turnover * 0.00015             # 0.015% delivery stamp duty
    gst = brokerage * config.GST_ON_BROKERAGE_PCT
    exchange_txn = total_turnover * config.EXCHANGE_TXN_PCT
    sebi_turnover = total_turnover * config.SEBI_TURNOVER_PCT

    return brokerage + stt + stamp_duty + gst + exchange_txn + sebi_turnover


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    """Computes 95% Bootstrap Confidence Interval."""
    if len(values) < 5:
        return (0.0, 0.0)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return (round(float(lower), 2), round(float(upper), 2))


def evaluate_swing_cohort(all_trades, start_date, end_date, cohort_name):
    """Evaluates trades strictly executed within a date range."""
    cohort_trades = [t for t in all_trades if pd.to_datetime(start_date) <= pd.to_datetime(t['entry_date']) <= pd.to_datetime(end_date)]

    n = len(cohort_trades)
    if n == 0:
        return {'cohort': cohort_name, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0, 'ev': 0.0}

    wins = [t for t in cohort_trades if t['net_pnl_rs'] > 0]
    losses = [t for t in cohort_trades if t['net_pnl_rs'] <= 0]

    win_rate = (len(wins) / n) * 100
    gross_win = sum(t['gross_pnl_rs'] for t in wins)
    gross_loss = abs(sum(t['gross_pnl_rs'] for t in losses))
    tot_costs = sum(t['costs_rs'] for t in cohort_trades)
    net_pnl = sum(t['net_pnl_rs'] for t in cohort_trades)
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else 99.0
    ev = round(net_pnl / n, 2)
    ev_pct = round(float(np.mean([t['net_ret_pct'] * 100 for t in cohort_trades])), 2)

    avg_win = round(float(np.mean([t['net_pnl_rs'] for t in wins])), 2) if wins else 0.0
    avg_loss = round(float(np.mean([t['net_pnl_rs'] for t in losses])), 2) if losses else 0.0
    payoff = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    avg_hold_days = round(float(np.mean([t['days_held'] for t in cohort_trades])), 1)

    # Max Drawdown
    pnls = [t['net_pnl_rs'] for t in cohort_trades]
    cum_eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_eq)
    max_dd = round(float(np.max(peak - cum_eq)), 2) if len(pnls) > 0 else 0.0

    ci_l, ci_u = bootstrap_ci(pnls)

    return {
        'cohort': cohort_name,
        'start_date': start_date,
        'end_date': end_date,
        'trades': n,
        'win_rate': round(win_rate, 2),
        'profit_factor': pf,
        'net_pnl': round(net_pnl, 2),
        'ev_rs': ev,
        'ev_pct': ev_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff,
        'avg_hold_days': avg_hold_days,
        'max_dd': max_dd,
        'ci_95': f"[{ci_l}, {ci_u}]"
    }


def run_swing_benchmark():
    print("\n" + "=" * 125)
    print("V13 10-YEAR MULTI-DAY SWING MOMENTUM BENCHMARK (40 NIFTY STOCKS)")
    print("Testing Daily Stage-2 Momentum Breakouts Under Full Statutory Delivery Costs & Slippage")
    print("=" * 125)

    db_path = os.path.join(config.DATA_DIR, "nifty_10year_stock_market.db")
    conn = sqlite3.connect(db_path)
    df_all = pd.read_sql_query("SELECT * FROM stock_daily_10y ORDER BY Symbol, Date ASC", conn)
    conn.close()

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    symbols = df_all['Symbol'].unique()
    print(f"Loaded 10-Year Database: {len(df_all)} daily bars across {len(symbols)} stocks (2016-2026).")

    print("\n[PRECOMPUTING] Calculating 50/200 EMA, Donchian levels, and volume metrics...")
    all_simulated_trades = []

    for sym in symbols:
        sym_df = df_all[df_all['Symbol'] == sym].copy().sort_values('Date').reset_index(drop=True)
        if len(sym_df) < 250:
            continue

        prep_df = DailySwingEngine.prepare_daily_features(sym_df)
        setups = DailySwingEngine.generate_swing_setups(prep_df, symbol=sym)

        for s in setups:
            sim = DailySwingEngine.simulate_swing_trade(s, prep_df)
            entry_p = s['entry_price'] * 1.0005  # 0.05% slippage on entry
            exit_p = sim['exit_price'] * 0.9995   # 0.05% slippage on exit
            raw_ret = (exit_p - entry_p) / entry_p

            # Position sizing: 2% risk of 100k capital
            stop_dist = abs(entry_p - s['stop_price'])
            risk_amt = 100000.0 * 0.02
            qty = int(risk_amt / stop_dist) if stop_dist > 0 else 1
            max_qty = int(25000.0 / entry_p)  # Max 25% capital per position
            qty = max(1, min(qty, max_qty))

            gross_pnl_rs = raw_ret * (entry_p * qty)
            costs_rs = compute_swing_costs(entry_p, exit_p, qty)
            net_pnl_rs = gross_pnl_rs - costs_rs
            net_ret_pct = net_pnl_rs / (entry_p * qty)

            all_simulated_trades.append({
                'symbol': sym,
                'entry_date': s['entry_date'],
                'exit_date': sim['exit_date'],
                'entry_price': entry_p,
                'exit_price': exit_p,
                'gross_pnl_rs': gross_pnl_rs,
                'costs_rs': costs_rs,
                'net_pnl_rs': net_pnl_rs,
                'net_ret_pct': net_ret_pct,
                'days_held': sim['days_held'],
                'exit_reason': sim['exit_reason']
            })

    # Sort trades chronologically
    all_simulated_trades = sorted(all_simulated_trades, key=lambda x: x['entry_date'])
    print(f"Generated and simulated {len(all_simulated_trades)} multi-day swing trades across 10 years.")

    # Evaluate across 3 Strict Temporal Cohorts
    cohort_1 = evaluate_swing_cohort(all_simulated_trades, "2016-08-16", "2022-12-31", "In-Sample (2016-2022)")
    cohort_2 = evaluate_swing_cohort(all_simulated_trades, "2023-01-01", "2024-12-31", "OOS Validation (2023-2024)")
    cohort_3 = evaluate_swing_cohort(all_simulated_trades, "2025-01-01", "2026-08-14", "Final Untouched OOS (2025-2026)")
    full_10y = evaluate_swing_cohort(all_simulated_trades, "2016-08-16", "2026-08-14", "Full 10-Year Combined")

    results = [cohort_1, cohort_2, cohort_3, full_10y]

    # Print Summary Table
    print("\n" + "=" * 125)
    print("FINAL 10-YEAR MULTI-DAY SWING MOMENTUM SUMMARY")
    print(f"{'Cohort Period':<30} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<14} | {'EV/Tr (Rs)':<11} | {'EV %':<7} | {'Payoff':<6} | {'Avg Hold':<8} | {'95% Bootstrap CI'}")
    print("-" * 125)
    for r in results:
        print(f"{r['cohort']:<30} | {r['trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs {r['net_pnl']:>10.2f} | Rs {r['ev_rs']:>8.2f} | {r['ev_pct']:>5.2f}% | {r['payoff_ratio']:>5.2f}x | {r['avg_hold_days']:>5.1f} d | {r['ci_95']}")
    print("=" * 125)

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v13_daily_swing_report.json")
    out_csv = os.path.join(config.RESULTS_DIR, "v13_daily_swing_summary.csv")

    with open(out_json, "w") as f:
        json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"), "results": results}, f, indent=2)

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n[REPORT SAVED] -> {out_json}")
    print(f"[SUMMARY CSV]  -> {out_csv}")


if __name__ == "__main__":
    run_swing_benchmark()
