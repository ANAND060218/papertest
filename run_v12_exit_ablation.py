"""
V12 Dynamic Exit & Profit Capture Benchmark
Evaluates 8 path-dependent exit policies on frozen breakout entries under full statutory costs and slippage.

Includes full statistical metrics:
  - Expectancy in R
  - Median Trade P&L
  - Average Winner vs Average Loser
  - Payoff Ratio
  - Maximum Strategy Drawdown
  - 1,000-Resample Bootstrap 95% Confidence Interval for Net EV
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
from core.v11_quality_engine import V11QualityEngine
from core.v12_exit_engine import V12ExitEngine
from risk.dynamic_risk import DynamicRiskEngine


def compute_trade_costs(entry_price, exit_price, quantity):
    """Calculates full statutory Indian market intraday equity costs."""
    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    total_turnover = buy_turnover + sell_turnover

    brokerage_buy = min(20.0, buy_turnover * config.BROKERAGE_PCT)
    brokerage_sell = min(20.0, sell_turnover * config.BROKERAGE_PCT)
    total_brokerage = brokerage_buy + brokerage_sell

    stt = sell_turnover * config.STT_SELL_PCT
    stamp_duty = buy_turnover * config.STAMP_DUTY_BUY_PCT
    gst = total_brokerage * config.GST_ON_BROKERAGE_PCT
    exchange_txn = total_turnover * config.EXCHANGE_TXN_PCT
    sebi_turnover = total_turnover * config.SEBI_TURNOVER_PCT

    total_costs = total_brokerage + stt + stamp_duty + gst + exchange_txn + sebi_turnover
    return total_costs


def bootstrap_ci_ev(net_pnls, n_boot=1000, alpha=0.05):
    """Computes 95% Bootstrap Confidence Interval for Net Expected Value (EV)."""
    if len(net_pnls) < 5:
        return (0.0, 0.0)
    means = []
    n = len(net_pnls)
    for _ in range(n_boot):
        sample = np.random.choice(net_pnls, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return (round(float(lower), 2), round(float(upper), 2))


def simulate_exit_variant(variant_code, candidate_setups, raw_stock_data):
    """
    Simulates a specific exit policy across all candidate trade setups.
    """
    trades_pnl = []
    trades_records = []

    for trade in candidate_setups:
        symbol = trade['symbol']
        raw_df = raw_stock_data[symbol]
        entry_idx = trade['bar_index']
        trade_date = trade['trade_date']
        direction = trade.get('direction', 'LONG')

        # Slice future bars for the same day
        future_df = raw_df.iloc[entry_idx + 1:].copy()
        if 'trade_date' not in future_df.columns:
            future_df['trade_date'] = future_df['Date'].dt.date
        future_df = future_df[future_df['trade_date'] == trade_date]

        # Simulate path-dependent exit
        sim_res = V12ExitEngine.simulate_trade_exit(trade, future_df, exit_policy=variant_code)

        raw_ret_pct = sim_res['net_pnl_pct']
        entry_price = trade['entry_price']

        # Apply Slippage & Spread to execution
        if direction == 'LONG':
            fill_entry = entry_price * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            # Exit price derived from simulated return percentage
            sim_exit_price = entry_price * (1 + raw_ret_pct)
            fill_exit = sim_exit_price * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            realized_ret_pct = (fill_exit - fill_entry) / fill_entry
        else:
            fill_entry = entry_price * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            sim_exit_price = entry_price * (1 - raw_ret_pct)
            fill_exit = sim_exit_price * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            realized_ret_pct = (fill_entry - fill_exit) / fill_entry

        # Sizing with 1% risk capital
        quantity = DynamicRiskEngine.calculate_position_size(
            entry_price=fill_entry,
            stop_price=fill_entry * 0.995,
            capital=config.INITIAL_CAPITAL,
            risk_pct=0.01,
            max_cap_pct=0.10
        )
        if quantity == 0:
            quantity = 1

        gross_pnl_rs = realized_ret_pct * (fill_entry * quantity)
        costs_rs = compute_trade_costs(fill_entry, fill_exit, quantity)
        net_pnl_rs = gross_pnl_rs - costs_rs

        trades_pnl.append(net_pnl_rs)
        trades_records.append({
            'gross_pnl_rs': gross_pnl_rs,
            'costs_rs': costs_rs,
            'net_pnl_rs': net_pnl_rs,
            'is_win': net_pnl_rs > 0,
            'r_multiple': sim_res['r_multiple'],
            'bars_held': sim_res['bars_held'],
            'exit_reason': sim_res['exit_reason']
        })

    n = len(trades_records)
    if n == 0:
        return {}

    wins = [t for t in trades_records if t['is_win']]
    losses = [t for t in trades_records if not t['is_win']]

    win_rate = (len(wins) / n) * 100
    gross_win = sum(t['gross_pnl_rs'] for t in wins)
    gross_loss = abs(sum(t['gross_pnl_rs'] for t in losses))
    tot_costs = sum(t['costs_rs'] for t in trades_records)
    net_pnl = sum(trades_pnl)
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    ev = round(net_pnl / n, 2)
    ev_r = round(float(np.mean([t['r_multiple'] for t in trades_records])), 3)
    median_pnl = round(float(np.median(trades_pnl)), 2)

    avg_win = round(float(np.mean([t['net_pnl_rs'] for t in wins])), 2) if wins else 0.0
    avg_loss = round(float(np.mean([t['net_pnl_rs'] for t in losses])), 2) if losses else 0.0
    payoff_ratio = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    # Max Drawdown
    cum_equity = np.cumsum(trades_pnl)
    peak = np.maximum.accumulate(cum_equity)
    drawdowns = peak - cum_equity
    max_dd = round(float(np.max(drawdowns)), 2) if len(drawdowns) > 0 else 0.0

    # Bootstrap 95% CI
    ci_lower, ci_upper = bootstrap_ci_ev(trades_pnl)

    return {
        'variant': variant_code,
        'trades': n,
        'win_rate': round(win_rate, 2),
        'profit_factor': pf,
        'net_pnl': round(net_pnl, 2),
        'ev': ev,
        'ev_r': ev_r,
        'median_pnl': median_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff_ratio,
        'max_dd': max_dd,
        'ci_95': f"[{ci_lower}, {ci_upper}]"
    }


def load_universe_data(db_path, symbols):
    """Loads stock data dict from SQLite database."""
    stock_data = {}
    if not os.path.exists(db_path):
        return stock_data

    conn = sqlite3.connect(db_path)
    for sym in symbols:
        clean_name = sym.replace('.NS', '').replace('^', 'IDX_')
        table_name = f"bars_5m_{clean_name}"
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY Date ASC", conn)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Symbol'] = sym
                stock_data[sym] = df
        except Exception:
            try:
                df = pd.read_sql_query(
                    "SELECT * FROM universe_intraday_5m WHERE Symbol = ? ORDER BY Date ASC",
                    conn, params=(sym,)
                )
                if not df.empty:
                    df['Date'] = pd.to_datetime(df['Date'])
                    stock_data[sym] = df
            except Exception:
                pass
    conn.close()
    return stock_data


def run_v12_exit_ablation():
    print("\n" + "=" * 125)
    print("V12 DYNAMIC EXIT & PROFIT CAPTURE SCIENTIFIC BENCHMARK")
    print("Evaluating 8 Path-Dependent Exit Policies on Frozen Breakout Setups Under Full Statutory Costs")
    print("=" * 125)

    engine = V11QualityEngine()

    core_db = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")
    core_stocks = config.UNIVERSE
    core_stock_data = load_universe_data(core_db, core_stocks)

    unseen_db = os.path.join(config.DATA_DIR, "unseen_universe_5m.db")
    unseen_stocks = ["AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "LT.NS", "WIPRO.NS", "ITC.NS"]
    unseen_stock_data = load_universe_data(unseen_db, unseen_stocks)

    index_data = load_universe_data(core_db, ["^NSEI"])
    nifty_df = index_data.get("^NSEI", None)

    print(f"Loaded Core Universe ({len(core_stock_data)} symbols) & Unseen Universe ({len(unseen_stock_data)} symbols).")

    # Generate Frozen Candidate Setups from V11 Baseline (V11-A provides full sample pool)
    print("\n[PRECOMPUTING] Generating frozen breakout candidate pool...")
    core_prep = {s: engine.prepare_dataset(df, nifty_df=nifty_df) for s, df in core_stock_data.items()}
    unseen_prep = {s: engine.prepare_dataset(df, nifty_df=nifty_df) for s, df in unseen_stock_data.items()}

    core_setups = []
    for s, df in core_prep.items():
        core_setups.extend(engine.generate_v11_setups(df, variant='V11_A', symbol=s))

    unseen_setups = []
    for s, df in unseen_prep.items():
        unseen_setups.extend(engine.generate_v11_setups(df, variant='V11_A', symbol=s))

    print(f"Frozen Pool: {len(core_setups)} Core Setups | {len(unseen_setups)} Unseen Setups.\n")

    exit_variants = [
        ('V12_A', 'Baseline Fixed Exit (SL -0.50%, TP +1.00%)'),
        ('V12_B', 'MFE-Calibrated Target (SL -0.50%, TP +0.50%)'),
        ('V12_C', 'Partial Exit 50% @ +0.40% + BE + Runner @ +1.0%'),
        ('V12_D', 'MFE Trailing Stop (Trigger @ +0.30%, Trail 0.20%)'),
        ('V12_E', 'Time-Based Exit (Max 4 bars / 20 min if < +0.30%)'),
        ('V12_F', 'Partial Exit 50% @ +0.40% + Trailing Runner'),
        ('V12_G', 'MFE-Adaptive Multi-Stage Policy'),
        ('V12_H', 'ATR-Calibrated Volatility Dynamic Exit')
    ]

    # --- Cohort 1: Core Universe ---
    print("#" * 125)
    print("COHORT 1: CORE IN-UNIVERSE DATASET (June-July 2026)")
    print("#" * 125)
    core_results = []
    for v_code, v_desc in exit_variants:
        res = simulate_exit_variant(v_code, core_setups, core_stock_data)
        res['description'] = v_desc
        core_results.append(res)
        print(f"  [{v_code:<8}] Trades: {res['trades']:<3} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['ev']:>6.2f} | EV(R): {res['ev_r']:>5.2f}R | 95% CI: {res['ci_95']}")

    # --- Cohort 2: Unseen Universe ---
    print("\n" + "#" * 125)
    print("COHORT 2: UNSEEN OUT-OF-SAMPLE STOCKS (AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC)")
    print("#" * 125)
    unseen_results = []
    for v_code, v_desc in exit_variants:
        res = simulate_exit_variant(v_code, unseen_setups, unseen_stock_data)
        res['description'] = v_desc
        unseen_results.append(res)
        print(f"  [{v_code:<8}] Trades: {res['trades']:<3} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['ev']:>6.2f} | EV(R): {res['ev_r']:>5.2f}R | 95% CI: {res['ci_95']}")

    # Master Table Presentation
    print("\n" + "=" * 125)
    print("FINAL SUMMARY: CORE UNIVERSE")
    print(f"{'Variant':<8} | {'Exit Architecture':<50} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L':<10} | {'EV (Rs)':<8} | {'EV (R)':<7} | {'Payoff':<6} | {'Max DD':<9}")
    print("-" * 125)
    for r in core_results:
        print(f"{r['variant']:<8} | {r['description']:<50} | {r['trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs{r['net_pnl']:>7.1f} | Rs{r['ev']:>6.2f} | {r['ev_r']:>5.2f}R | {r['payoff_ratio']:>5.2f}x | Rs{r['max_dd']:>7.1f}")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("FINAL SUMMARY: UNSEEN UNIVERSE (TRUE GENERALIZATION)")
    print(f"{'Variant':<8} | {'Exit Architecture':<50} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L':<10} | {'EV (Rs)':<8} | {'EV (R)':<7} | {'Payoff':<6} | {'Max DD':<9}")
    print("-" * 125)
    for r in unseen_results:
        print(f"{r['variant']:<8} | {r['description']:<50} | {r['trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs{r['net_pnl']:>7.1f} | Rs{r['ev']:>6.2f} | {r['ev_r']:>5.2f}R | {r['payoff_ratio']:>5.2f}x | Rs{r['max_dd']:>7.1f}")
    print("=" * 125)

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v12_exit_ablation_report.json")
    out_csv = os.path.join(config.RESULTS_DIR, "v12_exit_ablation_summary.csv")

    summary_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "core_universe_results": core_results,
        "unseen_universe_results": unseen_results
    }

    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    df_combined = pd.DataFrame([
        {**r, "Cohort": "Core_Universe"} for r in core_results
    ] + [
        {**r, "Cohort": "Unseen_Universe"} for r in unseen_results
    ])
    df_combined.to_csv(out_csv, index=False)

    print(f"\n[REPORT SAVED] -> {out_json}")
    print(f"[SUMMARY CSV]  -> {out_csv}")


if __name__ == "__main__":
    run_v12_exit_ablation()
