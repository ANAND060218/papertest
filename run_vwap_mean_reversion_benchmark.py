"""
V14 Intraday VWAP Mean-Reversion Scientific Benchmark
Evaluates fading extreme statistical VWAP band excursions back to equilibrium midline under full statutory costs.

Cohorts:
  - Cohort 1: Core In-Universe Stocks (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN)
  - Cohort 2: Unseen Out-of-Sample Stocks (AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC)
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
from core.vwap_mean_reversion_engine import VWAPMeanReversionEngine
from core.labeler import TradeLabeler
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

    return total_brokerage + stt + stamp_duty + gst + exchange_txn + sebi_turnover


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


def simulate_reversion_cohort(prepared_stock_data, raw_stock_data, sigma=2.0):
    """Simulates VWAP mean-reversion setups."""
    labeler = TradeLabeler()
    engine = VWAPMeanReversionEngine()

    all_setups = []
    for symbol, df_prep in prepared_stock_data.items():
        setups = engine.generate_reversion_setups(df_prep, symbol=symbol, sigma_threshold=sigma)
        all_setups.extend(setups)

    if not all_setups:
        return {'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0, 'ev': 0.0}

    # Label trades forward
    labeled_trades = []
    for symbol, df_raw in raw_stock_data.items():
        sym_setups = [s for s in all_setups if s['symbol'] == symbol]
        if sym_setups:
            labeled = labeler.label_setups(sym_setups, df_raw)
            labeled_trades.extend(labeled)

    trade_records = []
    for trade in labeled_trades:
        entry_p = trade['entry_price']
        stop_p = trade['stop_price']
        raw_exit = trade['exit_price']
        direction = trade.get('direction', 'LONG')

        if direction == 'LONG':
            fill_entry = entry_p * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            fill_exit = raw_exit * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            gross_ret = (fill_exit - fill_entry) / fill_entry
        else:
            fill_entry = entry_p * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            fill_exit = raw_exit * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            gross_ret = (fill_entry - fill_exit) / fill_entry

        quantity = DynamicRiskEngine.calculate_position_size(
            entry_price=fill_entry,
            stop_price=stop_p,
            capital=config.INITIAL_CAPITAL,
            risk_pct=0.01,
            max_cap_pct=0.10
        )
        if quantity == 0:
            quantity = 1

        gross_pnl_rs = gross_ret * (fill_entry * quantity)
        costs_rs = compute_trade_costs(fill_entry, fill_exit, quantity)
        net_pnl_rs = gross_pnl_rs - costs_rs

        trade_records.append({
            'gross_pnl_rs': gross_pnl_rs,
            'costs_rs': costs_rs,
            'net_pnl_rs': net_pnl_rs,
            'is_win': net_pnl_rs > 0
        })

    n = len(trade_records)
    if n == 0:
        return {'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0, 'ev': 0.0}

    wins = [t for t in trade_records if t['is_win']]
    losses = [t for t in trade_records if not t['is_win']]

    win_rate = (len(wins) / n) * 100
    gross_win = sum(t['gross_pnl_rs'] for t in wins)
    gross_loss = abs(sum(t['gross_pnl_rs'] for t in losses))
    tot_costs = sum(t['costs_rs'] for t in trade_records)
    net_pnl = sum(t['net_pnl_rs'] for t in trade_records)
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else 99.0
    ev = round(net_pnl / n, 2)

    avg_win = round(float(np.mean([t['net_pnl_rs'] for t in wins])), 2) if wins else 0.0
    avg_loss = round(float(np.mean([t['net_pnl_rs'] for t in losses])), 2) if losses else 0.0
    payoff = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    pnls = [t['net_pnl_rs'] for t in trade_records]
    ci_l, ci_u = bootstrap_ci(pnls)

    return {
        'trades': n,
        'win_rate': round(win_rate, 2),
        'profit_factor': pf,
        'gross_win': round(gross_win, 2),
        'gross_loss': round(gross_loss, 2),
        'total_costs': round(tot_costs, 2),
        'net_pnl': round(net_pnl, 2),
        'ev': ev,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff,
        'ci_95': f"[{ci_l}, {ci_u}]"
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
            pass
    conn.close()
    return stock_data


def run_vwap_reversion_benchmark():
    print("\n" + "=" * 125)
    print("V14 INTRADAY VWAP MEAN-REVERSION BENCHMARK")
    print("Fading Extreme +/-2.0 and +/-2.5 Sigma Bands Back to Equilibrium Under Full Statutory Costs")
    print("=" * 125)

    core_db = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")
    core_stocks = config.UNIVERSE
    core_stock_data = load_universe_data(core_db, core_stocks)

    unseen_db = os.path.join(config.DATA_DIR, "unseen_universe_5m.db")
    unseen_stocks = ["AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "LT.NS", "WIPRO.NS", "ITC.NS"]
    unseen_stock_data = load_universe_data(unseen_db, unseen_stocks)

    print(f"Loaded Core ({len(core_stock_data)} stocks) and Unseen ({len(unseen_stock_data)} stocks).")

    print("\n[PRECOMPUTING] Building intraday VWAP standard deviation bands...")
    core_prep = {s: VWAPMeanReversionEngine.prepare_dataset(df) for s, df in core_stock_data.items()}
    unseen_prep = {s: VWAPMeanReversionEngine.prepare_dataset(df) for s, df in unseen_stock_data.items()}
    print("[PRECOMPUTING] Ready.\n")

    # Benchmark 2.0 Sigma vs 2.5 Sigma
    variants = [
        ('Core Universe (2.0-Sigma Fade)', core_prep, core_stock_data, 2.0),
        ('Core Universe (2.5-Sigma Fade)', core_prep, core_stock_data, 2.5),
        ('Unseen Universe (2.0-Sigma Fade)', unseen_prep, unseen_stock_data, 2.0),
        ('Unseen Universe (2.5-Sigma Fade)', unseen_prep, unseen_stock_data, 2.5),
    ]

    results = []
    print("=" * 125)
    print(f"{'Cohort / Configuration':<36} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<12} | {'EV/Tr (Rs)':<10} | {'Payoff':<6} | {'95% Bootstrap CI'}")
    print("-" * 125)

    for name, prep_dict, raw_dict, sig in variants:
        res = simulate_reversion_cohort(prep_dict, raw_dict, sigma=sig)
        res['name'] = name
        results.append(res)
        print(f"{name:<36} | {res['trades']:<6} | {res['win_rate']:>5.1f}% | {res['profit_factor']:>6.3f} | Rs {res['net_pnl']:>9.2f} | Rs {res['ev']:>7.2f} | {res['payoff_ratio']:>5.2f}x | {res['ci_95']}")
    print("=" * 125)

    out_json = os.path.join(config.RESULTS_DIR, "v14_vwap_reversion_report.json")
    out_csv = os.path.join(config.RESULTS_DIR, "v14_vwap_reversion_summary.csv")

    with open(out_json, "w") as f:
        json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"), "results": results}, f, indent=2)

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n[REPORT SAVED] -> {out_json}")
    print(f"[SUMMARY CSV]  -> {out_csv}")


if __name__ == "__main__":
    run_vwap_reversion_benchmark()
