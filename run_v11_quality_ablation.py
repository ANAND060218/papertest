"""
V11 Breakout Quality, Retest & MAE/MFE Microstructure Ablation Benchmark
Isolates post-breakout mechanics, retest confirmation, candle quality, expansion distance, and MAE/MFE analytics.

Variants Benchmarked:
  - V11-A : Baseline Immediate ORB + NIFTY Alignment
  - V11-B : + Breakout Candle Quality Filter (Strong body & small wick)
  - V11-C : + Controlled Breakout Distance (5% to 40% expansion)
  - V11-D : + Opening Range Width Filter (0.35% to 1.80%)
  - V11-E : + Relative Strength vs NIFTY (RS >= 0.25%)
  - V11-F : Breakout + Clean Retest Confirmation
  - V11-G : Master Integrated Quality + Retest Engine

Evaluated under the FULL Indian Statutory Cost Model & 0.05% Slippage.
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

    total_costs = total_brokerage + stt + stamp_duty + gst + exchange_txn + sebi_turnover
    return total_costs


def simulate_v11_variant(variant, prepared_stock_data, raw_stock_data):
    """
    Simulates a V11 strategy variant and computes MAE/MFE distributions.
    """
    labeler = TradeLabeler()
    engine = V11QualityEngine()

    all_setups = []

    for symbol, df_prep in prepared_stock_data.items():
        if df_prep.empty or len(df_prep) < 20:
            continue

        setups = engine.generate_v11_setups(df_prep, variant=variant, symbol=symbol)
        all_setups.extend(setups)

    if not all_setups:
        return {
            'variant': variant, 'total_setups': 0, 'accepted_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0, 'expectancy': 0.0,
            'gross_win': 0.0, 'gross_loss': 0.0, 'total_costs': 0.0,
            'avg_mae': 0.0, 'avg_mfe': 0.0, 'mfe_mae_ratio': 0.0,
            'pct_hit_05_mfe': 0.0, 'pct_hit_10_mfe': 0.0
        }

    # Label setups forward
    labeled_trades = []
    for symbol, df_raw in raw_stock_data.items():
        sym_setups = [s for s in all_setups if s['symbol'] == symbol]
        if sym_setups:
            labeled = labeler.label_setups(sym_setups, df_raw)
            # Compute MAE/MFE for each trade
            for t in labeled:
                excursion = engine.calculate_mae_mfe(t, df_raw)
                t['mae_pct'] = excursion['mae_pct']
                t['mfe_pct'] = excursion['mfe_pct']
            labeled_trades.extend(labeled)

    accepted_trades = []

    for trade in labeled_trades:
        entry_price = trade['entry_price']
        direction = trade.get('direction', 'LONG')
        stop_price = trade.get('stop_price', entry_price * 0.996)
        raw_exit = trade['exit_price']

        if direction == 'LONG':
            fill_entry = entry_price * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            fill_exit = raw_exit * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            gross_pnl_pct = (fill_exit - fill_entry) / fill_entry
        else:
            fill_entry = entry_price * (1 - config.SLIPPAGE_PCT - config.SPREAD_PCT / 2)
            fill_exit = raw_exit * (1 + config.SLIPPAGE_PCT + config.SPREAD_PCT / 2)
            gross_pnl_pct = (fill_entry - fill_exit) / fill_entry

        quantity = DynamicRiskEngine.calculate_position_size(
            entry_price=fill_entry,
            stop_price=stop_price,
            capital=config.INITIAL_CAPITAL,
            risk_pct=0.01,
            max_cap_pct=0.10
        )
        if quantity == 0:
            quantity = 1

        gross_pnl_rs = gross_pnl_pct * (fill_entry * quantity)
        costs_rs = compute_trade_costs(fill_entry, fill_exit, quantity)
        net_pnl_rs = gross_pnl_rs - costs_rs

        trade_record = {
            'trade': trade,
            'direction': direction,
            'is_win': net_pnl_rs > 0,
            'gross_pnl_rs': gross_pnl_rs,
            'costs_rs': costs_rs,
            'net_pnl_rs': net_pnl_rs,
            'mae_pct': trade['mae_pct'],
            'mfe_pct': trade['mfe_pct']
        }
        accepted_trades.append(trade_record)

    n_acc = len(accepted_trades)

    if n_acc > 0:
        wins = [t for t in accepted_trades if t['is_win']]
        win_rate = (len(wins) / n_acc) * 100
        gross_win = sum(t['gross_pnl_rs'] for t in accepted_trades if t['gross_pnl_rs'] > 0)
        gross_loss = abs(sum(t['gross_pnl_rs'] for t in accepted_trades if t['gross_pnl_rs'] < 0))
        tot_costs = sum(t['costs_rs'] for t in accepted_trades)
        net_pnl = sum(t['net_pnl_rs'] for t in accepted_trades)
        pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
        expectancy = round(net_pnl / n_acc, 2)

        # MAE / MFE Metrics
        avg_mae = np.mean([t['mae_pct'] for t in accepted_trades])
        avg_mfe = np.mean([t['mfe_pct'] for t in accepted_trades])
        mfe_mae_ratio = round(abs(avg_mfe / avg_mae), 2) if avg_mae != 0 else 0.0
        pct_05 = sum(1 for t in accepted_trades if t['mfe_pct'] >= 0.50) / n_acc * 100
        pct_10 = sum(1 for t in accepted_trades if t['mfe_pct'] >= 1.00) / n_acc * 100
    else:
        win_rate, gross_win, gross_loss, tot_costs, net_pnl, pf, expectancy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        avg_mae, avg_mfe, mfe_mae_ratio, pct_05, pct_10 = 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        'variant': variant,
        'accepted_trades': n_acc,
        'win_rate': round(win_rate, 2),
        'profit_factor': pf,
        'gross_win': round(gross_win, 2),
        'gross_loss': round(gross_loss, 2),
        'total_costs': round(tot_costs, 2),
        'net_pnl': round(net_pnl, 2),
        'expectancy': expectancy,
        'avg_mae': round(float(avg_mae), 2),
        'avg_mfe': round(float(avg_mfe), 2),
        'mfe_mae_ratio': mfe_mae_ratio,
        'pct_hit_05_mfe': round(pct_05, 1),
        'pct_hit_10_mfe': round(pct_10, 1)
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


def run_v11_ablation():
    print("\n" + "=" * 115)
    print("V11 BREAKOUT QUALITY, RETEST & MAE/MFE MICROSTRUCTURE ABLATION BENCHMARK")
    print("Evaluating Breakout Mechanics & Post-Breakout Behavior Under Full Statutory Costs")
    print("=" * 115)

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

    print("\n[PRECOMPUTING] Calculating candle quality, expansion distance, and NIFTY relative strength...")
    core_prep = {s: engine.prepare_dataset(df, nifty_df=nifty_df) for s, df in core_stock_data.items()}
    unseen_prep = {s: engine.prepare_dataset(df, nifty_df=nifty_df) for s, df in unseen_stock_data.items()}
    print("[PRECOMPUTING] Ready.\n")

    variants = [
        ('V11_A', 'Baseline Immediate ORB + NIFTY Alignment'),
        ('V11_B', '+ Breakout Candle Quality Filter (Strong Body/Wick)'),
        ('V11_C', '+ Controlled Breakout Distance (5% to 40%)'),
        ('V11_D', '+ Opening Range Width Filter (0.35% to 1.80%)'),
        ('V11_E', '+ Relative Strength vs NIFTY (RS >= 0.25%)'),
        ('V11_F', 'Breakout + Clean Retest Confirmation'),
        ('V11_G', 'Master Integrated Quality + Retest Engine')
    ]

    # --- Cohort 1: Core Universe ---
    print("#" * 115)
    print("COHORT 1: CORE IN-UNIVERSE DATASET (June-July 2026)")
    print("#" * 115)
    core_results = []
    for v_code, v_desc in variants:
        res = simulate_v11_variant(
            variant=v_code,
            prepared_stock_data=core_prep,
            raw_stock_data=core_stock_data
        )
        res['description'] = v_desc
        core_results.append(res)
        print(f"  [{v_code:<8}] Trades: {res['accepted_trades']:<4} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['expectancy']:>6.2f} | MFE/MAE: {res['mfe_mae_ratio']}")

    # --- Cohort 2: Unseen Universe ---
    print("\n" + "#" * 115)
    print("COHORT 2: UNSEEN OUT-OF-SAMPLE STOCKS (AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC)")
    print("#" * 115)
    unseen_results = []
    for v_code, v_desc in variants:
        res = simulate_v11_variant(
            variant=v_code,
            prepared_stock_data=unseen_prep,
            raw_stock_data=unseen_stock_data
        )
        res['description'] = v_desc
        unseen_results.append(res)
        print(f"  [{v_code:<8}] Trades: {res['accepted_trades']:<4} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['expectancy']:>6.2f} | MFE/MAE: {res['mfe_mae_ratio']}")

    # Summary Tables
    print("\n" + "=" * 115)
    print("FINAL SUMMARY: CORE UNIVERSE")
    print(f"{'Variant':<8} | {'Strategy Architecture':<52} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<12} | {'MFE/MAE':<8}")
    print("-" * 115)
    for r in core_results:
        print(f"{r['variant']:<8} | {r['description']:<52} | {r['accepted_trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs {r['net_pnl']:>9.2f} | {r['mfe_mae_ratio']:>7.2f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("FINAL SUMMARY: UNSEEN UNIVERSE (TRUE GENERALIZATION)")
    print(f"{'Variant':<8} | {'Strategy Architecture':<52} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<12} | {'MFE/MAE':<8}")
    print("-" * 115)
    for r in unseen_results:
        print(f"{r['variant']:<8} | {r['description']:<52} | {r['accepted_trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs {r['net_pnl']:>9.2f} | {r['mfe_mae_ratio']:>7.2f}")
    print("=" * 115)

    # MAE / MFE Microstructure Table
    print("\n" + "#" * 115)
    print("MAE / MFE MICROSTRUCTURE DIAGNOSTIC TABLE (Excursion Dynamics)")
    print("#" * 115)
    print(f"{'Variant':<8} | {'Avg MAE %':<10} | {'Avg MFE %':<10} | {'MFE/MAE':<10} | {'% Reaching +0.5% MFE':<22} | {'% Reaching +1.0% MFE':<22}")
    print("-" * 115)
    for r in core_results:
        print(f"{r['variant']:<8} | {r['avg_mae']:>8.2f}% | {r['avg_mfe']:>8.2f}% | {r['mfe_mae_ratio']:>8.2f} | {r['pct_hit_05_mfe']:>20.1f}% | {r['pct_hit_10_mfe']:>20.1f}%")

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v11_quality_ablation_report.json")
    out_csv = os.path.join(config.RESULTS_DIR, "v11_quality_ablation_summary.csv")

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
    run_v11_ablation()
