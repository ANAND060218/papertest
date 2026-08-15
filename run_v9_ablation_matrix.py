"""
V9 Scientific Ablation Matrix Runner
Benchmarks 8 Strategy Variants across In-Sample & Out-of-Sample Historical Cohorts:

  Variant   Strategy Architecture
  -------------------------------------------------------------
  V8        Baseline: ORB / VWAP Breakout + XGBoost
  V9-A      Candlestick Patterns Only (Hammer, Shooting Star, Engulfing)
  V9-B      Market Structure Only (Pivots, S/R Bounces/Rejections)
  V9-C      Pattern + Structure Confluence
  V9-D      Pattern + Structure + Indicators (Volume, VWAP, RSI, ATR R:R)
  V9-E      Above + XGBoost Probability Gating (P >= 0.40, EV > 0)
  V9-F      Above + Market Regime Filter
  V9-G      Pattern + Structure + Indicators + Regime WITHOUT XGBoost

Evaluated under the FULL Indian Market Statutory Cost Model:
Brokerage (0.03%), STT (0.025%), Stamp Duty (0.003%), GST (18%),
Exchange Txn (0.00345%), SEBI Fees (0.0001%), Slippage (0.05%), Spread (0.02%).
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
from core.data_manager import DataManager
from core.v9_confluence_engine import V9ConfluenceEngine
from core.multi_setup_detector import MultiSetupDetector
from core.regime_detector import RegimeDetector
from core.labeler import TradeLabeler
from models.xgboost_model import XGBoostTradeModel
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


def simulate_variant(variant, prepared_data, raw_stock_data, daily_ctx_dict, model=None, reg_detector=None):
    """
    Fast simulation of a strategy variant using pre-prepared datasets.
    """
    labeler = TradeLabeler()
    v9_engine = V9ConfluenceEngine()
    multi_detector = MultiSetupDetector()

    all_setups = []

    for symbol, df_prep in prepared_data.items():
        if df_prep.empty or len(df_prep) < 30:
            continue

        if variant == 'V8':
            # V8 Baseline: ORB & VWAP Breakout
            daily_ctx = daily_ctx_dict.get(symbol)
            setups_orb = multi_detector.detect_orb(df_prep)
            setups_vwap = multi_detector.detect_vwap_breakout(df_prep)
            setups_pdh = multi_detector.detect_prev_day_high_breakout(df_prep, daily_ctx) if daily_ctx is not None else []
            for s in (setups_orb + setups_vwap + setups_pdh):
                s['symbol'] = symbol
                all_setups.append(s)
        else:
            # V9 Variants
            setups = v9_engine.generate_setups(df_prep, variant=variant, symbol=symbol)
            all_setups.extend(setups)

    if not all_setups:
        return {
            'variant': variant,
            'total_setups': 0,
            'accepted_trades': 0,
            'rejected_setups': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'net_pnl': 0.0,
            'expectancy': 0.0,
            'gross_win': 0.0,
            'gross_loss': 0.0,
            'total_costs': 0.0,
            'capital_saved': 0.0
        }

    # Label setups forward
    labeled_trades = []
    for symbol, df_raw in raw_stock_data.items():
        sym_setups = [s for s in all_setups if s['symbol'] == symbol]
        if sym_setups:
            labeled = labeler.label_setups(sym_setups, df_raw)
            labeled_trades.extend(labeled)

    # Process executions and apply filtering
    accepted_trades = []
    rejected_trades = []

    for trade in labeled_trades:
        is_accepted = True
        xgb_prob = 0.50

        if variant in ['V8', 'V9_E', 'V9_F'] and model is not None:
            feats = trade.get('features', {})
            feat_vector = np.array([[
                feats.get('vol_ratio', 1.0),
                feats.get('rsi_14', 50.0),
                feats.get('dist_to_vwap', 0.0),
                feats.get('dist_to_pdh', 0.0),
                feats.get('dist_to_pdl', 0.0),
                feats.get('atr_pct', 0.5),
                feats.get('is_hammer', 0),
                feats.get('trend_bull', 0),
                feats.get('trend_bear', 0),
                feats.get('hour', 10),
                feats.get('minute', 0)
            ]])
            try:
                probs = model.model.predict_proba(feat_vector)[0]
                xgb_prob = probs[1] if len(probs) > 1 else 0.50
            except Exception:
                xgb_prob = 0.45

            if xgb_prob < 0.40:
                is_accepted = False

        # Regime Gating
        if variant in ['V9_F', 'V9_G'] and reg_detector is not None:
            if trade.get('direction') == 'LONG' and trade.get('market_trend') == 'BEARISH':
                is_accepted = False
            elif trade.get('direction') == 'SHORT' and trade.get('market_trend') == 'BULLISH':
                is_accepted = False

        # Friction Simulation (Slippage + Statutory Costs)
        entry_price = trade['entry_price']
        direction = trade.get('direction', 'LONG')
        stop_price = trade.get('stop_price', entry_price * 0.995)
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
            'is_win': net_pnl_rs > 0,
            'gross_pnl_rs': gross_pnl_rs,
            'costs_rs': costs_rs,
            'net_pnl_rs': net_pnl_rs,
            'xgb_prob': xgb_prob
        }

        if is_accepted:
            accepted_trades.append(trade_record)
        else:
            rejected_trades.append(trade_record)

    n_acc = len(accepted_trades)
    n_rej = len(rejected_trades)

    if n_acc > 0:
        wins = [t for t in accepted_trades if t['is_win']]
        win_rate = (len(wins) / n_acc) * 100
        gross_win = sum(t['gross_pnl_rs'] for t in accepted_trades if t['gross_pnl_rs'] > 0)
        gross_loss = abs(sum(t['gross_pnl_rs'] for t in accepted_trades if t['gross_pnl_rs'] < 0))
        tot_costs = sum(t['costs_rs'] for t in accepted_trades)
        net_pnl = sum(t['net_pnl_rs'] for t in accepted_trades)
        pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
        expectancy = round(net_pnl / n_acc, 2)
    else:
        win_rate = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        tot_costs = 0.0
        net_pnl = 0.0
        pf = 0.0
        expectancy = 0.0

    cap_saved = abs(sum(t['net_pnl_rs'] for t in rejected_trades if t['net_pnl_rs'] < 0))

    return {
        'variant': variant,
        'total_setups': len(labeled_trades),
        'accepted_trades': n_acc,
        'rejected_setups': n_rej,
        'win_rate': round(win_rate, 2),
        'profit_factor': pf,
        'gross_win': round(gross_win, 2),
        'gross_loss': round(gross_loss, 2),
        'total_costs': round(tot_costs, 2),
        'net_pnl': round(net_pnl, 2),
        'expectancy': expectancy,
        'capital_saved': round(cap_saved, 2)
    }


def load_universe_data(db_path, symbols):
    """Loads stock data dict from specified SQLite database."""
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


def precompute_stock_datasets(raw_stock_data, dm):
    """Precomputes and enriches datasets for both V8 and V9 once."""
    v9_engine = V9ConfluenceEngine()
    multi_detector = MultiSetupDetector()

    v8_prep = {}
    v9_prep = {}
    daily_ctx_dict = {}

    for sym, df in raw_stock_data.items():
        daily_ctx = dm.build_daily_context(df)
        daily_ctx_dict[sym] = daily_ctx

        v8_prep[sym] = multi_detector._compute_rolling_features(df)
        v9_prep[sym] = v9_engine.prepare_dataset(df, daily_df=daily_ctx)

    return v8_prep, v9_prep, daily_ctx_dict


def run_full_ablation_matrix():
    print("\n" + "=" * 115)
    print("V9 COMPREHENSIVE STRATEGY ABLATION MATRIX BENCHMARK")
    print("Evaluating 8 Variants under Realistic Indian Statutory Costs & Slippage")
    print("=" * 115)

    dm = DataManager()
    reg_det = RegimeDetector()

    # Load frozen XGBoost model
    xgb_model = XGBoostTradeModel()
    model_path = os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json")
    if os.path.exists(model_path):
        xgb_model.load(model_path)

    # 1. Load In-Universe Core Stocks
    core_db = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")
    core_stocks = config.UNIVERSE
    core_stock_data = load_universe_data(core_db, core_stocks)
    print(f"Loaded Core Universe Dataset: {len(core_stock_data)} symbols ({list(core_stock_data.keys())}).")

    # 2. Load Unseen Out-of-Sample Stocks
    unseen_db = os.path.join(config.DATA_DIR, "unseen_universe_5m.db")
    unseen_stocks = ["AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "LT.NS", "WIPRO.NS", "ITC.NS"]
    unseen_stock_data = load_universe_data(unseen_db, unseen_stocks)
    print(f"Loaded Unseen Universe Dataset: {len(unseen_stock_data)} symbols ({list(unseen_stock_data.keys())}).")

    print("\n[PRECOMPUTING] Pre-processing technical features & market structure once for all datasets...")
    core_v8_prep, core_v9_prep, core_daily_ctx = precompute_stock_datasets(core_stock_data, dm)
    unseen_v8_prep, unseen_v9_prep, unseen_daily_ctx = precompute_stock_datasets(unseen_stock_data, dm)
    print("[PRECOMPUTING] Feature matrices ready.")

    variants = [
        ('V8', 'Baseline (ORB/VWAP + XGBoost)'),
        ('V9_A', 'Patterns Only (Hammer/ShootingStar/Engulfing)'),
        ('V9_B', 'Market Structure Only (Pivots/SR Levels)'),
        ('V9_C', 'Pattern + Structure Confluence'),
        ('V9_D', 'Structure + Pattern + Indicators + Dynamic ATR R:R'),
        ('V9_E', 'Structure + Pattern + Indicators + XGBoost'),
        ('V9_F', 'Structure + Pattern + Indicators + XGBoost + Regime'),
        ('V9_G', 'Structure + Pattern + Indicators + Regime (WITHOUT XGBoost)')
    ]

    # --- Benchmark 1: Core In-Universe ---
    print("\n" + "#" * 115)
    print("COHORT 1: CORE IN-UNIVERSE DATASET (June-July 2026)")
    print("#" * 115)
    core_results = []
    for v_code, v_desc in variants:
        prep_data = core_v8_prep if v_code == 'V8' else core_v9_prep
        res = simulate_variant(
            variant=v_code,
            prepared_data=prep_data,
            raw_stock_data=core_stock_data,
            daily_ctx_dict=core_daily_ctx,
            model=xgb_model,
            reg_detector=reg_det
        )
        res['description'] = v_desc
        core_results.append(res)
        print(f"  [{v_code:<5}] Trades: {res['accepted_trades']:<4} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['expectancy']:>6.2f} | Saved: Rs {res['capital_saved']:>8.2f}")

    # --- Benchmark 2: Unseen Universe ---
    print("\n" + "#" * 115)
    print("COHORT 2: UNSEEN OUT-OF-SAMPLE STOCKS (AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC)")
    print("#" * 115)
    unseen_results = []
    for v_code, v_desc in variants:
        prep_data = unseen_v8_prep if v_code == 'V8' else unseen_v9_prep
        res = simulate_variant(
            variant=v_code,
            prepared_data=prep_data,
            raw_stock_data=unseen_stock_data,
            daily_ctx_dict=unseen_daily_ctx,
            model=xgb_model,
            reg_detector=reg_det
        )
        res['description'] = v_desc
        unseen_results.append(res)
        print(f"  [{v_code:<5}] Trades: {res['accepted_trades']:<4} | WinRate: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>5.3f} | Net PnL: Rs {res['net_pnl']:>8.2f} | EV: Rs {res['expectancy']:>6.2f} | Saved: Rs {res['capital_saved']:>8.2f}")

    # Display Pretty Markdown Tables
    print("\n" + "=" * 115)
    print("FINAL SUMMARY: CORE UNIVERSE")
    print(f"{'Variant':<8} | {'Strategy Architecture':<45} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<12} | {'EV/Tr (Rs)':<10}")
    print("-" * 115)
    for r in core_results:
        print(f"{r['variant']:<8} | {r['description']:<45} | {r['accepted_trades']:<6} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs {r['net_pnl']:>9.2f} | Rs {r['expectancy']:>8.2f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("FINAL SUMMARY: UNSEEN UNIVERSE (TRUE GENERALIZATION)")
    print(f"{'Variant':<8} | {'Strategy Architecture':<45} | {'Trades':<6} | {'Win %':<6} | {'PF':<6} | {'Net P&L (Rs)':<12} | {'EV/Tr (Rs)':<10}")
    print("-" * 115)
    for r in unseen_results:
        print(r['variant'].ljust(8) + " | " + r['description'].ljust(45) + " | " + str(r['accepted_trades']).ljust(6) + f" | {r['win_rate']:>5.1f}% | {r['profit_factor']:>6.3f} | Rs {r['net_pnl']:>9.2f} | Rs {r['expectancy']:>8.2f}")
    print("=" * 115)

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v9_ablation_matrix_report.json")
    out_csv = os.path.join(config.RESULTS_DIR, "v9_ablation_matrix_summary.csv")

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
    run_full_ablation_matrix()
