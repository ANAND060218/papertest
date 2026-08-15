"""
V6 -- Multi-Stock Universe + Market Regime A/B Testing
Tests setups across 6 major liquid stocks:
  1. RELIANCE.NS
  2. TCS.NS
  3. INFY.NS
  4. HDFCBANK.NS
  5. ICICIBANK.NS
  6. SBIN.NS

Performs:
  1. Per-stock independent performance evaluation (Profit Factor, Win Rate, Expectancy)
  2. Pooled multi-stock dataset for XGBoost training (500+ trades for statistical power)
  3. Regime Detector A/B Testing: Does filtering by market regime improve performance?
"""
import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from core.feature_engine import FeatureEngine
from core.regime_detector import RegimeDetector
from run_daily_backtest import DailySetupDetector, DailyLabeler
from backtest.backtester import Backtester
from models.xgboost_model import XGBoostTradeModel, ThresholdAnalyzer
from models.walk_forward import WalkForwardValidator
from risk.expected_value import ExpectedValueCalculator


def run_v6_multi_stock_analysis():
    dm = DataManager()
    fe = FeatureEngine()
    dsd = DailySetupDetector()
    labeler = DailyLabeler()
    regime_det = RegimeDetector()

    symbols = config.UNIVERSE
    print("=" * 100)
    print(f"V6: MULTI-STOCK UNIVERSE EVALUATION ({len(symbols)} stocks)")
    print("=" * 100)

    per_stock_results = []
    all_labeled_trades = []

    for sym in symbols:
        daily_df = dm.load_daily(sym)
        if daily_df is None or len(daily_df) < 200:
            print(f"  [SKIP] Insufficient data for {sym}")
            continue

        # 1. Feature engineering & Regime tagging
        df_feat = fe.build_features(daily_df)
        df_reg = regime_det.classify_regimes(df_feat)
        df_setups_input = dsd.compute_features(df_reg)

        # Merge feature columns
        for col in fe.get_feature_columns():
            if col in df_reg.columns:
                df_setups_input[col] = df_reg[col].values
        df_setups_input['regime'] = df_reg['regime'].values

        # 2. Detect setups (INSIDE_BAR and VOLUME_BREAKOUT)
        setups_ib = dsd.detect_inside_bar_breakout(df_setups_input)
        setups_vb = dsd.detect_volume_breakout(df_setups_input, vol_spike=2.0)

        combined_setups = setups_ib + setups_vb
        if not combined_setups:
            continue

        # 3. Label outcomes
        labeled = labeler.label_setups(combined_setups, df_setups_input)

        # Attach symbol and regime
        for t in labeled:
            t['symbol'] = sym
            idx = t['bar_index']
            t['regime'] = df_setups_input.iloc[idx]['regime'] if idx < len(df_setups_input) else 'UNKNOWN'
            t['adx'] = df_setups_input.iloc[idx].get('feat_adx', 0)
            t['atr_pct'] = df_setups_input.iloc[idx].get('feat_atr_pct', 0)
            t['rsi'] = df_setups_input.iloc[idx].get('feat_rsi_14', 50)
            # Store all features
            for f_col in fe.get_feature_columns():
                t[f_col] = df_setups_input.iloc[idx].get(f_col, np.nan)

        all_labeled_trades.extend(labeled)

        # 4. Backtest per stock
        bt = Backtester(initial_capital=config.INITIAL_CAPITAL)
        res_df, perf = bt.run(labeled)

        if perf:
            per_stock_results.append({
                'Symbol': sym,
                'Trades': perf['total_trades'],
                'WinRate%': perf['win_rate'],
                'ProfitFactor': perf['profit_factor'],
                'Expectancy': perf['expectancy'],
                'NetPnL': perf['total_pnl_net'],
                'MaxDD%': perf['max_drawdown_pct'],
                'Sharpe': perf['sharpe_ratio'],
            })

    # Summary table per stock
    ps_df = pd.DataFrame(per_stock_results).sort_values('ProfitFactor', ascending=False)
    print("\n" + "=" * 100)
    print("PER-STOCK INDEPENDENT PERFORMANCE")
    print("=" * 100)
    print(ps_df.to_string(index=False))

    # Pooled Universe Trades Analysis
    pooled_df = pd.DataFrame(all_labeled_trades)
    pooled_df['timestamp'] = pd.to_datetime(pooled_df['timestamp'])
    pooled_df = pooled_df.sort_values('timestamp').reset_index(drop=True)
    pooled_df['label'] = (pooled_df['result'] == 'TARGET').astype(int)

    print("\n" + "=" * 100)
    print(f"POOLED MULTI-STOCK DATASET: {len(pooled_df)} TOTAL TRADES")
    print("=" * 100)
    print(f"Total Wins: {(pooled_df['label'] == 1).sum()} ({(pooled_df['label'] == 1).mean()*100:.1f}%)")
    print(f"Total Losses: {(pooled_df['label'] == 0).sum()} ({(pooled_df['label'] == 0).mean()*100:.1f}%)")
    print(f"Average Trade P&L: {pooled_df['pnl_pct'].mean():+.3f}%")

    # ==========================================
    # REGIME A/B TESTING
    # ==========================================
    print("\n" + "=" * 100)
    print("V6 REGIME A/B TEST: DOES REGIME FILTERING ADD VALUE?")
    print("=" * 100)

    reg_summary = regime_det.evaluate_regime_impact(pooled_df, regime_col='regime', pnl_col='pnl_pct')
    print("\nPerformance by Regime:")
    print(reg_summary.to_string(index=False))

    # Baseline (All regimes) vs Favorable Regimes Only
    baseline_bt = Backtester()
    _, base_perf = baseline_bt.run(all_labeled_trades)

    # Filter out unfavorable regimes (e.g. HIGH_VOL_CHOP or TRENDING_BEAR for long-only setups)
    favorable_trades = [t for t in all_labeled_trades if t['regime'] in ['TRENDING_BULL', 'WEAK_BULL', 'SIDEWAYS']]
    fav_bt = Backtester()
    _, fav_perf = fav_bt.run(favorable_trades)

    print("\n" + "-" * 70)
    print(f"BASELINE (No Regime Filter):")
    print(f"  Trades: {base_perf['total_trades']} | Win Rate: {base_perf['win_rate']:.1f}% | PF: {base_perf['profit_factor']:.3f} | Net P&L: Rs.{base_perf['total_pnl_net']:,.2f} | Sharpe: {base_perf['sharpe_ratio']:.2f}")

    print(f"\nWITH REGIME FILTER (Bull/Sideways Only):")
    print(f"  Trades: {fav_perf['total_trades']} | Win Rate: {fav_perf['win_rate']:.1f}% | PF: {fav_perf['profit_factor']:.3f} | Net P&L: Rs.{fav_perf['total_pnl_net']:,.2f} | Sharpe: {fav_perf['sharpe_ratio']:.2f}")
    print("-" * 70)

    if fav_perf['profit_factor'] > base_perf['profit_factor']:
        print(">>> REGIME FILTER VERDICT: POSITIVE IMPACT. Keep regime filter in live execution.")
    else:
        print(">>> REGIME FILTER VERDICT: NEUTRAL/NEGATIVE. Remove regime filter.")

    # ==========================================
    # POOLED MULTI-STOCK XGBOOST TRAINING
    # ==========================================
    print("\n" + "=" * 100)
    print("V6 POOLED MULTI-STOCK XGBOOST MODEL TRAINING")
    print("=" * 100)

    feat_cols = fe.get_feature_columns()
    available_feats = [c for c in feat_cols if c in pooled_df.columns]

    # Chronological Split (80% train, 20% test across the entire universe)
    split_idx = int(len(pooled_df) * 0.8)
    train_pool = pooled_df.iloc[:split_idx]
    test_pool = pooled_df.iloc[split_idx:]

    X_train = train_pool[available_feats].fillna(0)
    y_train = train_pool['label']
    X_test = test_pool[available_feats].fillna(0)
    y_test = test_pool['label']

    multi_xgb = XGBoostTradeModel()
    multi_xgb.train(X_train, y_train, X_test, y_test)

    test_proba = multi_xgb.predict_proba(X_test)
    analyzer = ThresholdAnalyzer()
    thresh_table = analyzer.analyze_thresholds(test_proba, y_test.values, test_pool['pnl_pct'].values)

    print("\nMulti-Stock Test Set Threshold Performance:")
    print(thresh_table.to_string(index=False))

    # Save Multi-Stock Model
    model_save_path = os.path.join(config.RESULTS_DIR, "xgb_universe_multistock.json")
    multi_xgb.save(model_save_path)

    # Save CSV Results
    ps_df.to_csv(os.path.join(config.RESULTS_DIR, "v6_per_stock_results.csv"), index=False)
    reg_summary.to_csv(os.path.join(config.RESULTS_DIR, "v6_regime_summary.csv"), index=False)
    thresh_table.to_csv(os.path.join(config.RESULTS_DIR, "v6_multistock_thresholds.csv"), index=False)
    print(f"\n[V6] All analysis artifacts saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    run_v6_multi_stock_analysis()
