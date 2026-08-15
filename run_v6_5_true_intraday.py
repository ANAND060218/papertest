"""
V6.5 -- True 5-Minute Intraday Research & Validation Pipeline
Executes rigorous intraday quantitative research on verified 5-minute candles:
  1. Multi-stock 5-minute feature engineering (VWAP, EMA, ATR, RSI, Session Phase)
  2. True Intraday Setup Detection (ORB 15M, VWAP Breakout, Prev-Day High, Momentum)
  3. Conservative Intraday Labeling with 15:15 IST compulsory close
  4. Real-world Indian cost-aware backtesting (Brokerage, STT, GST, Slippage)
  5. Strict 3-Way Split Walk-Forward Validation (Train -> Validation for threshold selection -> Test for uncorrupted out-of-sample evaluation)
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from core.feature_engine import FeatureEngine
from core.regime_detector import RegimeDetector
from core.multi_setup_detector import MultiSetupDetector
from core.labeler import TradeLabeler
from backtest.backtester import Backtester, CostModel
from models.xgboost_model import XGBoostTradeModel, ThresholdAnalyzer
from risk.expected_value import ExpectedValueCalculator


INTRADAY_UNIVERSE_DB = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")


class TrueIntradayFeatureEngine:
    """
    Features engineered specifically for 5-minute intraday bars (Zero look-ahead).
    """

    def build_intraday_features(self, df):
        df = df.copy()

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        df['trade_date'] = df['Date'].dt.date
        df['time'] = df['Date'].dt.time

        # 1. Moving Averages
        df['feat_ema_9'] = close.ewm(span=9, adjust=False).mean()
        df['feat_ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['feat_ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['feat_price_vs_ema20'] = (close - df['feat_ema_20']) / df['feat_ema_20']

        # 2. Cumulative VWAP per session
        df['feat_vwap'] = (
            df.groupby('trade_date')
            .apply(lambda g: (g['Close'] * g['Volume']).cumsum() / g['Volume'].cumsum().replace(0, np.nan), include_groups=False)
            .reset_index(level=0, drop=True)
        )
        df['feat_dist_from_vwap'] = (close - df['feat_vwap']) / df['feat_vwap']

        # 3. ATR 14-bar (normalized by price)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['feat_atr_14'] = tr.rolling(14, min_periods=5).mean()
        df['feat_atr_pct'] = df['feat_atr_14'] / close

        # 4. RSI 14-bar
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        df['feat_rsi_14'] = 100 - (100 / (1 + rs))

        # 5. Volume Features
        df['feat_vol_avg_20'] = df['Volume'].rolling(20, min_periods=5).mean()
        df['feat_volume_ratio'] = volume / df['feat_vol_avg_20'].replace(0, np.nan)

        # 6. Intraday Momentum & Range
        df['feat_return_1bar'] = close.pct_change(1)
        df['feat_return_3bar'] = close.pct_change(3)
        df['feat_bar_range'] = (high - low) / close

        # 7. Time / Session Phase
        # 0 = 09:15, 1 = 15:30
        minutes_since_open = df['Date'].dt.hour * 60 + df['Date'].dt.minute - (9 * 60 + 15)
        df['feat_session_progress'] = np.clip(minutes_since_open / 375.0, 0.0, 1.0)

        # 8. Distance from day's high/low
        df['day_high_so_far'] = df.groupby('trade_date')['High'].cummax()
        df['day_low_so_far'] = df.groupby('trade_date')['Low'].cummin()
        df['feat_dist_day_high'] = (close - df['day_high_so_far']) / df['feat_atr_14'].replace(0, np.nan)
        df['feat_dist_day_low'] = (close - df['day_low_so_far']) / df['feat_atr_14'].replace(0, np.nan)

        return df

    def get_feature_columns(self):
        return [
            'feat_price_vs_ema20', 'feat_dist_from_vwap', 'feat_atr_pct',
            'feat_rsi_14', 'feat_volume_ratio', 'feat_return_1bar',
            'feat_return_3bar', 'feat_bar_range', 'feat_session_progress',
            'feat_dist_day_high', 'feat_dist_day_low'
        ]


def run_true_intraday_research():
    dm = DataManager()
    fe = TrueIntradayFeatureEngine()
    labeler = TradeLabeler()
    regime_det = RegimeDetector()
    cost_model = CostModel()

    symbols = config.UNIVERSE
    conn = sqlite3.connect(INTRADAY_UNIVERSE_DB)

    print("=" * 100)
    print("V6.5: TRUE 5-MINUTE INTRADAY RESEARCH & STATISTICAL VALIDATION")
    print(f"Target Symbols ({len(symbols)}): {symbols}")
    print("=" * 100)

    per_stock_intraday_results = []
    all_intraday_trades = []

    for sym in symbols:
        table_name = f"bars_5m_{sym.replace('.NS', '')}"
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY Date ASC", conn)
        except Exception:
            print(f"  [SKIP] Table {table_name} not found.")
            continue

        if df.empty or len(df) < 500:
            print(f"  [SKIP] Insufficient 5m data for {sym}")
            continue

        df['Date'] = pd.to_datetime(df['Date'])
        daily_ctx = dm.build_daily_context(df)

        # 1. Compute 5-minute features
        df_feat = fe.build_intraday_features(df)

        # 2. Detect True Intraday Setups (ORB 15m, VWAP Breakout, Prev Day High)
        msd = MultiSetupDetector()
        # Compute baseline rolling features for MSD
        df_msd = msd._compute_rolling_features(df)

        setups_orb = msd.detect_orb(df_msd, orb_minutes=15, vol_min=1.1)
        setups_vwap = msd.detect_vwap_breakout(df_msd, vol_min=1.1)
        setups_pdh = msd.detect_prev_day_high_breakout(df_msd, daily_ctx, vol_min=1.1)

        combined_setups = setups_orb + setups_vwap + setups_pdh
        if not combined_setups:
            continue

        # 3. Label with conservative worst-case assumptions & 15:15 square-off
        labeled = labeler.label_setups(combined_setups, df)

        # Attach symbol, features, and metadata
        feat_cols = fe.get_feature_columns()
        for t in labeled:
            t['symbol'] = sym
            idx = t['bar_index']
            if idx < len(df_feat):
                for f_col in feat_cols:
                    t[f_col] = df_feat.iloc[idx].get(f_col, np.nan)
            else:
                for f_col in feat_cols:
                    t[f_col] = np.nan

        all_intraday_trades.extend(labeled)

        # 4. Backtest per stock
        bt = Backtester(initial_capital=config.INITIAL_CAPITAL)
        res_df, perf = bt.run(labeled)

        if perf:
            per_stock_intraday_results.append({
                'Symbol': sym,
                '5m_Bars': len(df),
                'Days': len(daily_ctx),
                'Trades': perf['total_trades'],
                'TARGET_Wins': perf['targets'],
                'STOP_Losses': perf['stops'],
                'TIMEOUT_Exits': perf['timeouts'],
                'WinRate%': perf['win_rate'],
                'ProfitFactor': perf['profit_factor'],
                'Expectancy': perf['expectancy'],
                'NetPnL': perf['total_pnl_net'],
                'MaxDD%': perf['max_drawdown_pct'],
                'Sharpe': perf['sharpe_ratio'],
            })

    conn.close()

    ps_df = pd.DataFrame(per_stock_intraday_results)
    print("\n" + "=" * 110)
    print("V6.5 PER-STOCK TRUE 5-MINUTE INTRADAY PERFORMANCE (AFTER ALL STATUTORY COSTS & SLIPPAGE)")
    print("=" * 110)
    if not ps_df.empty:
        ps_df = ps_df.sort_values('ProfitFactor', ascending=False)
        print(ps_df.to_string(index=False))

    if not all_intraday_trades:
        print("\n[V6.5] No intraday trades collected.")
        return

    pooled_5m_df = pd.DataFrame(all_intraday_trades)
    pooled_5m_df['timestamp'] = pd.to_datetime(pooled_5m_df['timestamp'])
    pooled_5m_df = pooled_5m_df.sort_values('timestamp').reset_index(drop=True)
    pooled_5m_df['label'] = (pooled_5m_df['result'] == 'TARGET').astype(int)

    print("\n" + "=" * 100)
    print(f"POOLED TRUE 5-MINUTE INTRADAY DATASET: {len(pooled_5m_df)} TOTAL SETUPS")
    print("=" * 100)
    print(f"Setup Types Breakdown: {pooled_5m_df['setup_name'].value_counts().to_dict()}")
    print(f"Outcome Distribution: {pooled_5m_df['result'].value_counts().to_dict()}")
    print(f"Average Trade Net P&L: {pooled_5m_df['pnl_pct'].mean():+.4f}%")

    # ==========================================
    # RIGOROUS 3-WAY SPLIT XGBOOST VALIDATION (TRAIN / VAL / TEST)
    # ==========================================
    print("\n" + "=" * 100)
    print("V6.5 UNBIASED XGBOOST VALIDATION (TRAIN 60% -> VAL 20% -> HELD-OUT TEST 20%)")
    print("=" * 100)

    feat_cols = fe.get_feature_columns()
    available_feats = [c for c in feat_cols if c in pooled_5m_df.columns]

    n_total = len(pooled_5m_df)
    n_train = int(n_total * 0.60)
    n_val = int(n_total * 0.80)

    train_split = pooled_5m_df.iloc[:n_train]
    val_split = pooled_5m_df.iloc[n_train:n_val]
    test_split = pooled_5m_df.iloc[n_val:]

    print(f"  Train partition:      {len(train_split)} trades ({train_split['timestamp'].min().strftime('%Y-%m-%d')} to {train_split['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Validation partition: {len(val_split)} trades ({val_split['timestamp'].min().strftime('%Y-%m-%d')} to {val_split['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Held-out Test:        {len(test_split)} trades ({test_split['timestamp'].min().strftime('%Y-%m-%d')} to {test_split['timestamp'].max().strftime('%Y-%m-%d')})")

    X_train = train_split[available_feats].fillna(0)
    y_train = train_split['label']
    X_val = val_split[available_feats].fillna(0)
    y_val = val_split['label']
    X_test = test_split[available_feats].fillna(0)
    y_test = test_split['label']

    # 1. Train model on Train partition
    xgb_intraday = XGBoostTradeModel()
    xgb_intraday.train(X_train, y_train, X_val, y_val)

    # 2. Select optimal threshold on VALIDATION partition ONLY
    val_proba = xgb_intraday.predict_proba(X_val)
    analyzer = ThresholdAnalyzer()
    val_thresh_table = analyzer.analyze_thresholds(val_proba, y_val.values, val_split['pnl_pct'].values)

    print("\n--- Validation Partition Threshold Scan (Used to PICK Threshold) ---")
    print(val_thresh_table.to_string(index=False))

    # Find best threshold on validation (highest total PnL with at least 5 trades)
    eligible_val = val_thresh_table[(val_thresh_table['trades'] >= 5) & (val_thresh_table['threshold'] != 'BASELINE')]
    if not eligible_val.empty:
        best_val_row = eligible_val.sort_values('total_pnl_pct', ascending=False).iloc[0]
        chosen_thresh_str = best_val_row['threshold']
        chosen_thresh = float(chosen_thresh_str.replace('>=', ''))
        print(f"\n>>> OPTIMAL THRESHOLD CHOSEN ON VALIDATION: P(win) >= {chosen_thresh:.2f}")
    else:
        chosen_thresh = 0.50
        print(f"\n>>> DEFAULT THRESHOLD CHOSEN: P(win) >= {chosen_thresh:.2f}")

    # 3. Evaluate LOCKED threshold ONCE on Held-out TEST set
    test_proba = xgb_intraday.predict_proba(X_test)
    test_thresh_table = analyzer.analyze_thresholds(test_proba, y_test.values, test_split['pnl_pct'].values)

    print("\n--- UNTOUCHED HELD-OUT TEST SET EVALUATION ---")
    print(test_thresh_table.to_string(index=False))

    # Compare Baseline vs Locked Threshold on Test
    baseline_test_pnl = test_split['pnl_pct'].sum()
    baseline_test_win = (test_split['label'] == 1).mean() * 100

    test_mask = test_proba >= chosen_thresh
    filtered_test_trades = test_split[test_mask]
    filtered_test_pnl = filtered_test_trades['pnl_pct'].sum() if len(filtered_test_trades) > 0 else 0.0
    filtered_test_win = (filtered_test_trades['label'] == 1).mean() * 100 if len(filtered_test_trades) > 0 else 0.0

    print("\n" + "=" * 90)
    print("V6.5 STATISTICAL INTRADAY VERDICT (OUT-OF-SAMPLE TEST)")
    print("=" * 90)
    print(f"  Baseline (Take All Setups):  Trades: {len(test_split)} | Win Rate: {baseline_test_win:.1f}% | Total P&L: {baseline_test_pnl:+.3f}%")
    print(f"  ML-Filtered (P >= {chosen_thresh:.2f}):    Trades: {len(filtered_test_trades)} | Win Rate: {filtered_test_win:.1f}% | Total P&L: {filtered_test_pnl:+.3f}%")
    print("-" * 90)

    if filtered_test_pnl > baseline_test_pnl:
        print(">>> RESULT: XGBoost ML Filter produced TRUE Out-of-Sample improvement on 5m Intraday Data.")
    else:
        print(">>> RESULT: ML did NOT beat baseline out-of-sample on this 5m dataset. Rule-based execution remains primary.")

    # Save artifacts
    ps_df.to_csv(os.path.join(config.RESULTS_DIR, "v6_5_per_stock_intraday.csv"), index=False)
    val_thresh_table.to_csv(os.path.join(config.RESULTS_DIR, "v6_5_val_thresholds.csv"), index=False)
    test_thresh_table.to_csv(os.path.join(config.RESULTS_DIR, "v6_5_test_thresholds.csv"), index=False)
    xgb_intraday.save(os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json"))
    print(f"\n[V6.5] Results saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    run_true_intraday_research()
