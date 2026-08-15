"""
Master V4/V5 Pipeline Runner
Runs V4 (XGBoost + Walk-Forward) and V5 (EV) on the best setup from V3.

Pipeline:
  1. Load 10-year daily data
  2. Compute features
  3. Detect INSIDE_BAR + VOLUME_BREAKOUT setups (best from V3)
  4. Label outcomes
  5. Train XGBoost with walk-forward validation
  6. Compare baseline vs ML-filtered vs EV-filtered
  7. Decision: Does ML add value?
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
from models.xgboost_model import XGBoostTradeModel, ThresholdAnalyzer
from models.walk_forward import WalkForwardValidator
from risk.expected_value import ExpectedValueCalculator
from backtest.backtester import Backtester


def run_v4_v5_pipeline(setup_name='INSIDE_BAR'):
    """Run V4/V5 on a specific setup."""

    symbol = config.PRIMARY_SYMBOL
    print("=" * 100)
    print(f"V4/V5 PIPELINE: {setup_name} on {symbol} (10-year daily)")
    print("=" * 100)

    # ==========================================
    # Step 1: Load + Compute Features
    # ==========================================
    dm = DataManager()
    daily_df = dm.load_daily(symbol)
    if daily_df is None:
        print("[FATAL] No daily data")
        return

    fe = FeatureEngine()
    df = fe.build_features(daily_df)

    # Also compute setup-specific features
    # (ATR, prev_high/low, inside_bar flag are already in feature_engine + daily setup detector)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=10).mean()

    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['is_inside_bar'] = ((df['High'] < df['prev_high']) & (df['Low'] > df['prev_low']))

    df['vol_avg_20'] = df['Volume'].rolling(20, min_periods=10).mean()
    df['volume_ratio'] = df['Volume'] / df['vol_avg_20'].replace(0, np.nan)

    # ==========================================
    # Step 2: Detect Setups + Label
    # ==========================================
    from run_daily_backtest import DailySetupDetector, DailyLabeler

    dsd = DailySetupDetector()
    df_with_feats = dsd.compute_features(daily_df)
    # Merge our ML features into the setup df
    for col in fe.get_feature_columns():
        if col in df.columns:
            df_with_feats[col] = df[col].values

    if setup_name == 'INSIDE_BAR':
        setups = dsd.detect_inside_bar_breakout(df_with_feats)
    elif setup_name == 'VOLUME_BREAKOUT':
        setups = dsd.detect_volume_breakout(df_with_feats, vol_spike=2.0)
    elif setup_name == '20D_BREAKOUT':
        setups = dsd.detect_nday_breakout(df_with_feats, n=20)
    else:
        print(f"Unknown setup: {setup_name}")
        return

    if not setups:
        print(f"[ERROR] No {setup_name} setups detected")
        return

    labeler = DailyLabeler()
    labeled = labeler.label_setups(setups, df_with_feats)

    print(f"\nTotal {setup_name} setups: {len(labeled)}")

    # ==========================================
    # Step 3: Build ML Dataset
    # ==========================================
    feature_cols = fe.get_feature_columns()

    # Build trades DataFrame with features
    trades_data = []
    for trade in labeled:
        idx = trade['bar_index']
        row = {}
        row['timestamp'] = trade['timestamp']
        row['entry_price'] = trade['entry_price']
        row['exit_price'] = trade['exit_price']
        row['result'] = trade['result']
        row['pnl_pct'] = trade['pnl_pct']
        row['bars_held'] = trade['bars_held']

        # Label: 1 = TARGET (win), 0 = STOP or TIMEOUT (loss)
        row['label'] = 1 if trade['result'] == 'TARGET' else 0

        # Extract features at entry bar
        for col in feature_cols:
            if col in df.columns:
                val = df.iloc[idx][col] if idx < len(df) else np.nan
                row[col] = val
            else:
                row[col] = np.nan

        trades_data.append(row)

    trades_df = pd.DataFrame(trades_data)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

    # Drop rows with too many NaN features
    available_feats = [c for c in feature_cols if c in trades_df.columns]
    trades_df = trades_df.dropna(subset=available_feats, thresh=len(available_feats) * 0.7)

    print(f"ML dataset: {len(trades_df)} trades with {len(available_feats)} features")
    print(f"Label distribution: {trades_df['label'].value_counts().to_dict()}")

    if len(trades_df) < 30:
        print("[ERROR] Not enough trades for ML. Need 30+.")
        return

    # ==========================================
    # Step 4: Walk-Forward Validation
    # ==========================================
    print(f"\n{'=' * 100}")
    print("V4: WALK-FORWARD VALIDATION")
    print(f"{'=' * 100}")

    wf = WalkForwardValidator()
    wf_results = wf.run_walk_forward(
        trades_df, available_feats, label_col='label',
        model_class=XGBoostTradeModel, date_col='timestamp'
    )
    wf_summary = wf.summarize(wf_results)

    # ==========================================
    # Step 5: Train final model on all data (for threshold analysis)
    # ==========================================
    print(f"\n{'=' * 100}")
    print("V4: THRESHOLD ANALYSIS (Full Dataset)")
    print(f"{'=' * 100}")

    # Use 80/20 chronological split for final threshold selection
    split_idx = int(len(trades_df) * 0.8)
    train_df = trades_df.iloc[:split_idx]
    test_df = trades_df.iloc[split_idx:]

    X_train = train_df[available_feats].fillna(0)
    y_train = train_df['label']
    X_test = test_df[available_feats].fillna(0)
    y_test = test_df['label']

    model = XGBoostTradeModel()
    model.train(X_train, y_train, X_test, y_test)

    # Feature importance
    fi = model.feature_importance()
    print(f"\nTop 10 Features:")
    print(fi.head(10).to_string(index=False))

    # Predict on test set
    test_proba = model.predict_proba(X_test)

    # Threshold analysis
    analyzer = ThresholdAnalyzer()
    threshold_results = analyzer.analyze_thresholds(
        test_proba, y_test.values, test_df['pnl_pct'].values
    )
    print(f"\nThreshold Comparison (Test Set):")
    print(threshold_results.to_string(index=False))

    # ==========================================
    # Step 6: V5 Expected Value Analysis
    # ==========================================
    print(f"\n{'=' * 100}")
    print("V5: EXPECTED VALUE ANALYSIS")
    print(f"{'=' * 100}")

    # Calculate avg win/loss from historical data
    wins = trades_df[trades_df['label'] == 1]['pnl_pct']
    losses = trades_df[trades_df['label'] == 0]['pnl_pct']
    avg_win = wins.mean() if len(wins) > 0 else 1.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.5

    ev_calc = ExpectedValueCalculator(
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss
    )

    breakeven_p = ev_calc.print_ev_analysis(
        test_proba,
        labels=y_test.values,
        pnl_pcts=test_df['pnl_pct'].values
    )

    # ==========================================
    # Step 7: Save model
    # ==========================================
    model_path = os.path.join(config.RESULTS_DIR, f"xgb_{setup_name.lower()}.json")
    model.save(model_path)

    # ==========================================
    # Final Summary
    # ==========================================
    print(f"\n{'=' * 100}")
    print("V4/V5 PIPELINE COMPLETE")
    print(f"{'=' * 100}")
    print(f"  Setup:           {setup_name}")
    print(f"  Total trades:    {len(trades_df)}")
    print(f"  Features used:   {len(available_feats)}")
    print(f"  Breakeven P:     {breakeven_p:.4f} ({breakeven_p*100:.1f}%)")
    print(f"  Model saved:     {model_path}")

    if wf_results:
        consistent = sum(1 for r in wf_results if r['improvement'] > 0)
        total_splits = len(wf_results)
        print(f"  Walk-forward:    {consistent}/{total_splits} splits improved by ML")

        if consistent > total_splits / 2:
            print(f"\n  >>> V4/V5 VERDICT: ML adds value. Proceed to V6 (multi-stock + regime)")
        else:
            print(f"\n  >>> V4/V5 VERDICT: ML does NOT consistently help.")
            print(f"      Consider staying rule-based for this setup.")


if __name__ == "__main__":
    # Check for xgboost
    try:
        import xgboost
    except ImportError:
        print("[INSTALLING] xgboost...")
        os.system("pip install xgboost scikit-learn")

    # Run on the two best setups from V3
    print("\n" + "#" * 100)
    print("# TESTING INSIDE_BAR BREAKOUT")
    print("#" * 100)
    run_v4_v5_pipeline('INSIDE_BAR')

    print("\n\n" + "#" * 100)
    print("# TESTING VOLUME_BREAKOUT")
    print("#" * 100)
    run_v4_v5_pipeline('VOLUME_BREAKOUT')
