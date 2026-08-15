"""
Master Backtest Runner
Runs the full V1 -> V2 -> V3 pipeline:
  V1: Load/validate 5-minute data
  V2: Detect setups + Label outcomes
  V3: Backtest with real costs + Decision Gate
"""
import sys
import os

# Ensure imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'backtest'))

import config
from core.data_manager import DataManager
from core.setup_detector import SetupDetector
from core.labeler import TradeLabeler
from backtest.backtester import Backtester


def main():
    symbol = config.PRIMARY_SYMBOL
    print("=" * 90)
    print(f"RUNNING FULL PIPELINE: V1 -> V2 -> V3 for {symbol}")
    print("=" * 90)

    # ==========================================
    # V1: Load Data
    # ==========================================
    dm = DataManager()
    intraday_df = dm.load_intraday(symbol)

    if intraday_df is None or intraday_df.empty:
        print("[V1] No saved intraday data found. Fetching from yfinance...")
        raw_df = dm.fetch_intraday_from_yfinance(symbol, days_back=config.YFINANCE_MAX_5M_DAYS)
        if raw_df is None:
            print("[FATAL] Could not fetch data. Exiting.")
            return
        intraday_df, report = dm.validate_intraday(raw_df, symbol=symbol)
        dm.save_intraday_to_db(intraday_df)
        dm.save_intraday_to_parquet(intraday_df)
    else:
        print(f"[V1] Loaded {len(intraday_df)} rows of saved intraday data.")

    # Build daily context
    daily_ctx = dm.build_daily_context(intraday_df)
    print(f"[V1] Built daily context: {len(daily_ctx)} trading days with prev-day data.")

    # ==========================================
    # V2: Detect Setups + Label
    # ==========================================
    detector = SetupDetector()
    setups = detector.detect_setups(intraday_df, daily_ctx)
    detector.summarize_setups(setups)

    if not setups:
        print("\n[V2] No setups detected. Cannot proceed to V3.")
        print("     This could mean:")
        print("     1. The volume filter (1.5x) is too strict for the available data")
        print("     2. There were no prev-day-high breakouts in this period")
        print("     Consider relaxing volume_ratio or testing a different setup.")
        return

    labeler = TradeLabeler()
    labeled = labeler.label_setups(setups, intraday_df)
    labeler.summarize_labels(labeled)

    # ==========================================
    # V3: Backtest with Costs
    # ==========================================
    bt = Backtester()
    results_df, performance = bt.run(labeled)
    bt.print_report(results_df, performance)

    # Save results
    if results_df is not None:
        results_path = os.path.join(config.RESULTS_DIR, "v3_backtest_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n[V3] Results saved to {results_path}")


if __name__ == "__main__":
    main()
