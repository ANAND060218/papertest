"""
V2 -- Setup Detector: Previous-Day High Breakout
Detects trading setups based on FIXED rules defined BEFORE looking at results.

Setup: Previous-Day High Breakout
  Entry:  Price crosses above previous day's high
  Filter: Volume must be >= 1.5x the 20-bar average
  Stop:   Entry - 0.5%
  Target: Entry + 1.0%
  Max Hold: 30 bars (2.5 hours at 5m interval)
  Time Filter: Only between 09:30 and 14:30 IST (skip opening chaos, no late entries)
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import time as dtime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SetupDetector:
    """
    Detects Previous-Day High Breakout setups from intraday 5-minute data.
    Rules are fixed in config.py. Do NOT change after backtesting.
    """

    def __init__(self):
        # Parameters from config (frozen)
        self.vol_ratio_min = config.BREAKOUT_VOLUME_RATIO_MIN
        self.target_pct = config.BREAKOUT_TARGET_PCT / 100.0
        self.stop_pct = config.BREAKOUT_STOP_PCT / 100.0
        self.max_hold_bars = config.BREAKOUT_MAX_HOLDING_BARS

        self.min_time = dtime(config.BREAKOUT_MIN_TIME_HOUR, config.BREAKOUT_MIN_TIME_MINUTE)
        self.max_time = dtime(config.BREAKOUT_MAX_TIME_HOUR, config.BREAKOUT_MAX_TIME_MINUTE)

    def detect_setups(self, intraday_df, daily_context_df):
        """
        Scan intraday data for Previous-Day High Breakout setups.

        Args:
            intraday_df: Clean 5-minute OHLCV DataFrame (Date, Open, High, Low, Close, Volume, Symbol)
            daily_context_df: Daily context with prev_day_high etc.

        Returns:
            List of setup dicts, each containing:
            {
                'bar_index': int,          # index in intraday_df
                'timestamp': datetime,
                'entry_price': float,      # Close of breakout bar (market entry next bar)
                'prev_day_high': float,
                'volume_ratio': float,
                'stop_price': float,
                'target_price': float,
                'trade_date': date,
            }
        """
        df = intraday_df.copy()
        df['trade_date'] = df['Date'].dt.date
        df['bar_time'] = df['Date'].dt.time

        # Calculate rolling 20-bar volume average
        df['vol_avg_20'] = df['Volume'].rolling(window=20, min_periods=10).mean()
        df['volume_ratio'] = df['Volume'] / df['vol_avg_20'].replace(0, np.nan)

        # Build lookup: trade_date -> prev_day_high
        prev_high_lookup = {}
        for _, row in daily_context_df.iterrows():
            prev_high_lookup[row['trade_date']] = row['prev_day_high']

        setups = []
        prev_bar_above = False  # Track if previous bar was already above prev-day high

        for i in range(1, len(df)):
            row = df.iloc[i]
            trade_date = row['trade_date']
            bar_time = row['bar_time']

            # Skip if no previous-day context
            if trade_date not in prev_high_lookup:
                prev_bar_above = False
                continue

            prev_day_high = prev_high_lookup[trade_date]

            # Time filter: only 09:30 to 14:30
            if bar_time < self.min_time or bar_time > self.max_time:
                prev_bar_above = (row['Close'] > prev_day_high)
                continue

            # Breakout condition: current bar closes above previous day's high
            # AND the previous bar was NOT already above (we want the FIRST break)
            current_above = row['Close'] > prev_day_high

            if current_above and not prev_bar_above:
                # Volume filter
                vol_ratio = row['volume_ratio']
                if pd.notna(vol_ratio) and vol_ratio >= self.vol_ratio_min:
                    entry_price = row['Close']
                    stop_price = entry_price * (1 - self.stop_pct)
                    target_price = entry_price * (1 + self.target_pct)

                    setups.append({
                        'bar_index': i,
                        'timestamp': row['Date'],
                        'entry_price': round(entry_price, 2),
                        'prev_day_high': round(prev_day_high, 2),
                        'volume_ratio': round(vol_ratio, 2),
                        'stop_price': round(stop_price, 2),
                        'target_price': round(target_price, 2),
                        'trade_date': trade_date,
                        'max_hold_bars': self.max_hold_bars,
                    })

            prev_bar_above = current_above

        return setups

    def summarize_setups(self, setups):
        """Print a summary of detected setups."""
        if not setups:
            print("\n[V2] No setups detected.")
            return

        print(f"\n{'=' * 90}")
        print(f"V2 SETUP DETECTION SUMMARY: Previous-Day High Breakout")
        print(f"{'=' * 90}")
        print(f"Total setups found: {len(setups)}")
        print(f"\nRules applied:")
        print(f"  - Entry: Close > Previous Day High (first bar to break)")
        print(f"  - Volume: >= {self.vol_ratio_min}x 20-bar average")
        print(f"  - Time: {self.min_time} to {self.max_time}")
        print(f"  - Target: +{self.target_pct*100:.1f}%  |  Stop: -{self.stop_pct*100:.1f}%")
        print(f"  - Max hold: {self.max_hold_bars} bars")

        print(f"\n{'Timestamp':>22s} {'Entry':>10s} {'PrevHigh':>10s} {'Stop':>10s} {'Target':>10s} {'VolRatio':>10s}")
        print(f"{'':>22s} {'':>10s} {'':>10s} {'':>10s} {'':>10s} {'':>10s}")
        for s in setups:
            print(
                f"{str(s['timestamp']):>22s} "
                f"{s['entry_price']:>10.2f} "
                f"{s['prev_day_high']:>10.2f} "
                f"{s['stop_price']:>10.2f} "
                f"{s['target_price']:>10.2f} "
                f"{s['volume_ratio']:>10.2f}"
            )


if __name__ == "__main__":
    from data_manager import DataManager

    dm = DataManager()
    print("Loading intraday data...")
    intraday_df = dm.load_intraday(config.PRIMARY_SYMBOL)
    if intraday_df is None:
        print("[ERROR] No intraday data. Run data_manager.py first.")
        sys.exit(1)

    daily_ctx = dm.build_daily_context(intraday_df)

    detector = SetupDetector()
    setups = detector.detect_setups(intraday_df, daily_ctx)
    detector.summarize_setups(setups)
