"""
V2 -- Trade Outcome Labeler
Labels each detected setup with its actual outcome by looking forward in price data.

Three outcomes (NOT binary):
  TARGET  = Price hit target BEFORE hitting stop   -> win
  STOP    = Price hit stop BEFORE hitting target    -> loss
  TIMEOUT = Neither hit within max holding period   -> analyzed separately

This is the MOST IMPORTANT file in the system.
ML learns from these labels. If labeling is wrong, everything downstream is wrong.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TradeLabeler:
    """
    Labels trade setups with actual forward-looking outcomes.
    For each setup, looks forward in the intraday data to determine
    whether TARGET, STOP, or TIMEOUT was hit first.
    """

    def label_setups(self, setups, intraday_df):
        """
        For each setup, look forward in intraday_df to determine the outcome.

        Args:
            setups: List of setup dicts from SetupDetector
            intraday_df: Full intraday DataFrame

        Returns:
            List of labeled trade dicts with additional fields:
            {
                ...setup fields...
                'result': 'TARGET' | 'STOP' | 'TIMEOUT',
                'exit_price': float,
                'exit_bar_index': int,
                'exit_timestamp': datetime,
                'bars_held': int,
                'future_high': float,   # highest price during holding period
                'future_low': float,    # lowest price during holding period
                'pnl_pct': float,       # (exit - entry) / entry * 100
            }
        """
        labeled_trades = []

        if 'trade_date' not in intraday_df.columns:
            intraday_df = intraday_df.copy()
            intraday_df['trade_date'] = intraday_df['Date'].dt.date

        dates_arr = intraday_df['trade_date'].values
        highs_arr = intraday_df['High'].values
        lows_arr = intraday_df['Low'].values
        closes_arr = intraday_df['Close'].values
        timestamps_arr = intraday_df['Date'].values
        n_bars_total = len(intraday_df)

        for setup in setups:
            entry_idx = setup['bar_index']
            entry_price = setup['entry_price']
            target_price = setup['target_price']
            stop_price = setup['stop_price']
            max_bars = setup['max_hold_bars']
            trade_date = setup['trade_date']
            direction = setup.get('direction', 'LONG')

            start_idx = entry_idx + 1
            end_idx = min(start_idx + max_bars, n_bars_total)

            if start_idx >= n_bars_total or dates_arr[start_idx] != trade_date:
                trade = {**setup}
                trade['result'] = 'TIMEOUT'
                trade['exit_price'] = entry_price
                trade['exit_bar_index'] = entry_idx
                trade['exit_timestamp'] = setup['timestamp']
                trade['bars_held'] = 0
                trade['future_high'] = entry_price
                trade['future_low'] = entry_price
                trade['pnl_pct'] = 0.0
                labeled_trades.append(trade)
                continue

            result = 'TIMEOUT'
            exit_price = None
            exit_idx = None
            exit_ts = None
            bars_held = 0
            running_high = entry_price
            running_low = entry_price

            for k in range(start_idx, end_idx):
                if dates_arr[k] != trade_date:
                    break # Reached end of trading day

                b_high = highs_arr[k]
                b_low = lows_arr[k]
                b_close = closes_arr[k]
                b_ts = timestamps_arr[k]

                running_high = max(running_high, b_high)
                running_low = min(running_low, b_low)
                bars_held += 1

                if direction == 'SHORT':
                    target_hit = b_low <= target_price
                    stop_hit = b_high >= stop_price
                else: # LONG
                    target_hit = b_high >= target_price
                    stop_hit = b_low <= stop_price

                if target_hit and stop_hit:
                    result = 'STOP'
                    exit_price = stop_price
                    exit_idx = k
                    exit_ts = b_ts
                    break
                elif target_hit:
                    result = 'TARGET'
                    exit_price = target_price
                    exit_idx = k
                    exit_ts = b_ts
                    break
                elif stop_hit:
                    result = 'STOP'
                    exit_price = stop_price
                    exit_idx = k
                    exit_ts = b_ts
                    break

            if result == 'TIMEOUT':
                last_k = start_idx + bars_held - 1
                exit_price = closes_arr[last_k]
                exit_idx = last_k
                exit_ts = timestamps_arr[last_k]

            if direction == 'SHORT':
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            else:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100

            trade = {**setup}
            trade['result'] = result
            trade['exit_price'] = round(exit_price, 2)
            trade['exit_bar_index'] = exit_idx
            trade['exit_timestamp'] = exit_ts
            trade['bars_held'] = bars_held
            trade['future_high'] = round(running_high, 2)
            trade['future_low'] = round(running_low, 2)
            trade['pnl_pct'] = round(pnl_pct, 4)

            labeled_trades.append(trade)

        return labeled_trades

    def summarize_labels(self, labeled_trades):
        """Print a detailed summary of labeled trade outcomes."""
        if not labeled_trades:
            print("\n[V2] No labeled trades to summarize.")
            return

        df = pd.DataFrame(labeled_trades)

        print(f"\n{'=' * 90}")
        print("V2 TRADE LABELING RESULTS")
        print(f"{'=' * 90}")
        print(f"Total setups labeled: {len(df)}")

        # Outcome distribution
        print(f"\nOutcome Distribution:")
        for result in ['TARGET', 'STOP', 'TIMEOUT']:
            count = (df['result'] == result).sum()
            pct = count / len(df) * 100 if len(df) > 0 else 0
            avg_pnl = df[df['result'] == result]['pnl_pct'].mean() if count > 0 else 0
            avg_bars = df[df['result'] == result]['bars_held'].mean() if count > 0 else 0
            print(f"  {result:>8s}: {count:>4d} ({pct:>5.1f}%)  avg P&L: {avg_pnl:>+7.3f}%  avg bars held: {avg_bars:>5.1f}")

        # Win rate (TARGET only as win)
        targets = (df['result'] == 'TARGET').sum()
        stops = (df['result'] == 'STOP').sum()
        timeouts = (df['result'] == 'TIMEOUT').sum()

        if targets + stops > 0:
            win_rate_excl_timeout = targets / (targets + stops) * 100
            print(f"\nWin Rate (excl timeouts): {win_rate_excl_timeout:.1f}%")

        win_rate_all = targets / len(df) * 100 if len(df) > 0 else 0
        print(f"Win Rate (all trades):   {win_rate_all:.1f}%")

        # Average P&L
        print(f"\nAverage P&L per trade: {df['pnl_pct'].mean():+.4f}%")
        print(f"Total P&L (sum):       {df['pnl_pct'].sum():+.4f}%")

        # Detail table
        print(f"\n{'Timestamp':>22s} {'Entry':>10s} {'Stop':>10s} {'Target':>10s} "
              f"{'Exit':>10s} {'Result':>8s} {'Bars':>5s} {'P&L%':>8s} {'FutHigh':>10s} {'FutLow':>10s}")
        for _, t in df.iterrows():
            print(
                f"{str(t['timestamp']):>22s} "
                f"{t['entry_price']:>10.2f} "
                f"{t['stop_price']:>10.2f} "
                f"{t['target_price']:>10.2f} "
                f"{t['exit_price']:>10.2f} "
                f"{t['result']:>8s} "
                f"{t['bars_held']:>5d} "
                f"{t['pnl_pct']:>+8.4f} "
                f"{t['future_high']:>10.2f} "
                f"{t['future_low']:>10.2f}"
            )

        return df


if __name__ == "__main__":
    from data_manager import DataManager
    from setup_detector import SetupDetector

    dm = DataManager()
    print("Loading intraday data...")
    intraday_df = dm.load_intraday(config.PRIMARY_SYMBOL)
    if intraday_df is None:
        print("[ERROR] No intraday data. Run data_manager.py first.")
        sys.exit(1)

    daily_ctx = dm.build_daily_context(intraday_df)

    # Detect setups
    detector = SetupDetector()
    setups = detector.detect_setups(intraday_df, daily_ctx)
    detector.summarize_setups(setups)

    # Label outcomes
    labeler = TradeLabeler()
    labeled = labeler.label_setups(setups, intraday_df)
    labeler.summarize_labels(labeled)
