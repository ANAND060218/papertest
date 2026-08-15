"""
V2 ALTERNATIVE -- Multi-Setup Detector
Tests multiple trading setups to find one with a real edge.

Setups to test:
  1. Previous-Day High Breakout (original, failed V3)
  2. Opening Range Breakout (ORB) -- 15-min opening range
  3. VWAP Breakout -- Price crosses above VWAP with volume
  4. Mean Reversion -- RSI oversold bounce
  5. Opening Momentum -- Strong first-15-min move continues

Each setup is tested with the SAME labeler and SAME cost model.
We compare them fairly and pick the one that survives V3.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import time as dtime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MultiSetupDetector:
    """
    Detects multiple types of trading setups from intraday 5-minute data.
    All setups output the same format for the labeler.
    """

    def __init__(self):
        self.min_time = dtime(config.BREAKOUT_MIN_TIME_HOUR, config.BREAKOUT_MIN_TIME_MINUTE)
        self.max_time = dtime(config.BREAKOUT_MAX_TIME_HOUR, config.BREAKOUT_MAX_TIME_MINUTE)

    def _compute_rolling_features(self, df):
        """Add rolling features needed by multiple setups."""
        df = df.copy()

        # Volume ratio
        df['vol_avg_20'] = df['Volume'].rolling(window=20, min_periods=10).mean()
        df['volume_ratio'] = df['Volume'] / df['vol_avg_20'].replace(0, np.nan)

        # RSI (14-bar)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=10).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # EMA 20
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # VWAP (cumulative per day)
        df['trade_date'] = df['Date'].dt.date
        df['cum_vol'] = df.groupby('trade_date')['Volume'].cumsum()
        df['pv'] = df['Close'] * df['Volume']
        df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
        df['cum_vwap'] = df['cum_pv'] / df['cum_vol'].replace(0, np.nan)

        df['bar_time'] = df['Date'].dt.time

        return df

    # ----------------------------------------------------------
    # SETUP 1: Previous-Day High Breakout (original)
    # ----------------------------------------------------------
    def detect_prev_day_high_breakout(self, df, daily_ctx, vol_min=1.5):
        """Same as original SetupDetector."""
        prev_high_lookup = {}
        for _, row in daily_ctx.iterrows():
            prev_high_lookup[row['trade_date']] = row['prev_day_high']

        setups = []
        prev_bar_above = False

        for i in range(1, len(df)):
            row = df.iloc[i]
            td = row['trade_date']
            bt = row['bar_time']

            if td not in prev_high_lookup:
                prev_bar_above = False
                continue

            prev_high = prev_high_lookup[td]
            if bt < self.min_time or bt > self.max_time:
                prev_bar_above = (row['Close'] > prev_high)
                continue

            current_above = row['Close'] > prev_high
            if current_above and not prev_bar_above:
                vr = row['volume_ratio']
                if pd.notna(vr) and vr >= vol_min:
                    entry = row['Close']
                    setups.append(self._make_setup(
                        i, row['Date'], entry, td,
                        target_pct=1.0, stop_pct=0.5,
                        setup_name='PREV_DAY_HIGH_BREAKOUT',
                        volume_ratio=vr
                    ))

            prev_bar_above = current_above

        return setups

    # ----------------------------------------------------------
    # SETUP 2: Opening Range Breakout (ORB)
    # ----------------------------------------------------------
    def detect_orb(self, df, orb_minutes=15, vol_min=1.2):
        """
        Opening Range Breakout: The first 15 minutes (3 x 5min bars)
        define the range. A breakout above the range high with volume = LONG setup.

        Rules:
          - ORB High = max(High) of first 3 bars of the day
          - ORB Low  = min(Low) of first 3 bars of the day
          - Entry when Close > ORB High AND volume_ratio > 1.2
          - Target: +0.8%
          - Stop: ORB Low or -0.4%, whichever is closer
          - Max hold: 30 bars
        """
        setups = []
        orb_bars = 3  # 3 x 5min = 15 minutes

        for trade_date, day_df in df.groupby('trade_date'):
            day_df = day_df.sort_values('Date').reset_index(drop=True)
            if len(day_df) < orb_bars + 5:
                continue

            # Calculate ORB range
            orb_slice = day_df.iloc[:orb_bars]
            orb_high = orb_slice['High'].max()
            orb_low = orb_slice['Low'].min()

            # Scan bars after ORB period
            triggered = False
            for j in range(orb_bars, len(day_df)):
                bar = day_df.iloc[j]
                bt = bar['bar_time']
                if bt > self.max_time:
                    break

                if not triggered and bar['Close'] > orb_high:
                    vr = bar['volume_ratio']
                    if pd.notna(vr) and vr >= vol_min:
                        entry = bar['Close']
                        # Stop = ORB low or -0.4%, whichever is tighter
                        stop_orb = orb_low
                        stop_pct = entry * 0.004
                        stop = max(stop_orb, entry - stop_pct)
                        stop_actual_pct = ((entry - stop) / entry) * 100

                        # Find the actual index in the full df
                        actual_idx = df.index[df['Date'] == bar['Date']]
                        if len(actual_idx) == 0:
                            continue
                        actual_idx = actual_idx[0]

                        setups.append({
                            'bar_index': actual_idx,
                            'timestamp': bar['Date'],
                            'entry_price': round(entry, 2),
                            'stop_price': round(stop, 2),
                            'target_price': round(entry * 1.008, 2),
                            'trade_date': trade_date,
                            'max_hold_bars': 30,
                            'setup_name': 'ORB',
                            'volume_ratio': round(vr, 2),
                            'orb_high': round(orb_high, 2),
                            'orb_low': round(orb_low, 2),
                        })
                        triggered = True  # Only one ORB entry per day

        return setups

    # ----------------------------------------------------------
    # SETUP 3: VWAP Breakout
    # ----------------------------------------------------------
    def detect_vwap_breakout(self, df, vol_min=1.3):
        """
        VWAP Breakout: Price crosses above intraday VWAP with volume.
        Indicates institutional buying.

        Rules:
          - Previous bar Close < VWAP
          - Current bar Close > VWAP
          - volume_ratio > 1.3
          - Target: +0.7%
          - Stop: -0.4%
          - Max hold: 24 bars (2 hours)
        """
        setups = []
        seen_dates = set()

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            bt = row['bar_time']
            td = row['trade_date']

            if bt < self.min_time or bt > self.max_time:
                continue

            # Only one VWAP cross per day
            if td in seen_dates:
                continue

            vwap = row['cum_vwap']
            if pd.isna(vwap) or vwap <= 0:
                continue

            # Cross above VWAP
            prev_vwap = prev_row['cum_vwap']
            if pd.isna(prev_vwap):
                continue

            if prev_row['Close'] < prev_vwap and row['Close'] > vwap:
                vr = row['volume_ratio']
                if pd.notna(vr) and vr >= vol_min:
                    entry = row['Close']
                    setups.append(self._make_setup(
                        i, row['Date'], entry, td,
                        target_pct=0.7, stop_pct=0.4,
                        setup_name='VWAP_BREAKOUT',
                        volume_ratio=vr,
                        max_hold=24
                    ))
                    seen_dates.add(td)

        return setups

    # ----------------------------------------------------------
    # SETUP 4: Mean Reversion (RSI Oversold Bounce)
    # ----------------------------------------------------------
    def detect_mean_reversion(self, df, rsi_threshold=30, vol_min=1.0):
        """
        Mean Reversion: RSI drops below threshold then bounces.

        Rules:
          - RSI was < 30 within last 3 bars
          - Current RSI > 30 (bouncing)
          - Price > EMA 20 (confirming bounce direction)
          - Target: +0.6%
          - Stop: -0.3%
          - Max hold: 20 bars
        """
        setups = []
        seen_dates = set()

        for i in range(3, len(df)):
            row = df.iloc[i]
            bt = row['bar_time']
            td = row['trade_date']

            if bt < self.min_time or bt > self.max_time:
                continue
            if td in seen_dates:
                continue

            rsi = row['rsi_14']
            if pd.isna(rsi):
                continue

            # Check if RSI was < threshold in last 3 bars
            recent_rsi = df.iloc[max(0, i-3):i]['rsi_14']
            was_oversold = (recent_rsi < rsi_threshold).any()

            if was_oversold and rsi > rsi_threshold:
                # Confirm bounce direction
                ema = row['ema_20']
                if pd.notna(ema) and row['Close'] > ema:
                    entry = row['Close']
                    setups.append(self._make_setup(
                        i, row['Date'], entry, td,
                        target_pct=0.6, stop_pct=0.3,
                        setup_name='MEAN_REVERSION',
                        volume_ratio=row.get('volume_ratio', 1.0),
                        max_hold=20
                    ))
                    seen_dates.add(td)

        return setups

    # ----------------------------------------------------------
    # SETUP 5: Opening Momentum
    # ----------------------------------------------------------
    def detect_opening_momentum(self, df, momentum_pct=0.3, vol_min=1.5):
        """
        Opening Momentum: If the first 15 minutes show strong upward momentum,
        the move often continues.

        Rules:
          - Price change in first 15min > +0.3%
          - Volume in first 15min > 1.5x average
          - Entry at 09:35 or 09:40 (after confirming momentum)
          - Target: +0.8%
          - Stop: -0.4%
          - Max hold: 30 bars
        """
        setups = []

        for trade_date, day_df in df.groupby('trade_date'):
            day_df = day_df.sort_values('Date').reset_index(drop=True)
            if len(day_df) < 6:
                continue

            # First 3 bars (15 min)
            first_3 = day_df.iloc[:3]
            open_price = first_3.iloc[0]['Open']
            close_15m = first_3.iloc[-1]['Close']
            pct_change = ((close_15m - open_price) / open_price) * 100

            if pct_change > momentum_pct:
                # Entry on 4th bar (09:35 or later)
                entry_bar = day_df.iloc[3]
                vr = entry_bar['volume_ratio']
                if pd.notna(vr) and vr >= vol_min:
                    entry = entry_bar['Close']
                    actual_idx = df.index[df['Date'] == entry_bar['Date']]
                    if len(actual_idx) == 0:
                        continue
                    actual_idx = actual_idx[0]

                    setups.append({
                        'bar_index': actual_idx,
                        'timestamp': entry_bar['Date'],
                        'entry_price': round(entry, 2),
                        'stop_price': round(entry * (1 - 0.004), 2),
                        'target_price': round(entry * (1 + 0.008), 2),
                        'trade_date': trade_date,
                        'max_hold_bars': 30,
                        'setup_name': 'OPENING_MOMENTUM',
                        'volume_ratio': round(vr, 2),
                        'opening_move_pct': round(pct_change, 2),
                    })

        return setups

    # ----------------------------------------------------------
    # HELPER
    # ----------------------------------------------------------
    def _make_setup(self, idx, timestamp, entry, trade_date,
                    target_pct, stop_pct, setup_name, volume_ratio=1.0, max_hold=30):
        return {
            'bar_index': idx,
            'timestamp': timestamp,
            'entry_price': round(entry, 2),
            'stop_price': round(entry * (1 - stop_pct/100), 2),
            'target_price': round(entry * (1 + target_pct/100), 2),
            'trade_date': trade_date,
            'max_hold_bars': max_hold,
            'setup_name': setup_name,
            'volume_ratio': round(volume_ratio, 2) if pd.notna(volume_ratio) else 1.0,
        }


# ============================================================
# COMPARATIVE RUNNER
# ============================================================
if __name__ == "__main__":
    from core.data_manager import DataManager
    from core.labeler import TradeLabeler
    from backtest.backtester import Backtester

    dm = DataManager()
    intraday_df = dm.load_intraday(config.PRIMARY_SYMBOL)
    if intraday_df is None:
        print("[ERROR] No intraday data. Run data_manager.py first.")
        sys.exit(1)

    daily_ctx = dm.build_daily_context(intraday_df)

    msd = MultiSetupDetector()

    # Compute features once
    df_features = msd._compute_rolling_features(intraday_df)

    # Detect all setups
    all_setups = {
        'PREV_DAY_HIGH': msd.detect_prev_day_high_breakout(df_features, daily_ctx, vol_min=1.5),
        'ORB': msd.detect_orb(df_features, vol_min=1.2),
        'VWAP_BREAKOUT': msd.detect_vwap_breakout(df_features, vol_min=1.3),
        'MEAN_REVERSION': msd.detect_mean_reversion(df_features, rsi_threshold=30, vol_min=1.0),
        'OPENING_MOMENTUM': msd.detect_opening_momentum(df_features, momentum_pct=0.3, vol_min=1.5),
    }

    # Also test with relaxed volume filters
    all_setups['PREV_DAY_HIGH_RELAXED'] = msd.detect_prev_day_high_breakout(df_features, daily_ctx, vol_min=1.0)
    all_setups['ORB_RELAXED'] = msd.detect_orb(df_features, vol_min=1.0)
    all_setups['VWAP_RELAXED'] = msd.detect_vwap_breakout(df_features, vol_min=1.0)
    all_setups['OPENING_MOMENTUM_RELAXED'] = msd.detect_opening_momentum(df_features, momentum_pct=0.2, vol_min=1.0)

    labeler = TradeLabeler()

    print(f"\n{'=' * 100}")
    print("V3 COMPARATIVE BACKTEST -- ALL SETUPS")
    print(f"{'=' * 100}")
    print(f"Symbol: {config.PRIMARY_SYMBOL}")
    print(f"Data: {intraday_df['Date'].min()} to {intraday_df['Date'].max()}")
    print(f"Trading days with prev-day context: {len(daily_ctx)}")

    comparison = []

    for name, setups in all_setups.items():
        if not setups:
            comparison.append({
                'Setup': name,
                'Trades': 0,
                'TARGET': 0,
                'STOP': 0,
                'TIMEOUT': 0,
                'WinRate%': 0,
                'PF': 0,
                'Expectancy': 0,
                'NetPnL': 0,
                'MaxDD%': 0,
            })
            continue

        labeled = labeler.label_setups(setups, intraday_df)
        bt = Backtester()
        results_df, perf = bt.run(labeled)

        if perf:
            comparison.append({
                'Setup': name,
                'Trades': perf['total_trades'],
                'TARGET': perf['targets'],
                'STOP': perf['stops'],
                'TIMEOUT': perf['timeouts'],
                'WinRate%': perf['win_rate'],
                'PF': perf['profit_factor'],
                'Expectancy': perf['expectancy'],
                'NetPnL': perf['total_pnl_net'],
                'MaxDD%': perf['max_drawdown_pct'],
            })

    # Print comparison table
    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('PF', ascending=False)

    print(f"\n{'=' * 100}")
    print("SETUP COMPARISON (sorted by Profit Factor)")
    print(f"{'=' * 100}")
    print(comp_df.to_string(index=False))

    # Decision
    best = comp_df.iloc[0] if len(comp_df) > 0 else None
    if best is not None and best['PF'] >= 1.15 and best['Expectancy'] > 0:
        print(f"\n>>> BEST SETUP: {best['Setup']} with PF={best['PF']:.2f}, Expectancy=Rs.{best['Expectancy']:.2f}")
        print(">>> Proceed to V4 with this setup.")
    elif best is not None and best['PF'] >= 1.0 and best['Expectancy'] > 0:
        print(f"\n>>> MARGINAL: {best['Setup']} with PF={best['PF']:.2f}")
        print(">>> Consider V4 to see if ML improves it, but be cautious.")
    else:
        print("\n>>> NO SETUP shows a positive edge in this ~40-day window.")
        print(">>> This is expected -- 40 days is a small sample.")
        print(">>> Options:")
        print("    1. Fetch more data (daily data from 10yr DB for daily-timeframe setups)")
        print("    2. Relax filters further")
        print("    3. Test daily-timeframe setups using the 10-year dataset")

    # Save comparison
    comp_path = os.path.join(config.RESULTS_DIR, "v3_setup_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[V3] Comparison saved to {comp_path}")
