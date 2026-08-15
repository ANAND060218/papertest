"""
V2/V3 DAILY TIMEFRAME -- Test setups on 10-year daily data
Since intraday data is limited to ~40 days (yfinance limit), we test
daily-timeframe setups on our 10-year daily OHLCV database to get
statistically meaningful results (2,400+ data points per symbol).

Daily Setups to test:
  1. Breakout above N-day High
  2. EMA Crossover (20/50)
  3. RSI Oversold Bounce
  4. Breakout + Volume
  5. Inside Bar Breakout

Target/Stop are defined as ATR multiples for dynamic sizing.
"""
import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from core.labeler import TradeLabeler
from backtest.backtester import Backtester, CostModel


class DailySetupDetector:
    """
    Detects trading setups on daily OHLCV data.
    Setups are tested on 10-year data for statistical significance.
    """

    def __init__(self):
        pass

    def compute_features(self, df):
        """Compute all needed features for daily data."""
        df = df.copy()

        # Moving averages
        df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # ATR (14-day)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14, min_periods=10).mean()

        # RSI (14-day)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=10).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Volume ratio
        df['vol_avg_20'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['volume_ratio'] = df['Volume'] / df['vol_avg_20'].replace(0, np.nan)

        # N-day highs/lows
        df['high_20'] = df['High'].rolling(20).max()
        df['low_20'] = df['Low'].rolling(20).min()
        df['high_50'] = df['High'].rolling(50).max()

        # Previous day values
        df['prev_close'] = df['Close'].shift(1)
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['prev_open'] = df['Open'].shift(1)

        # Inside bar detection
        df['is_inside_bar'] = (
            (df['High'] < df['prev_high']) &
            (df['Low'] > df['prev_low'])
        )

        return df

    # ----------------------------------------------------------
    # SETUP 1: N-Day High Breakout
    # ----------------------------------------------------------
    def detect_nday_breakout(self, df, n=20, atr_target=2.0, atr_stop=1.0, vol_min=1.0):
        """
        Close breaks above N-day high.
        Target: entry + ATR * target_multiplier
        Stop: entry - ATR * stop_multiplier
        """
        setups = []
        for i in range(n + 14, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            high_n = df.iloc[i-n:i]['High'].max()  # N-day high BEFORE today
            atr = row['atr_14']

            if pd.isna(atr) or atr <= 0:
                continue

            # Breakout: close above n-day high (excluding today)
            if row['Close'] > high_n and prev_row['Close'] <= high_n:
                vr = row['volume_ratio']
                if pd.notna(vr) and vr >= vol_min:
                    entry = row['Close']
                    setups.append({
                        'bar_index': i,
                        'timestamp': row['Date'],
                        'entry_price': round(entry, 2),
                        'stop_price': round(entry - atr * atr_stop, 2),
                        'target_price': round(entry + atr * atr_target, 2),
                        'trade_date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                        'max_hold_bars': 10,  # 10 trading days
                        'setup_name': f'{n}D_BREAKOUT',
                        'volume_ratio': round(vr, 2),
                        'atr': round(atr, 2),
                    })
        return setups

    # ----------------------------------------------------------
    # SETUP 2: EMA Crossover (Bullish)
    # ----------------------------------------------------------
    def detect_ema_crossover(self, df, fast=20, slow=50, atr_target=2.0, atr_stop=1.0):
        """
        EMA fast crosses above EMA slow (golden cross signal).
        """
        setups = []
        for i in range(slow + 14, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            atr = row['atr_14']

            if pd.isna(atr) or atr <= 0:
                continue

            ema_f = row[f'ema_{fast}']
            ema_s = row[f'ema_{slow}']
            prev_ema_f = prev_row[f'ema_{fast}']
            prev_ema_s = prev_row[f'ema_{slow}']

            if pd.isna(ema_f) or pd.isna(ema_s) or pd.isna(prev_ema_f) or pd.isna(prev_ema_s):
                continue

            # Crossover: fast was below slow, now above
            if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                entry = row['Close']
                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'entry_price': round(entry, 2),
                    'stop_price': round(entry - atr * atr_stop, 2),
                    'target_price': round(entry + atr * atr_target, 2),
                    'trade_date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                    'max_hold_bars': 15,
                    'setup_name': f'EMA_{fast}_{slow}_CROSS',
                    'volume_ratio': round(row['volume_ratio'], 2) if pd.notna(row['volume_ratio']) else 1.0,
                    'atr': round(atr, 2),
                })
        return setups

    # ----------------------------------------------------------
    # SETUP 3: RSI Oversold Bounce
    # ----------------------------------------------------------
    def detect_rsi_bounce(self, df, rsi_low=30, rsi_recover=35, atr_target=1.5, atr_stop=1.0):
        """
        RSI drops below threshold then recovers.
        """
        setups = []
        for i in range(3, len(df)):
            row = df.iloc[i]
            atr = row['atr_14']
            rsi = row['rsi_14']

            if pd.isna(atr) or atr <= 0 or pd.isna(rsi):
                continue

            # RSI was < rsi_low in last 3 days, now > rsi_recover
            recent_rsi = df.iloc[max(0, i-3):i]['rsi_14']
            was_oversold = (recent_rsi < rsi_low).any()

            if was_oversold and rsi > rsi_recover:
                entry = row['Close']
                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'entry_price': round(entry, 2),
                    'stop_price': round(entry - atr * atr_stop, 2),
                    'target_price': round(entry + atr * atr_target, 2),
                    'trade_date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                    'max_hold_bars': 10,
                    'setup_name': 'RSI_BOUNCE',
                    'volume_ratio': round(row['volume_ratio'], 2) if pd.notna(row['volume_ratio']) else 1.0,
                    'atr': round(atr, 2),
                })
        return setups

    # ----------------------------------------------------------
    # SETUP 4: Volume Breakout
    # ----------------------------------------------------------
    def detect_volume_breakout(self, df, vol_spike=2.0, atr_target=2.0, atr_stop=1.0):
        """
        Price up on very high volume (>2x average).
        Indicates institutional accumulation.
        """
        setups = []
        for i in range(20, len(df)):
            row = df.iloc[i]
            atr = row['atr_14']
            vr = row['volume_ratio']

            if pd.isna(atr) or atr <= 0 or pd.isna(vr):
                continue

            # Green candle + volume spike
            if row['Close'] > row['Open'] and vr >= vol_spike:
                entry = row['Close']
                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'entry_price': round(entry, 2),
                    'stop_price': round(entry - atr * atr_stop, 2),
                    'target_price': round(entry + atr * atr_target, 2),
                    'trade_date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                    'max_hold_bars': 10,
                    'setup_name': 'VOLUME_BREAKOUT',
                    'volume_ratio': round(vr, 2),
                    'atr': round(atr, 2),
                })
        return setups

    # ----------------------------------------------------------
    # SETUP 5: Inside Bar Breakout
    # ----------------------------------------------------------
    def detect_inside_bar_breakout(self, df, atr_target=2.0, atr_stop=1.0):
        """
        Previous day was an inside bar. Today breaks above prev-day high.
        Inside bars represent consolidation; breakout = expansion.
        """
        setups = []
        for i in range(2, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            atr = row['atr_14']

            if pd.isna(atr) or atr <= 0:
                continue

            # Previous bar was inside bar, current closes above prev high
            if prev_row.get('is_inside_bar', False) and row['Close'] > prev_row['High']:
                entry = row['Close']
                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'entry_price': round(entry, 2),
                    'stop_price': round(entry - atr * atr_stop, 2),
                    'target_price': round(entry + atr * atr_target, 2),
                    'trade_date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                    'max_hold_bars': 10,
                    'setup_name': 'INSIDE_BAR_BREAKOUT',
                    'volume_ratio': round(row['volume_ratio'], 2) if pd.notna(row['volume_ratio']) else 1.0,
                    'atr': round(atr, 2),
                })
        return setups


class DailyLabeler:
    """
    Labels daily setups by looking forward in the DAILY OHLCV data.
    Same logic as intraday labeler but operates on daily bars.
    """

    def label_setups(self, setups, daily_df):
        """Label each setup with TARGET/STOP/TIMEOUT."""
        labeled = []

        for setup in setups:
            entry_idx = setup['bar_index']
            entry_price = setup['entry_price']
            target_price = setup['target_price']
            stop_price = setup['stop_price']
            max_bars = setup['max_hold_bars']

            start_idx = entry_idx + 1
            end_idx = min(start_idx + max_bars, len(daily_df))
            future_bars = daily_df.iloc[start_idx:end_idx]

            if future_bars.empty:
                trade = {**setup}
                trade.update({
                    'result': 'TIMEOUT', 'exit_price': entry_price,
                    'exit_bar_index': entry_idx, 'exit_timestamp': setup['timestamp'],
                    'bars_held': 0, 'future_high': entry_price,
                    'future_low': entry_price, 'pnl_pct': 0.0
                })
                labeled.append(trade)
                continue

            result = 'TIMEOUT'
            exit_price = None
            exit_idx = None
            exit_ts = None
            bars_held = 0
            running_high = entry_price
            running_low = entry_price

            for j, (idx, bar) in enumerate(future_bars.iterrows()):
                running_high = max(running_high, bar['High'])
                running_low = min(running_low, bar['Low'])
                bars_held = j + 1

                target_hit = bar['High'] >= target_price
                stop_hit = bar['Low'] <= stop_price

                if target_hit and stop_hit:
                    result = 'STOP'
                    exit_price = stop_price
                    exit_idx = idx
                    exit_ts = bar['Date']
                    break
                elif target_hit:
                    result = 'TARGET'
                    exit_price = target_price
                    exit_idx = idx
                    exit_ts = bar['Date']
                    break
                elif stop_hit:
                    result = 'STOP'
                    exit_price = stop_price
                    exit_idx = idx
                    exit_ts = bar['Date']
                    break

            if result == 'TIMEOUT':
                last_bar = future_bars.iloc[-1]
                exit_price = last_bar['Close']
                exit_idx = future_bars.index[-1]
                exit_ts = last_bar['Date']
                bars_held = len(future_bars)

            pnl_pct = ((exit_price - entry_price) / entry_price) * 100

            trade = {**setup}
            trade.update({
                'result': result,
                'exit_price': round(exit_price, 2),
                'exit_bar_index': exit_idx,
                'exit_timestamp': exit_ts,
                'bars_held': bars_held,
                'future_high': round(running_high, 2),
                'future_low': round(running_low, 2),
                'pnl_pct': round(pnl_pct, 4),
            })
            labeled.append(trade)

        return labeled


# ============================================================
# MAIN: Run all daily setups on 10-year data
# ============================================================
if __name__ == "__main__":
    dm = DataManager()
    symbol = config.PRIMARY_SYMBOL

    print("=" * 100)
    print(f"V3 DAILY TIMEFRAME BACKTEST -- 10 YEARS -- {symbol}")
    print("=" * 100)

    daily_df = dm.load_daily(symbol)
    if daily_df is None or daily_df.empty:
        print(f"[ERROR] No daily data for {symbol}")
        sys.exit(1)

    print(f"Loaded {len(daily_df)} daily rows: {daily_df['Date'].min()} to {daily_df['Date'].max()}")

    # Compute features
    dsd = DailySetupDetector()
    df = dsd.compute_features(daily_df)

    # Detect all setups
    all_setups = {
        '20D_BREAKOUT': dsd.detect_nday_breakout(df, n=20, atr_target=2.0, atr_stop=1.0, vol_min=1.0),
        '20D_BREAKOUT_VOL': dsd.detect_nday_breakout(df, n=20, atr_target=2.0, atr_stop=1.0, vol_min=1.3),
        '50D_BREAKOUT': dsd.detect_nday_breakout(df, n=50, atr_target=2.5, atr_stop=1.0, vol_min=1.0),
        'EMA_20_50_CROSS': dsd.detect_ema_crossover(df, fast=20, slow=50, atr_target=2.0, atr_stop=1.0),
        'RSI_BOUNCE': dsd.detect_rsi_bounce(df, rsi_low=30, rsi_recover=35),
        'RSI_BOUNCE_TIGHT': dsd.detect_rsi_bounce(df, rsi_low=25, rsi_recover=30, atr_target=1.0, atr_stop=0.5),
        'VOLUME_BREAKOUT': dsd.detect_volume_breakout(df, vol_spike=2.0),
        'VOLUME_BREAKOUT_3X': dsd.detect_volume_breakout(df, vol_spike=3.0),
        'INSIDE_BAR': dsd.detect_inside_bar_breakout(df),
    }

    # Label and backtest each
    labeler = DailyLabeler()
    comparison = []

    for name, setups in all_setups.items():
        if not setups:
            comparison.append({
                'Setup': name, 'Trades': 0, 'TARGET': 0, 'STOP': 0, 'TIMEOUT': 0,
                'WinRate%': 0, 'PF': 0, 'Expectancy': 0, 'NetPnL': 0, 'MaxDD%': 0,
            })
            continue

        labeled = labeler.label_setups(setups, df)
        bt = Backtester(initial_capital=config.INITIAL_CAPITAL)
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
                'Return%': perf['total_return_pct'],
                'Sharpe': perf['sharpe_ratio'],
            })

    # Print comparison
    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('PF', ascending=False)

    print(f"\n{'=' * 120}")
    print(f"SETUP COMPARISON -- 10 YEARS -- {symbol} (sorted by Profit Factor)")
    print(f"{'=' * 120}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(comp_df.to_string(index=False))

    # Decision gate
    best = comp_df.iloc[0] if len(comp_df) > 0 else None
    print(f"\n{'=' * 120}")
    print("V3 DECISION GATE (10-Year Daily)")
    print(f"{'=' * 120}")

    if best is not None and best['Trades'] >= 30:
        if best['PF'] >= 1.15 and best['Expectancy'] > 0:
            print(f"  >>> POSITIVE EDGE: {best['Setup']}")
            print(f"      PF={best['PF']:.2f}, WinRate={best['WinRate%']:.1f}%, "
                  f"Expectancy=Rs.{best['Expectancy']:.2f}, Trades={best['Trades']}")
            print(f"  --> PROCEED to V4 (XGBoost on this setup)")
        elif best['PF'] >= 1.0 and best['Expectancy'] > 0:
            print(f"  >>> MARGINAL EDGE: {best['Setup']}")
            print(f"      PF={best['PF']:.2f} -- edge exists but weak")
            print(f"  --> Consider V4 to see if ML improves it")
        else:
            print(f"  >>> NO EDGE in daily setups either.")
            print(f"      Best PF={best['PF']:.2f}")
    elif best is not None:
        print(f"  >>> INSUFFICIENT SAMPLE: {best['Setup']} has only {best['Trades']} trades")
        print(f"      Need 30+ trades for statistical significance")

    # Positive-edge setups
    positive = comp_df[(comp_df['PF'] >= 1.0) & (comp_df['Expectancy'] > 0) & (comp_df['Trades'] >= 10)]
    if not positive.empty:
        print(f"\n  All setups with PF >= 1.0 and positive expectancy:")
        for _, row in positive.iterrows():
            print(f"    {row['Setup']:25s} PF={row['PF']:>6.2f}  Win={row['WinRate%']:>5.1f}%  "
                  f"Trades={row['Trades']:>4d}  Net=Rs.{row['NetPnL']:>+10.2f}")

    # Save
    comp_path = os.path.join(config.RESULTS_DIR, "v3_daily_10yr_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[V3] Saved to {comp_path}")
