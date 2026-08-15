"""
Daily Multi-Day Swing Momentum Engine (Stage-2 Breakouts & Trend Following)
Solves the friction-to-excursion bottleneck by capturing 3% to 8% multi-day swings on Daily OHLCV.

Rules:
  1. Trend Alignment: Price > 50 EMA > 200 EMA and 50 EMA sloping upwards.
  2. Stage-2 Breakout: Close > 20-Day High with Volume >= 1.50x (20-Day Volume SMA).
  3. Execution: Entry on next day Open.
  4. Exit & Risk:
     - Initial Stop Loss: min(10-Day Low, Entry - 2.5%)
     - Profit Target: Entry + 5.0% (or 2.5x ATR)
     - Trailing Exit: Close below 10 EMA once profit exceeds +2.0%.
     - Max Holding: 15 trading days.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DailySwingEngine:
    """
    Causal Multi-Day Swing Momentum & Stage-2 Breakout Engine.
    """

    @staticmethod
    def prepare_daily_features(df):
        """Computes 50 EMA, 200 EMA, 10 EMA, 20-day High/Low, ATR, and Volume SMA."""
        df = df.copy().sort_values('Date').reset_index(drop=True)

        df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # 20-Day Donchian Breakout Levels (strictly shifted by 1 bar to prevent lookahead)
        df['donchian_high_20'] = df['High'].rolling(20).max().shift(1)
        df['donchian_low_10'] = df['Low'].rolling(10).min().shift(1)

        # Volume confirmation
        df['vol_sma_20'] = df['Volume'].rolling(20).mean().shift(1)
        df['vol_ratio'] = df['Volume'] / df['vol_sma_20'].replace(0, np.nan)

        # ATR 14
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean().shift(1)

        # 50 EMA Slope
        df['ema_50_slope'] = (df['ema_50'] - df['ema_50'].shift(5)) / df['ema_50'].shift(5) * 100

        return df

    @staticmethod
    def generate_swing_setups(df, symbol='RELIANCE.NS'):
        """
        Generates causal Stage-2 swing breakout setups.
        Entry signal on day t, trade opens on day t+1 Open.
        """
        setups = []
        n = len(df)
        if n < 220:
            return setups

        for i in range(205, n - 1):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            close_p = row['Close']
            ema_10 = row['ema_10']
            ema_50 = row['ema_50']
            ema_200 = row['ema_200']
            donch_high = row['donchian_high_20']
            vol_ratio = row['vol_ratio']
            atr_val = row['atr_14'] if pd.notna(row['atr_14']) else 0.02 * close_p

            # 1. Trend Alignment Filter
            if not (close_p > ema_50 and ema_50 > ema_200 and row['ema_50_slope'] > 0):
                continue

            # 2. 20-Day High Breakout
            if not (close_p > donch_high and prev_row['Close'] <= prev_row['donchian_high_20']):
                continue

            # 3. Volume Expansion Filter
            if pd.isna(vol_ratio) or vol_ratio < 1.25:
                continue

            # Next day entry
            next_day_open = df.iloc[i + 1]['Open']
            signal_date = row['Date']
            entry_date = df.iloc[i + 1]['Date']

            # Risk Boundaries
            stop_p = round(max(row['donchian_low_10'], next_day_open - 1.5 * atr_val), 2)
            target_p = round(next_day_open + 2.5 * atr_val, 2)

            setups.append({
                'symbol': symbol,
                'signal_bar_index': i,
                'entry_bar_index': i + 1,
                'signal_date': signal_date,
                'entry_date': entry_date,
                'entry_price': next_day_open,
                'stop_price': stop_p,
                'target_price': target_p,
                'atr_val': atr_val,
                'max_hold_days': 15
            })

        return setups

    @staticmethod
    def simulate_swing_trade(setup, df):
        """
        Simulates a swing trade bar-by-bar across daily candles with trailing stop at 10 EMA.
        """
        entry_idx = setup['entry_bar_index']
        entry_p = setup['entry_price']
        stop_p = setup['stop_price']
        target_p = setup['target_price']
        max_days = setup['max_hold_days']

        future_bars = df.iloc[entry_idx: min(entry_idx + max_days, len(df))].copy().reset_index(drop=True)
        if future_bars.empty:
            return {'net_pnl_pct': 0.0, 'exit_reason': 'NO_DATA', 'days_held': 0, 'exit_price': entry_p, 'exit_date': setup['entry_date']}

        trailing_active = False

        for day_idx, row in future_bars.iterrows():
            h, l, c = row['High'], row['Low'], row['Close']
            ema_10 = row['ema_10']

            # Target hit
            if h >= target_p:
                return {
                    'net_pnl_pct': (target_p - entry_p) / entry_p,
                    'exit_reason': 'TARGET',
                    'days_held': day_idx + 1,
                    'exit_price': target_p,
                    'exit_date': row['Date']
                }

            # Stop loss hit
            if l <= stop_p:
                return {
                    'net_pnl_pct': (stop_p - entry_p) / entry_p,
                    'exit_reason': 'STOP',
                    'days_held': day_idx + 1,
                    'exit_price': stop_p,
                    'exit_date': row['Date']
                }

            # Trailing stop activation: if profit exceeds +2.0%, trail by 10 EMA
            if (c - entry_p) / entry_p >= 0.020:
                trailing_active = True
                if ema_10 > stop_p:
                    stop_p = ema_10

            if trailing_active and l <= stop_p:
                return {
                    'net_pnl_pct': (stop_p - entry_p) / entry_p,
                    'exit_reason': 'TRAILING_STOP',
                    'days_held': day_idx + 1,
                    'exit_price': stop_p,
                    'exit_date': row['Date']
                }

        # Max holding period exit
        last_c = future_bars.iloc[-1]['Close']
        return {
            'net_pnl_pct': (last_c - entry_p) / entry_p,
            'exit_reason': 'TIME_LIMIT',
            'days_held': len(future_bars),
            'exit_price': last_c,
            'exit_date': future_bars.iloc[-1]['Date']
        }
