"""
V9 Phase 1 -- Algorithmic Candlestick Pattern Detector
Deterministic, mathematical pattern recognition for 5-minute intraday candles.

Supported Patterns:
  1. Hammer (Bullish Pin Bar)
  2. Shooting Star (Bearish Pin Bar)
  3. Bullish Engulfing
  4. Bearish Engulfing
  5. Doji / Indecision
  6. Inside Bar
"""
import pandas as pd
import numpy as np


class CandlestickPatternDetector:
    """
    Algorithmic Candlestick Pattern Detector.
    Evaluates exact mathematical ratios for candle bodies, wicks, and transitions.
    """

    @staticmethod
    def detect_patterns(df):
        """
        Adds boolean and categorical pattern columns to OHLCV DataFrame.

        Required Columns in df: Open, High, Low, Close, Volume.
        """
        df = df.copy()

        # Candle anatomy
        df['body'] = (df['Close'] - df['Open']).abs()
        df['candle_range'] = df['High'] - df['Low']
        # Prevent division by zero
        safe_range = df['candle_range'].replace(0, 0.0001)

        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['is_green'] = df['Close'] > df['Open']
        df['is_red'] = df['Close'] < df['Open']

        # 1. Hammer / Bullish Pin Bar
        # Lower wick >= 2 * body, upper wick <= 15% of range, close in top 35% of candle
        df['is_hammer'] = (
            (df['lower_wick'] >= 2.0 * df['body']) &
            (df['upper_wick'] <= 0.15 * safe_range) &
            (df['Close'] >= df['Low'] + 0.65 * safe_range) &
            (df['candle_range'] > 0)
        )

        # 2. Shooting Star / Bearish Pin Bar
        # Upper wick >= 2 * body, lower wick <= 15% of range, close in bottom 35% of candle
        df['is_shooting_star'] = (
            (df['upper_wick'] >= 2.0 * df['body']) &
            (df['lower_wick'] <= 0.15 * safe_range) &
            (df['Close'] <= df['Low'] + 0.35 * safe_range) &
            (df['candle_range'] > 0)
        )

        # 3. Bullish Engulfing (Current Green engulfs Previous Red)
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        prev_body = df['body'].shift(1)
        prev_is_red = df['is_red'].shift(1)

        df['is_bullish_engulfing'] = (
            (prev_is_red == True) &
            (df['is_green'] == True) &
            (df['Open'] <= prev_close + 0.05 * df['body']) &
            (df['Close'] >= prev_open - 0.05 * df['body']) &
            (df['body'] > prev_body)
        )

        # 4. Bearish Engulfing (Current Red engulfs Previous Green)
        prev_is_green = df['is_green'].shift(1)
        df['is_bearish_engulfing'] = (
            (prev_is_green == True) &
            (df['is_red'] == True) &
            (df['Open'] >= prev_close - 0.05 * df['body']) &
            (df['Close'] <= prev_open + 0.05 * df['body']) &
            (df['body'] > prev_body)
        )

        # 5. Doji (Body <= 10% of range)
        df['is_doji'] = (
            (df['body'] <= 0.10 * safe_range) &
            (df['candle_range'] > 0)
        )

        # 6. Inside Bar (Current candle completely inside prior candle's high/low)
        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        df['is_inside_bar'] = (
            (df['High'] < prev_high) &
            (df['Low'] > prev_low)
        )

        # Composite Pattern String
        def get_primary_pattern(row):
            if row['is_hammer']:
                return 'HAMMER'
            if row['is_shooting_star']:
                return 'SHOOTING_STAR'
            if row['is_bullish_engulfing']:
                return 'BULLISH_ENGULFING'
            if row['is_bearish_engulfing']:
                return 'BEARISH_ENGULFING'
            if row['is_inside_bar']:
                return 'INSIDE_BAR'
            if row['is_doji']:
                return 'DOJI'
            return 'NONE'

        df['pattern_name'] = df.apply(get_primary_pattern, axis=1)

        return df
