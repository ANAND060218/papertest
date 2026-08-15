"""
V9.5 Phase 1 -- Causal Multi-Timeframe (MTF) Engine
Computes 15-minute and 60-minute Higher-Timeframe trend indicators strictly causally.

ZERO-LOOKAHEAD GUARANTEE:
At any 5-minute bar t (e.g. 09:35 IST):
  - The 15-minute indicators are derived strictly from the last COMPLETED 15m candle (09:15-09:30).
  - The 60-minute indicators are derived strictly from the last COMPLETED 60m candle.
  - The uncompleted 15m candle (09:30-09:45) is NEVER used for higher-timeframe metrics.
"""
import pandas as pd
import numpy as np


class MultiTimeframeEngine:
    """
    Causal Multi-Timeframe Aggregator & Indicator Engine.
    """

    @staticmethod
    def build_mtf_features(df_5m):
        """
        Enriches a 5-minute DataFrame with strictly causal 15m and 60m indicators.

        Args:
            df_5m: DataFrame with Date, Open, High, Low, Close, Volume sorted by Date.

        Returns:
            DataFrame with additional columns:
            - is_15m_bullish : bool (15m EMA_20 > EMA_50)
            - is_15m_bearish : bool (15m EMA_20 < EMA_50)
            - is_60m_bullish : bool (60m EMA_20 > EMA_50)
            - is_60m_bearish : bool (60m EMA_20 < EMA_50)
            - trend_15m : 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            - trend_60m : 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            - ema_20_15m, ema_50_15m, ema_20_60m, ema_50_60m
        """
        df = df_5m.copy().sort_values('Date').reset_index(drop=True)

        # Set Date as index for clean resampling
        df_idx = df.set_index('Date')

        # -------------------------------------------------------------
        # 1. Causal 15-Minute Resampling & Indicator Computation
        # -------------------------------------------------------------
        # Resample with closed='right', label='right' so 09:15-09:30 is labeled 09:30
        df_15m = df_idx.resample('15min', closed='right', label='right').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()

        df_15m['ema_20_15m'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['ema_50_15m'] = df_15m['Close'].ewm(span=50, adjust=False).mean()
        df_15m['trend_15m'] = np.where(
            df_15m['ema_20_15m'] > df_15m['ema_50_15m'], 'BULLISH',
            np.where(df_15m['ema_20_15m'] < df_15m['ema_50_15m'], 'BEARISH', 'NEUTRAL')
        )

        # -------------------------------------------------------------
        # 2. Causal 60-Minute Resampling & Indicator Computation
        # -------------------------------------------------------------
        df_60m = df_idx.resample('60min', closed='right', label='right').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()

        df_60m['ema_20_60m'] = df_60m['Close'].ewm(span=20, adjust=False).mean()
        df_60m['ema_50_60m'] = df_60m['Close'].ewm(span=50, adjust=False).mean()
        df_60m['trend_60m'] = np.where(
            df_60m['ema_20_60m'] > df_60m['ema_50_60m'], 'BULLISH',
            np.where(df_60m['ema_20_60m'] < df_60m['ema_50_60m'], 'BEARISH', 'NEUTRAL')
        )

        # -------------------------------------------------------------
        # 3. Causal Backward Merge (merge_asof backward)
        # -------------------------------------------------------------
        # merge_asof matches each 5m bar with the latest completed 15m / 60m bar (timestamp <= 5m Date)
        cols_15m = ['Date', 'ema_20_15m', 'ema_50_15m', 'trend_15m']
        cols_60m = ['Date', 'ema_20_60m', 'ema_50_60m', 'trend_60m']

        df = pd.merge_asof(
            df,
            df_15m[cols_15m],
            on='Date',
            direction='backward'
        )

        df = pd.merge_asof(
            df,
            df_60m[cols_60m],
            on='Date',
            direction='backward'
        )

        df['trend_15m'] = df['trend_15m'].fillna('NEUTRAL')
        df['trend_60m'] = df['trend_60m'].fillna('NEUTRAL')

        df['is_15m_bullish'] = df['trend_15m'] == 'BULLISH'
        df['is_15m_bearish'] = df['trend_15m'] == 'BEARISH'
        df['is_60m_bullish'] = df['trend_60m'] == 'BULLISH'
        df['is_60m_bearish'] = df['trend_60m'] == 'BEARISH'

        return df
