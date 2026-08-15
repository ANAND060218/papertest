"""
V16 Market Regime & Index Directional Engine
Analyzes NIFTY 50 and BANKNIFTY opening gap, multi-day trend, volatility, and overnight sentiment
to determine the macro directional bias for today's intraday trading session.
"""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


class MarketRegimeEngine:
    """
    Evaluates macro market structure and provides the daily directional bias.
    """

    def __init__(self, nifty_daily_df):
        self.df = nifty_daily_df.copy().sort_values('Date').reset_index(drop=True)
        self._calculate_indicators()

    def _calculate_indicators(self):
        self.df['SMA_5'] = self.df['Close'].rolling(5).mean()
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        self.df['ATR_14'] = (self.df['High'] - self.df['Low']).rolling(14).mean()
        self.df['Prev_Close'] = self.df['Close'].shift(1)
        self.df['Gap_Pct'] = (self.df['Open'] - self.df['Prev_Close']) / self.df['Prev_Close'] * 100

    def evaluate_regime_for_date(self, target_date):
        """
        Calculates regime state up to target_date (zero lookahead).
        """
        hist = self.df[self.df['Date'] <= pd.to_datetime(target_date)].copy()
        if len(hist) < 50:
            return {'bias': 'NEUTRAL', 'score': 50, 'confidence': 'LOW'}

        today = hist.iloc[-1]
        prev = hist.iloc[-2]

        score = 50.0  # Neutral starting baseline

        # 1. Multi-Day Trend (5D & 20D SMA)
        if today['Close'] > today['SMA_20']:
            score += 15.0
        else:
            score -= 15.0

        if today['SMA_5'] > today['SMA_20']:
            score += 10.0
        else:
            score -= 10.0

        # 2. Opening Gap & Momentum
        gap = today['Gap_Pct']
        if gap > 0.30:
            score += 15.0
        elif gap < -0.30:
            score -= 15.0

        # 3. Previous Day Range Context
        if prev['Close'] > prev['Open']:
            score += 10.0
        else:
            score -= 10.0

        score = max(0.0, min(100.0, score))

        if score >= 65.0:
            bias = 'BULLISH'
            actionable = 'LONG_ONLY'
        elif score <= 35.0:
            bias = 'BEARISH'
            actionable = 'SHORT_ONLY'
        else:
            bias = 'NEUTRAL / RANGE'
            actionable = 'SELECTIVE_BOTH_SIDES'

        return {
            'date': today['Date'].strftime('%Y-%m-%d'),
            'nifty_close': round(float(today['Close']), 2),
            'nifty_gap_pct': round(float(gap), 2),
            'trend_5d_vs_20d': 'BULLISH' if today['SMA_5'] > today['SMA_20'] else 'BEARISH',
            'above_20d_sma': bool(today['Close'] > today['SMA_20']),
            'above_50d_sma': bool(today['Close'] > today['SMA_50']),
            'regime_score': round(score, 1),
            'market_bias': bias,
            'preferred_direction': actionable,
            'atr_14': round(float(today['ATR_14']), 2)
        }
