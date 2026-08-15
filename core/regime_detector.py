"""
V6 -- Market Regime Detector
Classifies market conditions into distinct regimes:
  1. TRENDING_BULL  : Strong upward trend (ADX > 25, Price > EMA20 > EMA50)
  2. TRENDING_BEAR  : Strong downward trend (ADX > 25, Price < EMA20 < EMA50)
  3. SIDEWAYS       : Range-bound consolidation (ADX < 20, BB Width low)
  4. HIGH_VOL_CHOP  : High volatility erratic moves (ATR spike > 1.5x, wide swings)

Principle (from roadmap):
  Compare strategy performance WITH regime filter vs WITHOUT regime filter.
  Only keep regime filter if it measurably improves Profit Factor / Sharpe.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RegimeDetector:
    """
    Market Regime Classifier for Single Stocks & Benchmark Indices.
    """

    def __init__(self, adx_trend_thresh=25, adx_range_thresh=20, atr_spike_mult=1.4):
        self.adx_trend_thresh = adx_trend_thresh
        self.adx_range_thresh = adx_range_thresh
        self.atr_spike_mult = atr_spike_mult

    def classify_regimes(self, df):
        """
        Classifies each bar/day into a market regime.
        Requires OHLCV DataFrame.
        """
        df = df.copy()

        close = df['Close']
        high = df['High']
        low = df['Low']

        # 1. EMAs
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        # 2. ATR & normalized ATR spike
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=10).mean()
        atr_baseline = atr_14.rolling(50, min_periods=20).mean()
        atr_ratio = atr_14 / atr_baseline.replace(0, np.nan)

        # 3. ADX
        adx = self._compute_adx(df, period=14)

        # 4. Bollinger Band Width
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_width = (4 * std_20) / ma_20

        regimes = []
        for i in range(len(df)):
            if i < 50:
                regimes.append('UNKNOWN')
                continue

            c = close.iloc[i]
            e20 = ema_20.iloc[i]
            e50 = ema_50.iloc[i]
            cur_adx = adx.iloc[i]
            cur_atr_rat = atr_ratio.iloc[i]

            # High volatility condition
            if pd.notna(cur_atr_rat) and cur_atr_rat >= self.atr_spike_mult:
                regimes.append('HIGH_VOL_CHOP')
            # Trending Bull
            elif cur_adx >= self.adx_trend_thresh and c > e20 and e20 > e50:
                regimes.append('TRENDING_BULL')
            # Trending Bear
            elif cur_adx >= self.adx_trend_thresh and c < e20 and e20 < e50:
                regimes.append('TRENDING_BEAR')
            # Sideways / Range
            elif cur_adx <= self.adx_range_thresh:
                regimes.append('SIDEWAYS')
            else:
                # Moderate/Transitional
                if c >= e20:
                    regimes.append('WEAK_BULL')
                else:
                    regimes.append('WEAK_BEAR')

        df['regime'] = regimes
        df['adx_14'] = adx
        df['atr_ratio'] = atr_ratio
        df['bb_width'] = bb_width

        return df

    def evaluate_regime_impact(self, trades_df, regime_col='regime', pnl_col='pnl_pct'):
        """
        Calculates performance breakdown per regime to test if filtering helps.
        """
        if regime_col not in trades_df.columns:
            return None

        results = []
        for reg, group in trades_df.groupby(regime_col):
            trades = len(group)
            wins = (group['label'] == 1).sum() if 'label' in group.columns else (group[pnl_col] > 0).sum()
            win_rate = (wins / trades) * 100 if trades > 0 else 0
            total_pnl = group[pnl_col].sum()
            avg_pnl = group[pnl_col].mean()

            gross_win = group[group[pnl_col] > 0][pnl_col].sum()
            gross_loss = abs(group[group[pnl_col] <= 0][pnl_col].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

            results.append({
                'Regime': reg,
                'Trades': trades,
                'Wins': int(wins),
                'WinRate%': round(win_rate, 2),
                'ProfitFactor': round(pf, 3),
                'TotalPnL%': round(total_pnl, 3),
                'AvgPnL%': round(avg_pnl, 4)
            })

        return pd.DataFrame(results).sort_values('ProfitFactor', ascending=False)

    @staticmethod
    def _compute_adx(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx
