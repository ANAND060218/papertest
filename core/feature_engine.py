"""
V4 -- Compact Feature Engine
Builds ~18 features available AT ENTRY TIME ONLY (no future data leakage).
Used by XGBoost to filter which setups to take.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureEngine:
    """
    Compact, non-redundant feature builder for ML models.
    All features are computed from data available BEFORE the entry bar.
    """

    def build_features(self, df):
        """
        Compute all features on the full DataFrame.
        Returns df with feature columns added.
        Call BEFORE detecting setups, then extract features at setup indices.
        """
        df = df.copy()

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # ---- PRICE FEATURES (4) ----
        df['feat_return_1'] = close.pct_change(1)
        df['feat_return_5'] = close.pct_change(5)
        df['feat_bar_range'] = (high - low) / close
        df['feat_gap'] = (df['Open'] - close.shift(1)) / close.shift(1)

        # ---- TREND FEATURES (4) ----
        df['feat_ema_10'] = close.ewm(span=10, adjust=False).mean()
        df['feat_ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['feat_ema_50'] = close.ewm(span=50, adjust=False).mean()
        # Price relative to EMAs (normalized)
        df['feat_price_vs_ema20'] = (close - df['feat_ema_20']) / df['feat_ema_20']
        df['feat_price_vs_ema50'] = (close - df['feat_ema_50']) / df['feat_ema_50']

        # ADX (14-period, trend strength)
        df['feat_adx'] = self._compute_adx(df, period=14)

        # ---- MOMENTUM FEATURES (2) ----
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=10).mean()
        rs = gain / loss.replace(0, np.nan)
        df['feat_rsi_14'] = 100 - (100 / (1 + rs))

        # MACD histogram
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['feat_macd_hist'] = macd - signal

        # ---- VOLATILITY FEATURES (2) ----
        # ATR (normalized by price)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=10).mean()
        df['feat_atr_pct'] = atr / close

        # Bollinger Band width
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = ma_20 + 2 * std_20
        bb_lower = ma_20 - 2 * std_20
        df['feat_bb_width'] = (bb_upper - bb_lower) / ma_20

        # ---- VOLUME FEATURES (2) ----
        vol_avg_20 = volume.rolling(20, min_periods=10).mean()
        df['feat_volume_ratio'] = volume / vol_avg_20.replace(0, np.nan)
        df['feat_volume_trend'] = volume.rolling(5).mean() / vol_avg_20.replace(0, np.nan)

        # ---- STRUCTURE FEATURES (2) ----
        # Distance from 20-day high and low (normalized by ATR)
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        df['feat_dist_from_high'] = (close - high_20) / atr
        df['feat_dist_from_low'] = (close - low_20) / atr

        # ---- PATTERN FEATURES (2) ----
        # Inside bar
        df['feat_is_inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        # Consecutive green/red days
        is_green = (close > df['Open']).astype(int)
        df['feat_consec_green'] = is_green.groupby((is_green != is_green.shift()).cumsum()).cumsum()

        return df

    def get_feature_columns(self):
        """Return the list of feature column names."""
        return [
            'feat_return_1', 'feat_return_5', 'feat_bar_range', 'feat_gap',
            'feat_price_vs_ema20', 'feat_price_vs_ema50', 'feat_adx',
            'feat_rsi_14', 'feat_macd_hist',
            'feat_atr_pct', 'feat_bb_width',
            'feat_volume_ratio', 'feat_volume_trend',
            'feat_dist_from_high', 'feat_dist_from_low',
            'feat_is_inside_bar', 'feat_consec_green',
        ]

    def extract_features_at_indices(self, df, indices):
        """
        Extract feature vectors at specific bar indices.
        Returns a DataFrame with only feature columns.
        """
        feature_cols = self.get_feature_columns()
        available = [c for c in feature_cols if c in df.columns]
        return df.loc[indices, available].copy()

    @staticmethod
    def _compute_adx(df, period=14):
        """Compute ADX (Average Directional Index)."""
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
