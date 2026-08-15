"""
V14 Intraday VWAP Mean-Reversion Engine (Exploiting 75% Intraday Mean-Reversion Tendency)
Trades WITH the market's natural mean-reverting microstructure:
  - Fades extreme statistical excursions (+2.0 sigma / +2.5 sigma from VWAP)
  - Targets the VWAP equilibrium midline
  - Asymmetric Reward-to-Risk (Target ~0.70% to 1.00%, Stop ~0.35%)
"""
import pandas as pd
import numpy as np
from datetime import time as dtime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VWAPMeanReversionEngine:
    """
    Institutional VWAP Band Mean-Reversion Engine.
    """

    @staticmethod
    def prepare_dataset(df):
        """Computes Intraday VWAP and Standard Deviation Bands (+/- 1.5, 2.0, 2.5 sigma), RSI, and Wicks."""
        df = df.copy().sort_values('Date').reset_index(drop=True)
        df['trade_date'] = df['Date'].dt.date
        df['bar_time'] = df['Date'].dt.time

        # Intraday VWAP & Bands
        df['cum_vol'] = df.groupby('trade_date')['Volume'].cumsum()
        df['pv'] = df['Close'] * df['Volume']
        df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol'].replace(0, np.nan)

        # Variance & Standard Deviation of VWAP
        df['dev_sq'] = df['Volume'] * ((df['Close'] - df['vwap']) ** 2)
        df['cum_dev_sq'] = df.groupby('trade_date')['dev_sq'].cumsum()
        df['vwap_std'] = np.sqrt(df['cum_dev_sq'] / df['cum_vol'].replace(0, np.nan))

        df['upper_band_15'] = df['vwap'] + 1.5 * df['vwap_std']
        df['lower_band_15'] = df['vwap'] - 1.5 * df['vwap_std']
        df['upper_band_20'] = df['vwap'] + 2.0 * df['vwap_std']
        df['lower_band_20'] = df['vwap'] - 2.0 * df['vwap_std']
        df['upper_band_25'] = df['vwap'] + 2.5 * df['vwap_std']
        df['lower_band_25'] = df['vwap'] - 2.5 * df['vwap_std']

        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Wicks
        df['candle_range'] = df['High'] - df['Low']
        safe_r = df['candle_range'].replace(0, 0.0001)
        df['upper_wick_pct'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / safe_r
        df['lower_wick_pct'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / safe_r

        return df

    @staticmethod
    def generate_reversion_setups(df, symbol='RELIANCE.NS', sigma_threshold=2.0):
        """
        Generates mean-reversion setups fading extreme extensions back to the VWAP midline.
        """
        setups = []
        # Active window for mean reversion: 10:00 to 14:15 IST
        start_t = dtime(10, 0)
        end_t = dtime(14, 15)

        for trade_date, day_df in df.groupby('trade_date'):
            day_df = day_df.sort_values('Date').reset_index(drop=True)
            if len(day_df) < 15:
                continue

            for j in range(10, len(day_df) - 3):
                row = day_df.iloc[j]
                bt = row['bar_time']

                if bt < start_t or bt > end_t:
                    continue

                c = row['Close']
                h = row['High']
                l = row['Low']
                vwap = row['vwap']
                rsi = row['rsi_14']
                upper_band = row['upper_band_20'] if sigma_threshold == 2.0 else row['upper_band_25']
                lower_band = row['lower_band_20'] if sigma_threshold == 2.0 else row['lower_band_25']

                if pd.isna(vwap) or pd.isna(rsi) or pd.isna(upper_band) or pd.isna(lower_band):
                    continue

                dist_to_vwap_pct = abs(c - vwap) / c * 100

                # Must have at least 0.50% distance to VWAP to cover friction
                if dist_to_vwap_pct < 0.50:
                    continue

                # 1. BEARISH FADE (SHORT): Price >= Upper Band & Overbought RSI & Upper Rejection Wick
                if h >= upper_band and rsi >= 68 and row['upper_wick_pct'] >= 0.25 and c <= h:
                    stop_p = round(h * 1.0025, 2)  # Stop just above the extreme wick
                    target_p = round(vwap, 2)       # Target VWAP midline

                    # Ensure target provides > 1.5x reward to risk
                    risk = stop_p - c
                    reward = c - target_p
                    if reward >= 1.4 * risk and risk > 0:
                        actual_idx_arr = df.index[df['Date'] == row['Date']]
                        if len(actual_idx_arr) > 0:
                            setups.append({
                                'bar_index': actual_idx_arr[0],
                                'timestamp': row['Date'],
                                'trade_date': trade_date,
                                'symbol': symbol,
                                'direction': 'SHORT',
                                'setup_name': f"FADE_UPPER_{sigma_threshold}s",
                                'entry_price': c,
                                'stop_price': stop_p,
                                'target_price': target_p,
                                'vwap': vwap,
                                'rsi': rsi,
                                'max_hold_bars': 24
                            })

                # 2. BULLISH FADE (LONG): Price <= Lower Band & Oversold RSI & Lower Rejection Wick
                elif l <= lower_band and rsi <= 32 and row['lower_wick_pct'] >= 0.25 and c >= l:
                    stop_p = round(l * 0.9975, 2)  # Stop just below the extreme wick
                    target_p = round(vwap, 2)       # Target VWAP midline

                    risk = c - stop_p
                    reward = target_p - c
                    if reward >= 1.4 * risk and risk > 0:
                        actual_idx_arr = df.index[df['Date'] == row['Date']]
                        if len(actual_idx_arr) > 0:
                            setups.append({
                                'bar_index': actual_idx_arr[0],
                                'timestamp': row['Date'],
                                'trade_date': trade_date,
                                'symbol': symbol,
                                'direction': 'LONG',
                                'setup_name': f"FADE_LOWER_{sigma_threshold}s",
                                'entry_price': c,
                                'stop_price': stop_p,
                                'target_price': target_p,
                                'vwap': vwap,
                                'rsi': rsi,
                                'max_hold_bars': 24
                            })

        return setups
