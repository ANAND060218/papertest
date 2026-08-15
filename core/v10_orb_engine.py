"""
V10 -- Opening Range Expansion (ORB) Specialized Engine
Specialized engine for 15-minute Opening Range Breakout (09:15 - 10:45 IST).

Rules & Definitions:
  1. Opening Range: Bars 09:15, 09:20, 09:25 define OR_High and OR_Low.
  2. Active Entry Window: 09:30 to 10:45 IST (Strictly 1 trade per symbol/day on first break).
  3. Direction:
     - LONG: Close > OR_High
     - SHORT: Close < OR_Low
  4. Time-of-Day Sub-Windows:
     - '09:30-09:45', '09:45-10:00', '10:00-10:15', '10:15-10:30', '10:30-10:45', '10:45+'
"""
import pandas as pd
import numpy as np
from datetime import time as dtime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from risk.dynamic_risk import DynamicRiskEngine


class V10ORBEngine:
    """
    Dedicated Institutional Opening Range Breakout (ORB) Engine.
    """

    def __init__(self):
        self.entry_start = dtime(9, 30)
        self.entry_end = dtime(10, 45)

    def prepare_dataset(self, df):
        """Precomputes VWAP, rolling 20-bar volume SMA, ATR, and time-of-day tags."""
        df = df.copy().sort_values('Date').reset_index(drop=True)
        df['trade_date'] = df['Date'].dt.date
        df['bar_time'] = df['Date'].dt.time

        # VWAP
        df['cum_vol'] = df.groupby('trade_date')['Volume'].cumsum()
        df['pv'] = df['Close'] * df['Volume']
        df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol'].replace(0, np.nan)

        # Volume Ratio
        df['vol_sma_20'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma_20'].replace(0, np.nan)

        # ATR 14
        df['atr_14'] = DynamicRiskEngine.compute_atr(df, period=14)

        return df

    def get_time_subwindow(self, bar_time):
        """Categorizes time of day into 15-minute intervals."""
        if dtime(9, 30) <= bar_time < dtime(9, 45):
            return "09:30-09:45"
        elif dtime(9, 45) <= bar_time < dtime(10, 0):
            return "09:45-10:00"
        elif dtime(10, 0) <= bar_time < dtime(10, 15):
            return "10:00-10:15"
        elif dtime(10, 15) <= bar_time < dtime(10, 30):
            return "10:15-10:30"
        elif dtime(10, 30) <= bar_time <= dtime(10, 45):
            return "10:30-10:45"
        else:
            return "10:45+"

    def generate_orb_setups(self, df, variant='V10_A', symbol='RELIANCE.NS', index_df=None):
        """
        Generates ORB candidate setups across all V10 ablation variants.

        Variants:
          - 'V10_A' : Pure 15m ORB (No filters)
          - 'V10_B' : ORB + Volume Confirmation (Vol Ratio >= 1.25)
          - 'V10_C' : ORB + VWAP Alignment (Long: Close > VWAP, Short: Close < VWAP)
          - 'V10_D' : ORB + Volume + VWAP Confluence
          - 'V10_E' : ORB + Volume + VWAP + XGBoost Gate
          - 'V10_F' : ORB + Volume + VWAP + XGBoost + Market/Index Alignment
          - 'V10_G' : Full Strategy + Dynamic ATR Risk Model
        """
        setups = []
        orb_bars_count = 3  # 09:15, 09:20, 09:25

        # Index trend lookup if provided
        index_trend_map = {}
        if index_df is not None and not index_df.empty:
            for td, g in index_df.groupby(index_df['Date'].dt.date):
                open_p = g.iloc[0]['Open']
                close_p = g.iloc[-1]['Close']
                index_trend_map[td] = 'BULLISH' if close_p >= open_p else 'BEARISH'

        for trade_date, day_df in df.groupby('trade_date'):
            day_df = day_df.sort_values('Date').reset_index(drop=True)
            if len(day_df) < orb_bars_count + 3:
                continue

            # 1. Establish 15-min Opening Range
            orb_slice = day_df.iloc[:orb_bars_count]
            orb_high = orb_slice['High'].max()
            orb_low = orb_slice['Low'].min()
            orb_range = orb_high - orb_low

            if orb_range <= 0:
                continue

            triggered_today = False

            # 2. Scan opening expansion window (09:30 to 10:45)
            for j in range(orb_bars_count, len(day_df)):
                if triggered_today:
                    break

                row = day_df.iloc[j]
                bt = row['bar_time']

                # Restrict entry window strictly to 09:30 - 10:45
                if bt < self.entry_start or bt > self.entry_end:
                    continue

                cur_close = row['Close']
                vr = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
                vwap = row['vwap'] if pd.notna(row['vwap']) else cur_close
                atr_val = row['atr_14'] if pd.notna(row['atr_14']) and row['atr_14'] > 0 else 0.005 * cur_close

                is_breakout_long = cur_close > orb_high
                is_breakout_short = cur_close < orb_low

                if not (is_breakout_long or is_breakout_short):
                    continue

                direction = 'LONG' if is_breakout_long else 'SHORT'
                subwindow = self.get_time_subwindow(bt)

                # =====================================================
                # V10 Ablation Filters
                # =====================================================
                # V10-B/D/E/F/G: Volume Filter
                if variant in ['V10_B', 'V10_D', 'V10_E', 'V10_F', 'V10_G']:
                    if vr < 1.25:
                        continue

                # V10-C/D/E/F/G: VWAP Alignment Filter
                if variant in ['V10_C', 'V10_D', 'V10_E', 'V10_F', 'V10_G']:
                    if direction == 'LONG' and cur_close < vwap:
                        continue
                    elif direction == 'SHORT' and cur_close > vwap:
                        continue

                # V10-F/G: Market Index Trend Alignment
                if variant in ['V10_F', 'V10_G'] and index_trend_map:
                    idx_trend = index_trend_map.get(trade_date, 'NEUTRAL')
                    if direction == 'LONG' and idx_trend == 'BEARISH':
                        continue
                    elif direction == 'SHORT' and idx_trend == 'BULLISH':
                        continue

                # Find exact index in full df
                actual_idx_arr = df.index[df['Date'] == row['Date']]
                if len(actual_idx_arr) == 0:
                    continue
                actual_idx = actual_idx_arr[0]

                # =====================================================
                # Risk Boundaries
                # =====================================================
                if variant == 'V10_G':
                    # Dynamic ATR Sizing
                    if direction == 'LONG':
                        stop_p = round(max(orb_low, cur_close - 0.75 * atr_val), 2)
                        target_p = round(cur_close + 1.50 * atr_val, 2)
                    else:
                        stop_p = round(min(orb_high, cur_close + 0.75 * atr_val), 2)
                        target_p = round(cur_close - 1.50 * atr_val, 2)
                else:
                    # Baseline Fixed ORB Boundaries (+0.8% target, -0.4% stop or OR level)
                    if direction == 'LONG':
                        stop_p = round(max(orb_low, cur_close * 0.996), 2)
                        target_p = round(cur_close * 1.008, 2)
                    else:
                        stop_p = round(min(orb_high, cur_close * 1.004), 2)
                        target_p = round(cur_close * 0.992, 2)

                setups.append({
                    'bar_index': actual_idx,
                    'timestamp': row['Date'],
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'direction': direction,
                    'setup_name': f"ORB_{direction}",
                    'subwindow': subwindow,
                    'entry_price': cur_close,
                    'stop_price': stop_p,
                    'target_price': target_p,
                    'orb_high': orb_high,
                    'orb_low': orb_low,
                    'orb_range_pct': (orb_range / cur_close) * 100,
                    'volume_ratio': vr,
                    'vwap': vwap,
                    'max_hold_bars': 30,
                    'features': {
                        'vol_ratio': vr,
                        'rsi_14': 50.0,
                        'dist_to_vwap': ((cur_close - vwap) / vwap) * 100,
                        'dist_to_pdh': 0.0,
                        'dist_to_pdl': 0.0,
                        'atr_pct': (atr_val / cur_close) * 100,
                        'is_hammer': 0,
                        'trend_bull': int(direction == 'LONG'),
                        'trend_bear': int(direction == 'SHORT'),
                        'hour': row['Date'].hour,
                        'minute': row['Date'].minute
                    }
                })

                triggered_today = True  # Strictly 1 ORB setup per stock per day

        return setups
