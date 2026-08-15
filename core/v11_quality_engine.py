"""
V11 -- Breakout Quality, Retest Confirmation & Microstructure Analytics Engine
Evaluates structural breakout mechanics, candle quality, expansion distance, relative strength vs NIFTY,
breakout + retest confirmation, and tracks exact Maximum Adverse/Favorable Excursions (MAE/MFE).
"""
import pandas as pd
import numpy as np
from datetime import time as dtime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class V11QualityEngine:
    """
    Advanced Breakout Quality & Microstructure Analytics Engine.
    """

    def __init__(self):
        self.entry_start = dtime(9, 30)
        self.entry_end = dtime(10, 45)

    def prepare_dataset(self, df, nifty_df=None):
        """Enriches 5m dataset with VWAP, volume SMA, ATR, and NIFTY relative strength."""
        df = df.copy().sort_values('Date').reset_index(drop=True)
        df['trade_date'] = df['Date'].dt.date
        df['bar_time'] = df['Date'].dt.time

        # Intraday VWAP
        df['cum_vol'] = df.groupby('trade_date')['Volume'].cumsum()
        df['pv'] = df['Close'] * df['Volume']
        df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol'].replace(0, np.nan)

        # Volume Ratio
        df['vol_sma_20'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma_20'].replace(0, np.nan)

        # Candle anatomy
        df['candle_range'] = df['High'] - df['Low']
        df['body'] = (df['Close'] - df['Open']).abs()
        safe_range = df['candle_range'].replace(0, 0.0001)
        df['body_pct'] = df['body'] / safe_range
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['upper_wick_pct'] = df['upper_wick'] / safe_range
        df['lower_wick_pct'] = df['lower_wick'] / safe_range

        # Candle Quality Flag
        # Strong Bullish Candle: Green, Body >= 55% range, Upper Wick <= 20% range
        df['is_strong_bull_candle'] = (
            (df['Close'] > df['Open']) &
            (df['body_pct'] >= 0.55) &
            (df['upper_wick_pct'] <= 0.20)
        )
        # Strong Bearish Candle: Red, Body >= 55% range, Lower Wick <= 20% range
        df['is_strong_bear_candle'] = (
            (df['Close'] < df['Open']) &
            (df['body_pct'] >= 0.55) &
            (df['lower_wick_pct'] <= 0.20)
        )

        # Relative Strength vs NIFTY
        df['nifty_ret_from_open'] = 0.0
        df['stock_ret_from_open'] = df.groupby('trade_date').apply(
            lambda g: (g['Close'] - g.iloc[0]['Open']) / g.iloc[0]['Open'] * 100
        ).reset_index(level=0, drop=True)

        if nifty_df is not None and not nifty_df.empty:
            nifty_clean = nifty_df.copy().sort_values('Date').reset_index(drop=True)
            nifty_clean['trade_date'] = nifty_clean['Date'].dt.date
            nifty_clean['nifty_ret'] = nifty_clean.groupby('trade_date').apply(
                lambda g: (g['Close'] - g.iloc[0]['Open']) / g.iloc[0]['Open'] * 100
            ).reset_index(level=0, drop=True)

            nifty_map = dict(zip(nifty_clean['Date'], nifty_clean['nifty_ret']))
            df['nifty_ret_from_open'] = df['Date'].map(nifty_map).fillna(0.0)

        df['relative_strength_vs_nifty'] = df['stock_ret_from_open'] - df['nifty_ret_from_open']

        return df

    def generate_v11_setups(self, df, variant='V11_A', symbol='RELIANCE.NS'):
        """
        Generates ORB setups evaluating breakout distance, candle quality, relative strength,
        and immediate vs retest execution modes.
        """
        setups = []
        orb_bars_count = 3  # 09:15, 09:20, 09:25

        for trade_date, day_df in df.groupby('trade_date'):
            day_df = day_df.sort_values('Date').reset_index(drop=True)
            if len(day_df) < orb_bars_count + 4:
                continue

            orb_slice = day_df.iloc[:orb_bars_count]
            orb_high = orb_slice['High'].max()
            orb_low = orb_slice['Low'].min()
            orb_range = orb_high - orb_low
            day_open = orb_slice.iloc[0]['Open']

            if orb_range <= 0 or day_open <= 0:
                continue

            or_range_pct = (orb_range / day_open) * 100

            # V11-D & above: Opening Range Width Filter (0.35% to 1.80%)
            if variant in ['V11_D', 'V11_E', 'V11_F', 'V11_G']:
                if or_range_pct < 0.35 or or_range_pct > 1.80:
                    continue

            triggered_today = False

            # Scan for initial breakout bar
            for j in range(orb_bars_count, len(day_df)):
                if triggered_today:
                    break

                row = day_df.iloc[j]
                bt = row['bar_time']

                # Entry window: 09:30 to 10:15 (avoiding late traps after 10:15)
                if bt < self.entry_start or bt > dtime(10, 15):
                    continue

                cur_close = row['Close']
                vr = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
                vwap = row['vwap'] if pd.notna(row['vwap']) else cur_close
                rs = row['relative_strength_vs_nifty']
                nifty_ret = row['nifty_ret_from_open']

                is_breakout_long = cur_close > orb_high
                is_breakout_short = cur_close < orb_low

                if not (is_breakout_long or is_breakout_short):
                    continue

                direction = 'LONG' if is_breakout_long else 'SHORT'

                # Volume & VWAP filter (Standard across all V11)
                if vr < 1.20:
                    continue
                if direction == 'LONG' and cur_close < vwap:
                    continue
                if direction == 'SHORT' and cur_close > vwap:
                    continue

                # NIFTY Alignment
                if direction == 'LONG' and nifty_ret < 0:
                    continue
                if direction == 'SHORT' and nifty_ret > 0:
                    continue

                # Breakout Distance & Quality Metrics
                if direction == 'LONG':
                    breakout_dist = (cur_close - orb_high) / orb_range
                    is_strong_candle = row['is_strong_bull_candle']
                else:
                    breakout_dist = (orb_low - cur_close) / orb_range
                    is_strong_candle = row['is_strong_bear_candle']

                # V11-B & above: Breakout Candle Quality Filter
                if variant in ['V11_B', 'V11_C', 'V11_D', 'V11_E', 'V11_F', 'V11_G']:
                    if not is_strong_candle:
                        continue

                # V11-C & above: Breakout Distance Filter (Controlled 5% to 40% expansion)
                if variant in ['V11_C', 'V11_D', 'V11_E', 'V11_F', 'V11_G']:
                    if breakout_dist < 0.05 or breakout_dist > 0.40:
                        continue

                # V11-E & above: Relative Strength Filter (Stock outperforming NIFTY by >= 0.25%)
                if variant in ['V11_E', 'V11_F', 'V11_G']:
                    if direction == 'LONG' and rs < 0.25:
                        continue
                    elif direction == 'SHORT' and rs > -0.25:
                        continue

                # -----------------------------------------------------
                # RETEST CONFIRMATION LOGIC (V11-F & V11-G)
                # -----------------------------------------------------
                if variant in ['V11_F', 'V11_G']:
                    # Look forward 1-4 bars for pullback to OR level + rejection
                    retest_found = False
                    entry_bar_idx = None
                    entry_price = None

                    for k in range(j + 1, min(j + 5, len(day_df))):
                        retest_bar = day_df.iloc[k]
                        r_close = retest_bar['Close']
                        r_low = retest_bar['Low']
                        r_high = retest_bar['High']

                        if direction == 'LONG':
                            # Price pulls back to within 0.2% of OR_High without closing below OR_Low
                            if r_low <= orb_high * 1.002 and r_close >= orb_high * 0.998:
                                # Rejection confirmation: green candle or lower wick rejection
                                if retest_bar['Close'] >= retest_bar['Open'] or retest_bar['lower_wick_pct'] >= 0.35:
                                    retest_found = True
                                    entry_bar_idx = k
                                    entry_price = r_close
                                    break
                        else: # SHORT
                            if r_high >= orb_low * 0.998 and r_close <= orb_low * 1.002:
                                if retest_bar['Close'] <= retest_bar['Open'] or retest_bar['upper_wick_pct'] >= 0.35:
                                    retest_found = True
                                    entry_bar_idx = k
                                    entry_price = r_close
                                    break

                    if not retest_found:
                        continue

                    target_bar_row = day_df.iloc[entry_bar_idx]
                    actual_idx_arr = df.index[df['Date'] == target_bar_row['Date']]
                    if len(actual_idx_arr) == 0:
                        continue
                    actual_idx = actual_idx_arr[0]
                    entry_p = entry_price
                    setup_type_tag = "ORB_RETEST"
                else:
                    # Immediate Breakout Entry
                    actual_idx_arr = df.index[df['Date'] == row['Date']]
                    if len(actual_idx_arr) == 0:
                        continue
                    actual_idx = actual_idx_arr[0]
                    entry_p = cur_close
                    setup_type_tag = "ORB_IMMEDIATE"

                # Targets & Stops
                if direction == 'LONG':
                    stop_p = round(max(orb_low, entry_p * 0.996), 2)
                    target_p = round(entry_p * 1.010, 2) # Target 1.0%
                else:
                    stop_p = round(min(orb_high, entry_p * 1.004), 2)
                    target_p = round(entry_p * 0.990, 2)

                setups.append({
                    'bar_index': actual_idx,
                    'timestamp': df.iloc[actual_idx]['Date'],
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'direction': direction,
                    'setup_name': f"{setup_type_tag}_{direction}",
                    'entry_price': entry_p,
                    'stop_price': stop_p,
                    'target_price': target_p,
                    'orb_high': orb_high,
                    'orb_low': orb_low,
                    'or_range_pct': or_range_pct,
                    'breakout_distance': breakout_dist,
                    'is_strong_candle': is_strong_candle,
                    'relative_strength': rs,
                    'max_hold_bars': 30
                })

                triggered_today = True

        return setups

    @staticmethod
    def calculate_mae_mfe(trade, intraday_df):
        """
        Calculates Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        over the holding trajectory of a trade.
        """
        entry_idx = trade['bar_index']
        entry_p = trade['entry_price']
        direction = trade.get('direction', 'LONG')
        max_bars = trade.get('max_hold_bars', 30)
        trade_date = trade['trade_date']

        start_idx = entry_idx + 1
        end_idx = min(start_idx + max_bars, len(intraday_df))

        future_bars = intraday_df.iloc[start_idx:end_idx].copy()
        if 'trade_date' not in future_bars.columns:
            future_bars['trade_date'] = future_bars['Date'].dt.date
        future_bars = future_bars[future_bars['trade_date'] == trade_date]

        if future_bars.empty:
            return {'mae_pct': 0.0, 'mfe_pct': 0.0}

        highs = future_bars['High'].values
        lows = future_bars['Low'].values

        if direction == 'LONG':
            mfe_pct = (max(highs) - entry_p) / entry_p * 100
            mae_pct = (min(lows) - entry_p) / entry_p * 100 # negative value
        else:
            mfe_pct = (entry_p - min(lows)) / entry_p * 100
            mae_pct = (entry_p - max(highs)) / entry_p * 100 # negative value

        return {
            'mae_pct': round(float(mae_pct), 3),
            'mfe_pct': round(float(mfe_pct), 3)
        }
