"""
V9 Phase 2 -- Causal Market Structure Engine
Calculates fractal swing pivots, market trend (HH/HL vs LH/LL), key reference levels (PDH, PDL, PDC, VWAP bands),
and breakout/retest states strictly without look-ahead bias.

RULE OF ZERO-LOOKAHEAD:
A 5-bar fractal pivot at bar (t-2) is confirmed ONLY when bar (t) closes.
At any bar t, the system only knows about pivots confirmed at or before bar t.
"""
import pandas as pd
import numpy as np
from datetime import time as dtime


class MarketStructureEngine:
    """
    Causal, Zero-Lookahead Market Structure Engine.
    """

    @staticmethod
    def compute_market_structure(intraday_df, daily_df=None):
        """
        Computes all market structure features for an intraday DataFrame.

        Args:
            intraday_df: DataFrame with Date, Open, High, Low, Close, Volume
            daily_df: Optional DataFrame with Date, Open, High, Low, Close (for PDH/PDL/PDC)

        Returns:
            DataFrame with enriched market structure columns:
            - is_swing_high, is_swing_low (marked at the confirmation bar)
            - last_swing_high_price, last_swing_low_price
            - market_trend ('BULLISH', 'BEARISH', 'CONSOLIDATION')
            - pdh, pdl, pdc, session_open, hod, lod
            - vwap, vwap_upper_1sd, vwap_lower_1sd, vwap_upper_2sd, vwap_lower_2sd
            - dist_to_pdh_pct, dist_to_pdl_pct, dist_to_vwap_pct
        """
        df = intraday_df.copy()
        df['trade_date'] = df['Date'].dt.date
        df['bar_time'] = df['Date'].dt.time

        # 1. Compute Daily Context (PDH, PDL, PDC)
        pdl_map, pdh_map, pdc_map = {}, {}, {}
        if daily_df is not None and not daily_df.empty:
            if 'prev_day_high' in daily_df.columns and 'trade_date' in daily_df.columns:
                for _, row in daily_df.iterrows():
                    pdl_map[row['trade_date']] = row['prev_day_low']
                    pdh_map[row['trade_date']] = row['prev_day_high']
                    pdc_map[row['trade_date']] = row['prev_day_close']
            elif 'Date' in daily_df.columns:
                daily_sorted = daily_df.sort_values('Date').copy()
                daily_sorted['trade_date'] = daily_sorted['Date'].dt.date if hasattr(daily_sorted['Date'].dt, 'date') else daily_sorted['Date']
                daily_sorted['prev_high'] = daily_sorted['High'].shift(1)
                daily_sorted['prev_low'] = daily_sorted['Low'].shift(1)
                daily_sorted['prev_close'] = daily_sorted['Close'].shift(1)

                for _, row in daily_sorted.iterrows():
                    pdl_map[row['trade_date']] = row['prev_low']
                    pdh_map[row['trade_date']] = row['prev_high']
                    pdc_map[row['trade_date']] = row['prev_close']
        
        if not pdh_map:
            # Derive daily context from intraday grouping if daily_df not provided or empty
            daily_agg = df.groupby('trade_date').agg({
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).reset_index()
            daily_agg['prev_high'] = daily_agg['High'].shift(1)
            daily_agg['prev_low'] = daily_agg['Low'].shift(1)
            daily_agg['prev_close'] = daily_agg['Close'].shift(1)

            for _, row in daily_agg.iterrows():
                pdl_map[row['trade_date']] = row['prev_low']
                pdh_map[row['trade_date']] = row['prev_high']
                pdc_map[row['trade_date']] = row['prev_close']

        df['pdh'] = df['trade_date'].map(pdh_map)
        df['pdl'] = df['trade_date'].map(pdl_map)
        df['pdc'] = df['trade_date'].map(pdc_map)

        # 2. Cumulative Intraday Session Features (VWAP, HOD, LOD, Session Open)
        df['cum_vol'] = df.groupby('trade_date')['Volume'].cumsum()
        df['pv'] = df['Close'] * df['Volume']
        df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol'].replace(0, np.nan)

        # VWAP standard deviation bands
        df['pv_dev_sq'] = ((df['Close'] - df['vwap']) ** 2) * df['Volume']
        df['cum_pv_dev_sq'] = df.groupby('trade_date')['pv_dev_sq'].cumsum()
        df['vwap_std'] = np.sqrt(df['cum_pv_dev_sq'] / df['cum_vol'].replace(0, np.nan)).fillna(0)
        df['vwap_upper_1sd'] = df['vwap'] + df['vwap_std']
        df['vwap_lower_1sd'] = df['vwap'] - df['vwap_std']
        df['vwap_upper_2sd'] = df['vwap'] + 2.0 * df['vwap_std']
        df['vwap_lower_2sd'] = df['vwap'] - 2.0 * df['vwap_std']

        # Session Open, HOD, LOD up to current bar
        session_open_map = df.groupby('trade_date')['Open'].first().to_dict()
        df['session_open'] = df['trade_date'].map(session_open_map)
        df['hod'] = df.groupby('trade_date')['High'].cummax()
        df['lod'] = df.groupby('trade_date')['Low'].cummin()

        # Distances to key levels in percentage
        df['dist_to_pdh_pct'] = (df['Close'] - df['pdh']) / df['pdh'] * 100
        df['dist_to_pdl_pct'] = (df['Close'] - df['pdl']) / df['pdl'] * 100
        df['dist_to_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap'] * 100

        # 3. Causal Fractal Swing Pivots (5-bar pivot confirmed at bar t for bar t-2)
        n = len(df)
        is_swing_high = [False] * n
        is_swing_low = [False] * n
        swing_high_prices = [np.nan] * n
        swing_low_prices = [np.nan] * n

        last_sh_price = np.nan
        last_sl_price = np.nan
        sh_history = []
        sl_history = []
        market_trends = ['CONSOLIDATION'] * n

        highs = df['High'].values
        lows = df['Low'].values
        dates = df['trade_date'].values

        for t in range(4, n):
            # Same trading day check across the 5-bar window (t-4 to t)
            if dates[t] == dates[t-4]:
                pivot_idx = t - 2
                # Swing High at t-2
                if (highs[pivot_idx] > highs[t-4] and
                    highs[pivot_idx] > highs[t-3] and
                    highs[pivot_idx] >= highs[t-1] and
                    highs[pivot_idx] >= highs[t]):
                    is_swing_high[t] = True
                    last_sh_price = highs[pivot_idx]
                    sh_history.append((t, last_sh_price))

                # Swing Low at t-2
                if (lows[pivot_idx] < lows[t-4] and
                    lows[pivot_idx] < lows[t-3] and
                    lows[pivot_idx] <= lows[t-1] and
                    lows[pivot_idx] <= lows[t]):
                    is_swing_low[t] = True
                    last_sl_price = lows[pivot_idx]
                    sl_history.append((t, last_sl_price))

            # Store current known swing high and low
            swing_high_prices[t] = last_sh_price
            swing_low_prices[t] = last_sl_price

            # Determine Trend (HH/HL = Bullish, LH/LL = Bearish)
            if len(sh_history) >= 2 and len(sl_history) >= 2:
                recent_sh1, recent_sh2 = sh_history[-2][1], sh_history[-1][1]
                recent_sl1, recent_sl2 = sl_history[-2][1], sl_history[-1][1]

                if recent_sh2 > recent_sh1 and recent_sl2 > recent_sl1:
                    market_trends[t] = 'BULLISH'
                elif recent_sh2 < recent_sh1 and recent_sl2 < recent_sl1:
                    market_trends[t] = 'BEARISH'
                else:
                    market_trends[t] = 'CONSOLIDATION'

        df['is_confirmed_swing_high'] = is_swing_high
        df['is_confirmed_swing_low'] = is_swing_low
        df['last_swing_high'] = swing_high_prices
        df['last_swing_low'] = swing_low_prices
        df['market_trend'] = market_trends

        return df
