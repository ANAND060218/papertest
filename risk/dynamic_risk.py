"""
V9 Phase 3 -- Dynamic ATR & Structural Risk Engine
Dynamically sizes stop-loss, profit-target, and position quantity based on market volatility and structural support/resistance.

Replaces fixed percentage targets/stops (+1.0% / -0.5%) with market-adaptive boundaries.
"""
import numpy as np
import pandas as pd


class DynamicRiskEngine:
    """
    Dynamic Volatility-Based Risk & Position Sizing Engine.
    """

    @staticmethod
    def compute_atr(df, period=14):
        """Calculates 14-bar Average True Range."""
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=max(5, period // 2)).mean()
        return atr.bfill()

    @staticmethod
    def calculate_trade_boundaries(
        direction,
        entry_price,
        current_atr,
        structure_support=None,
        structure_resistance=None,
        pattern_extreme=None,
        min_rr=1.50
    ):
        """
        Calculates dynamic stop-loss and profit-target prices.

        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Float entry price
            current_atr: Float 5m ATR value
            structure_support: Float major support level (PDL, Swing Low, etc.)
            structure_resistance: Float major resistance level (PDH, Swing High, etc.)
            pattern_extreme: Float low of hammer or high of shooting star
            min_rr: Float minimum acceptable Reward-to-Risk ratio (default 1.50)

        Returns:
            dict with stop_price, target_price, risk_distance, target_distance, rr_ratio, is_valid_rr
        """
        # Minimum ATR buffer
        atr_buffer = 0.20 * current_atr if current_atr > 0 else 0.002 * entry_price
        min_stop_dist = max(0.50 * current_atr, 0.003 * entry_price)   # At least 0.3% stop
        max_stop_dist = max(2.50 * current_atr, 0.015 * entry_price)   # At most 1.5% stop

        if direction == 'LONG':
            # Stop below structure support or pattern low
            candidates = []
            if pattern_extreme is not None and pd.notna(pattern_extreme) and pattern_extreme < entry_price:
                candidates.append(pattern_extreme - atr_buffer)
            if structure_support is not None and pd.notna(structure_support) and structure_support < entry_price:
                candidates.append(structure_support - atr_buffer)

            if candidates:
                raw_stop = min(candidates)
                raw_stop_dist = entry_price - raw_stop
                stop_dist = np.clip(raw_stop_dist, min_stop_dist, max_stop_dist)
            else:
                stop_dist = np.clip(1.0 * current_atr, min_stop_dist, max_stop_dist)

            stop_price = round(entry_price - stop_dist, 2)
            risk_dist = entry_price - stop_price

            # Target: Nearest resistance or at least min_rr * risk_dist
            target_candidates = [entry_price + min_rr * risk_dist]
            if structure_resistance is not None and pd.notna(structure_resistance) and structure_resistance > entry_price:
                target_candidates.append(structure_resistance)

            target_price = round(max(target_candidates), 2)
            target_dist = target_price - entry_price

        else: # SHORT
            candidates = []
            if pattern_extreme is not None and pd.notna(pattern_extreme) and pattern_extreme > entry_price:
                candidates.append(pattern_extreme + atr_buffer)
            if structure_resistance is not None and pd.notna(structure_resistance) and structure_resistance > entry_price:
                candidates.append(structure_resistance + atr_buffer)

            if candidates:
                raw_stop = max(candidates)
                raw_stop_dist = raw_stop - entry_price
                stop_dist = np.clip(raw_stop_dist, min_stop_dist, max_stop_dist)
            else:
                stop_dist = np.clip(1.0 * current_atr, min_stop_dist, max_stop_dist)

            stop_price = round(entry_price + stop_dist, 2)
            risk_dist = stop_price - entry_price

            # Target
            target_candidates = [entry_price - min_rr * risk_dist]
            if structure_support is not None and pd.notna(structure_support) and structure_support < entry_price:
                target_candidates.append(structure_support)

            target_price = round(min(target_candidates), 2)
            target_dist = entry_price - target_price

        rr_ratio = round(target_dist / max(risk_dist, 0.01), 2)
        is_valid_rr = rr_ratio >= min_rr

        return {
            'stop_price': stop_price,
            'target_price': target_price,
            'risk_distance': risk_dist,
            'target_distance': target_dist,
            'rr_ratio': rr_ratio,
            'is_valid_rr': is_valid_rr
        }

    @staticmethod
    def calculate_position_size(entry_price, stop_price, capital=100000, risk_pct=0.01, max_cap_pct=0.10):
        """
        Sizes shares so maximum risk = capital * risk_pct (1.0%), capped at capital * max_cap_pct (10.0%).
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        max_risk_amount = capital * risk_pct
        shares_by_risk = int(max_risk_amount / risk_per_share)

        max_position_capital = capital * max_cap_pct
        shares_by_cap = int(max_position_capital / entry_price)

        shares = min(shares_by_risk, shares_by_cap)
        return max(1, shares) if shares > 0 else 0
