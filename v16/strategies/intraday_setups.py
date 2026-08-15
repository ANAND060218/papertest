"""
V16 Intraday Strategy Setups & Signal Generation
Calculates precise Entry Price, Stop Loss, Target 1 (1:1.5), Target 2 (1:2.5), and Risk:Reward ratios.
"""
import pandas as pd
import numpy as np


class IntradaySetupEngine:
    """
    Evaluates intraday price action against concrete setups.
    """

    @staticmethod
    def evaluate_orb_setup(stock_data_5m, direction='LONG'):
        """
        Setup 1: 15-Minute Opening Range Breakout (ORB).
        Range defined by the first three 5m bars (09:15 - 09:30).
        """
        if len(stock_data_5m) < 3:
            return None

        # First 3 bars = 15m Opening Range
        or_bars = stock_data_5m.iloc[:3]
        or_high = or_bars['High'].max()
        or_low = or_bars['Low'].min()
        or_range = or_high - or_low

        if or_range <= 0:
            return None

        # Sizing / Targets
        if direction == 'LONG':
            entry_p = or_high + 0.05  # Slight buffer above OR High
            stop_loss = or_low  # Stop at OR Low
            risk_amt = entry_p - stop_loss
            target_1 = entry_p + (risk_amt * 1.5)
            target_2 = entry_p + (risk_amt * 2.5)
        else:
            entry_p = or_low - 0.05
            stop_loss = or_high
            risk_amt = stop_loss - entry_p
            target_1 = entry_p - (risk_amt * 1.5)
            target_2 = entry_p - (risk_amt * 2.5)

        risk_pct = (risk_amt / entry_p) * 100

        return {
            'setup_name': '15M_OPENING_RANGE_BREAKOUT',
            'direction': direction,
            'or_high': round(float(or_high), 2),
            'or_low': round(float(or_low), 2),
            'entry_trigger_price': round(float(entry_p), 2),
            'stop_loss_price': round(float(stop_loss), 2),
            'target_1_price': round(float(target_1), 2),
            'target_2_price': round(float(target_2), 2),
            'risk_per_share_rs': round(float(risk_amt), 2),
            'risk_pct': round(float(risk_pct), 2),
            'reward_risk_ratio': '1 : 1.5 - 2.5',
            'trigger_condition': f"5m bar closes {'above' if direction == 'LONG' else 'below'} Rs {entry_p:.2f}",
            'execution_window': "09:30 - 11:00 IST",
            'eod_square_off': "15:15 IST MANDATORY"
        }

    @staticmethod
    def evaluate_vwap_pullback_setup(current_price, vwap_price, direction='LONG', atr_val=10.0):
        """
        Setup 2: VWAP Trend Pullback & Continuation.
        """
        if direction == 'LONG':
            entry_p = current_price
            stop_loss = vwap_price - (atr_val * 0.3)
            risk_amt = max(0.5, entry_p - stop_loss)
            target_1 = entry_p + (risk_amt * 1.5)
            target_2 = entry_p + (risk_amt * 2.5)
        else:
            entry_p = current_price
            stop_loss = vwap_price + (atr_val * 0.3)
            risk_amt = max(0.5, stop_loss - entry_p)
            target_1 = entry_p - (risk_amt * 1.5)
            target_2 = entry_p - (risk_amt * 2.5)

        risk_pct = (risk_amt / entry_p) * 100

        return {
            'setup_name': 'VWAP_TREND_PULLBACK',
            'direction': direction,
            'vwap_reference_price': round(float(vwap_price), 2),
            'entry_trigger_price': round(float(entry_p), 2),
            'stop_loss_price': round(float(stop_loss), 2),
            'target_1_price': round(float(target_1), 2),
            'target_2_price': round(float(target_2), 2),
            'risk_per_share_rs': round(float(risk_amt), 2),
            'risk_pct': round(float(risk_pct), 2),
            'reward_risk_ratio': '1 : 2.0',
            'trigger_condition': f"Price bounces off VWAP (Rs {vwap_price:.2f}) with 5m candle confirmation",
            'execution_window': "10:00 - 14:00 IST",
            'eod_square_off': "15:15 IST MANDATORY"
        }
