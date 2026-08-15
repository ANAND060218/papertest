"""
V16 Indian Equity Intraday Transaction Cost & Friction Calculator
Calculates exact STT, Brokerage, Exchange Turnover, GST, Stamp Duty, and Slippage.
"""
import numpy as np


class IntradayCostModel:
    """
    Computes statutory Indian intraday transaction costs.
    """

    @staticmethod
    def calculate_intraday_friction(buy_price, sell_price, qty, slippage_pct=0.0003):
        """
        Calculates all friction items for an intraday equity round-trip trade.
        """
        # Slippage on execution
        effective_buy = buy_price * (1 + slippage_pct)
        effective_sell = sell_price * (1 - slippage_pct)

        buy_turnover = effective_buy * qty
        sell_turnover = effective_sell * qty
        total_turnover = buy_turnover + sell_turnover

        # 1. Brokerage (Zerodha / Upstox flat Rs 20 per order or 0.03%)
        brokerage_buy = min(20.0, buy_turnover * 0.0003)
        brokerage_sell = min(20.0, sell_turnover * 0.0003)
        total_brokerage = brokerage_buy + brokerage_sell

        # 2. STT (Securities Transaction Tax) - 0.025% on Sell for Intraday Equity
        stt = sell_turnover * 0.00025

        # 3. Exchange Turnover Charges (NSE: 0.00345%)
        exchange_charges = total_turnover * 0.0000345

        # 4. GST (18% on Brokerage + Exchange Charges)
        gst = (total_brokerage + exchange_charges) * 0.18

        # 5. Stamp Duty (0.003% on Buy for Intraday)
        stamp_duty = buy_turnover * 0.00003

        # 6. SEBI Turnover Charges (Rs 10 per crore = 0.0001%)
        sebi_charges = total_turnover * 0.000001

        total_statutory_costs = total_brokerage + stt + exchange_charges + gst + stamp_duty + sebi_charges
        slippage_cost = ((effective_buy - buy_price) + (sell_price - effective_sell)) * qty
        total_friction = total_statutory_costs + slippage_cost

        gross_pnl = (sell_price - buy_price) * qty
        net_pnl = (effective_sell - effective_buy) * qty - total_statutory_costs

        return {
            'gross_pnl_rs': round(float(gross_pnl), 2),
            'net_pnl_rs': round(float(net_pnl), 2),
            'brokerage_rs': round(float(total_brokerage), 2),
            'stt_rs': round(float(stt), 2),
            'exchange_charges_rs': round(float(exchange_charges), 2),
            'gst_rs': round(float(gst), 2),
            'stamp_duty_rs': round(float(stamp_duty), 2),
            'slippage_cost_rs': round(float(slippage_cost), 2),
            'total_friction_rs': round(float(total_friction), 2),
            'friction_drag_pct': round(float(total_friction / buy_turnover * 100), 3) if buy_turnover > 0 else 0.0
        }
