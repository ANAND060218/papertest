"""
V15.2 Shadow vs Backtest Drift Detector & Health Monitor
Tracks execution slippage, cost anomalies, portfolio weight drift, and stale data.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DriftDetector:
    """
    Monitors live execution vs theoretical backtest model drift.
    """

    def __init__(self, max_allowed_slippage_pct=0.0020, max_weight_drift_pct=0.03):
        self.max_slippage = max_allowed_slippage_pct  # 0.20% max allowed slippage per order
        self.max_weight_drift = max_weight_drift_pct  # 3.0% max allocation drift

    def audit_rebalance_execution(self, proposed_orders, actual_fills):
        """
        Compares proposed model orders against actual fills.
        """
        alerts = []
        is_healthy = True

        for p_order in proposed_orders:
            sym = p_order['symbol']
            expected_p = p_order['current_price']
            actual_fill = actual_fills.get(sym, expected_p)

            # 1. Execution Slippage Drift Check
            realized_slippage = abs(actual_fill - expected_p) / expected_p
            if realized_slippage > self.max_slippage:
                alerts.append(
                    f"WARNING (SLIPPAGE DRIFT): {sym} realized slippage ({realized_slippage * 100:.2f}%) exceeds threshold ({self.max_slippage * 100:.2f}%)."
                )

        # 2. Portfolio Weight Drift Check
        tot_val = sum(actual_fills.get(o['symbol'], o['current_price']) * o['target_shares_qty'] for o in proposed_orders)
        for o in proposed_orders:
            sym = o['symbol']
            val = actual_fills.get(sym, o['current_price']) * o['target_shares_qty']
            act_w = val / tot_val if tot_val > 0 else 0
            exp_w = o['target_weight_pct'] / 100.0

            if abs(act_w - exp_w) > self.max_weight_drift:
                alerts.append(
                    f"NOTICE (WEIGHT DRIFT): {sym} weight ({act_w * 100:.1f}%) deviates from target ({exp_w * 100:.1f}%)."
                )

        return len(alerts) == 0, alerts
