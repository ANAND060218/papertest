"""
V15.2 Production Risk Guard & Pre-Trade Safety Layer
Implements structural fail-safes, data integrity checks, position caps, and automated kill-switches.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RiskGuard:
    """
    Pre-Trade Safety Validator & Risk Enforcer.
    """

    def __init__(self, config):
        self.cfg = config
        self.max_position_weight = self.cfg['portfolio_construction'].get('max_position_cap_pct', 20.0) / 100.0
        self.min_cash_buffer = self.cfg['portfolio_construction'].get('cash_buffer_pct', 2.0) / 100.0
        self.top_n = self.cfg['portfolio_construction'].get('holdings_count_top_n', 7)

    def validate_market_data(self, price_matrix, latest_date):
        """
        Pillar 1: Data Integrity & Freshness Check (Kill-Switch Condition).
        """
        errors = []

        if price_matrix.empty:
            errors.append("KILL-SWITCH: Price matrix is empty.")
            return False, errors

        # Check for duplicate symbols
        if len(price_matrix.columns) != len(set(price_matrix.columns)):
            errors.append("KILL-SWITCH: Duplicate symbols detected in price matrix.")

        # Check for NaN / missing values in the latest row
        latest_prices = price_matrix.loc[latest_date]
        nan_symbols = latest_prices[latest_prices.isna()].index.tolist()
        if len(nan_symbols) > 5:
            errors.append(f"KILL-SWITCH: Excessive missing prices on {latest_date}: {nan_symbols}")

        # Check for non-positive prices
        invalid_prices = latest_prices[latest_prices <= 0].index.tolist()
        if invalid_prices:
            errors.append(f"KILL-SWITCH: Zero or negative prices detected: {invalid_prices}")

        # Check minimum history length (at least 24 months for 12m momentum + 10m SMA)
        if len(price_matrix) < 24:
            errors.append(f"KILL-SWITCH: Insufficient historical depth ({len(price_matrix)} months < 24 months required).")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_allocations(self, proposed_allocations, total_portfolio_capital):
        """
        Pillar 2: Pre-Trade Capital & Position Sizing Risk Guard.
        """
        errors = []
        warnings = []

        if not proposed_allocations:
            errors.append("KILL-SWITCH: Proposed allocations list is empty.")
            return False, errors, warnings

        # Total Weight Cap Check
        tot_target_weight = sum(a['target_weight_pct'] for a in proposed_allocations) / 100.0
        max_allowed_weight = 1.0 - self.min_cash_buffer

        if tot_target_weight > 1.001:
            errors.append(f"RISK VIOLATION: Total proposed weight ({tot_target_weight * 100:.1f}%) exceeds 100%.")

        # Single Position Cap Check
        for a in proposed_allocations:
            w = a['target_weight_pct'] / 100.0
            if w > self.max_position_weight:
                errors.append(f"RISK VIOLATION: {a['symbol']} weight ({w * 100:.1f}%) exceeds max cap ({self.max_position_weight * 100:.1f}%).")

            # Check order value vs minimum threshold
            val = a.get('target_value_inr', 0)
            if val < 3000.0:
                warnings.append(f"SMALL ALLOCATION WARNING: {a['symbol']} value (Rs {val:,.2f}) is below Rs 3,000 threshold.")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings
