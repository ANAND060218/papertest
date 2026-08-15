"""
V15.2 Signal Reporter & Transparent Decision Audit Engine
Generates comprehensive 'Why' audit sheets detailing exact scores, filter statuses, and trade rationale.
"""
import pandas as pd
import numpy as np


class SignalReporter:
    """
    Generates human-auditable signal breakdown and decision rationale.
    """

    @staticmethod
    def generate_detailed_report(df_all_ranked, top_n=7, current_holdings=None):
        """
        Creates an auditable decision sheet explaining why each stock is selected or rejected.
        """
        current_holdings = current_holdings or {}
        report_rows = []

        for idx, row in df_all_ranked.iterrows():
            sym = row['symbol']
            c_p = row['current_price']
            m6 = row['mom_6m_pct']
            m12 = row['mom_12m_pct']
            score = row['combined_momentum_score']
            sma = row['sma_10m']
            passes_sma = row['is_above_10m_sma']
            is_currently_held = sym in current_holdings

            # Decision Logic & Rationale
            if passes_sma and idx < top_n:
                rank = idx + 1
                action = "HOLD" if is_currently_held else "BUY"
                status = "SELECTED"
                rationale = f"Rank #{rank} leader with strong 6M ({m6:+.1f}%) & 12M ({m12:+.1f}%) momentum. Price (Rs {c_p:.2f}) > 10M SMA (Rs {sma:.2f})."
            elif not passes_sma and is_currently_held:
                rank = "-"
                action = "SELL"
                status = "LIQUIDATE_TREND_FAIL"
                rationale = f"Trend filter failed: Price (Rs {c_p:.2f}) < 10M SMA (Rs {sma:.2f}). Protecting capital."
            elif is_currently_held and idx >= top_n:
                rank = idx + 1
                action = "SELL"
                status = "LIQUIDATE_RANK_DROP"
                rationale = f"Fell out of Top {top_n} leaders (Ranked #{rank}). Replaced by higher momentum leader."
            elif not passes_sma:
                rank = "-"
                action = "IGNORE"
                status = "FILTER_REJECTED"
                rationale = f"Below 10M SMA (Rs {sma:.2f}). Absolute momentum filter failed."
            else:
                rank = idx + 1
                action = "IGNORE"
                status = "WATCHLIST"
                rationale = f"Valid uptrend, but ranked #{rank} (outside Top {top_n} cut-off)."

            report_rows.append({
                'rank': rank,
                'symbol': sym,
                'current_price': c_p,
                'mom_6m_pct': m6,
                'mom_12m_pct': m12,
                'combined_score': score,
                'sma_10m_level': sma,
                'trend_filter': "PASS" if passes_sma else "FAIL",
                'action': action,
                'status': status,
                'rationale_reason': rationale
            })

        return pd.DataFrame(report_rows)
