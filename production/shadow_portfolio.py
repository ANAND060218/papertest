"""
V15.2 Shadow Portfolio Tracker
Maintains the shadow/paper portfolio ledger, mark-to-market valuations, cash buffer, and drawdown tracking.
"""
import os
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ShadowPortfolio:
    """
    Paper/Shadow Portfolio Ledger & Position Tracker.
    """

    def __init__(self, ledger_path=None, initial_capital=100000.0):
        self.ledger_path = ledger_path or os.path.join(BASE_DIR, "results", "shadow_positions.csv")
        self.history_path = os.path.join(BASE_DIR, "results", "shadow_equity_curve.csv")
        self.initial_capital = initial_capital
        self._ensure_files()

    def _ensure_files(self):
        if not os.path.exists(self.ledger_path):
            df_empty = pd.DataFrame(columns=[
                'symbol', 'shares_qty', 'avg_entry_price', 'current_price',
                'current_value_inr', 'unrealized_pnl_inr', 'unrealized_pnl_pct', 'weight_pct'
            ])
            df_empty.to_csv(self.ledger_path, index=False)

        if not os.path.exists(self.history_path):
            df_hist = pd.DataFrame(columns=['date', 'portfolio_equity', 'cash', 'holdings_count', 'drawdown_pct'])
            df_hist.to_csv(self.history_path, index=False)

    def update_from_rebalance(self, approved_orders, current_prices, rebalance_date):
        """
        Updates the shadow portfolio state following a rebalance.
        """
        positions = []
        invested_val = 0.0

        for o in approved_orders:
            sym = o['symbol']
            qty = o['target_shares_qty']
            c_price = current_prices.get(sym, o.get('current_price', 0.0))
            entry_p = c_price * 1.0005  # Assumed slippage on entry
            val = qty * c_price
            invested_val += val

            positions.append({
                'symbol': sym,
                'shares_qty': qty,
                'avg_entry_price': round(entry_p, 2),
                'current_price': round(c_price, 2),
                'current_value_inr': round(val, 2),
                'unrealized_pnl_inr': 0.0,
                'unrealized_pnl_pct': 0.0,
                'weight_pct': 0.0  # Will normalize below
            })

        cash_remaining = max(0.0, self.initial_capital - invested_val)
        tot_eq = invested_val + cash_remaining

        for p in positions:
            p['weight_pct'] = round((p['current_value_inr'] / tot_eq) * 100, 2)

        df_pos = pd.DataFrame(positions)
        df_pos.to_csv(self.ledger_path, index=False)

        # Update Equity Curve History
        df_hist = pd.read_csv(self.history_path)
        peak = max(tot_eq, df_hist['portfolio_equity'].max() if not df_hist.empty else tot_eq)
        dd = (tot_eq - peak) / peak * 100

        new_hist_row = pd.DataFrame([{
            'date': rebalance_date.strftime("%Y-%m-%d"),
            'portfolio_equity': round(tot_eq, 2),
            'cash': round(cash_remaining, 2),
            'holdings_count': len(positions),
            'drawdown_pct': round(dd, 2)
        }])

        df_hist = pd.concat([df_hist, new_hist_row], ignore_index=True)
        df_hist.to_csv(self.history_path, index=False)

        return df_pos, tot_eq, cash_remaining
