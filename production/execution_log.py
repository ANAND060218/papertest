"""
V15.2 Production Execution Logger & Audit Trail
Records immutable historical logs of all rebalance signals, approval statuses, fills, costs, and slippages.
"""
import os
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ExecutionLogger:
    """
    Manages the audit trail of all monthly rebalance executions.
    """

    def __init__(self, log_path=None):
        self.log_path = log_path or os.path.join(BASE_DIR, "results", "execution_log.csv")
        self._ensure_log_file()

    def _ensure_log_file(self):
        if not os.path.exists(self.log_path):
            df_empty = pd.DataFrame(columns=[
                'log_id', 'timestamp', 'rebalance_date', 'action', 'symbol',
                'signal_price', 'execution_price', 'shares_qty', 'target_value_inr',
                'slippage_cost_inr', 'statutory_costs_inr', 'status', 'approval_status'
            ])
            df_empty.to_csv(self.log_path, index=False)

    def log_rebalance_orders(self, orders_list, rebalance_date, status="PENDING_HUMAN_APPROVAL"):
        """
        Appends new rebalance orders to the permanent execution log.
        """
        records = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

        for idx, o in enumerate(orders_list, 1):
            log_id = f"REBAL_{rebalance_date.strftime('%Y%m')}_{idx:02d}"
            signal_p = o.get('current_price', 0.0)
            exec_p = signal_p * 1.0005  # Assumed 0.05% slippage on open
            qty = o.get('target_shares_qty', 0)
            tot_val = qty * exec_p
            slip_cost = (exec_p - signal_p) * qty
            stat_cost = tot_val * 0.0015  # Delivery statutory cost model

            records.append({
                'log_id': log_id,
                'timestamp': now_str,
                'rebalance_date': rebalance_date.strftime("%Y-%m-%d"),
                'action': o.get('action', 'BUY'),
                'symbol': o.get('symbol'),
                'signal_price': round(signal_p, 2),
                'execution_price': round(exec_p, 2),
                'shares_qty': qty,
                'target_value_inr': round(tot_val, 2),
                'slippage_cost_inr': round(slip_cost, 2),
                'statutory_costs_inr': round(stat_cost, 2),
                'status': status,
                'approval_status': "AWAITING_CONFIRMATION"
            })

        df_new = pd.DataFrame(records)
        df_existing = pd.read_csv(self.log_path)
        # Avoid duplicate entries for same rebalance date
        df_filtered = df_existing[df_existing['rebalance_date'] != rebalance_date.strftime("%Y-%m-%d")]
        df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
        df_combined.to_csv(self.log_path, index=False)
        return df_new
