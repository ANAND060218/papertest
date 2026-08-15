"""
V8 Forward Validation Report Generator
Automatically evaluates live forward paper trading performance without modifying the frozen strategy.

Metrics Evaluated:
  1. Overall Forward Performance (Win Rate, Profit Factor, Expectancy, Max Drawdown, Sharpe)
  2. Setup-Wise Performance (ORB, VWAP, Prev-Day High)
  3. Stock-Wise Performance (TCS, INFY, RELIANCE, SBIN, ICICI, HDFC)
  4. Market Regime Breakdown (Trending vs Sideways vs High Vol Chop)
  5. Execution Quality & Slippage Divergence (Modeled vs Actual Realized Slippage)
  6. Model Calibration (Predicted P(win) Buckets vs Realized Win Rates)
  7. Counterfactual Analysis (Accepted Trades vs Rejected Setups)
  8. Minimum Evidence Progress (Calendar Duration, Trade Count, Regime Diversity)
"""
import sys
import os
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from execution.trade_journal import JOURNAL_DB_PATH


FORWARD_REPORT_JSON = os.path.join(config.RESULTS_DIR, "v8_forward_validation_report.json")


class ForwardValidationAuditor:
    """
    Automated scientific auditor for V8 forward paper trading logs.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or JOURNAL_DB_PATH

    def load_data(self):
        from execution.trade_journal import TradeJournal
        tj = TradeJournal(self.db_path)  # Ensures all tables are created

        conn = sqlite3.connect(self.db_path)
        try:
            trades_df = pd.read_sql_query("SELECT * FROM journal_trades ORDER BY created_at ASC", conn)
        except Exception:
            trades_df = pd.DataFrame()

        try:
            rejected_df = pd.read_sql_query("SELECT * FROM journal_rejected_setups ORDER BY created_at ASC", conn)
        except Exception:
            rejected_df = pd.DataFrame()
        conn.close()

        if not trades_df.empty:
            trades_df['created_at'] = pd.to_datetime(trades_df['created_at'])
            if 'exit_timestamp' in trades_df.columns:
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'])

        if not rejected_df.empty:
            rejected_df['created_at'] = pd.to_datetime(rejected_df['created_at'])

        return trades_df, rejected_df

    def audit_overall_performance(self, closed_trades):
        if closed_trades.empty:
            return {
                'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0, 'net_pnl': 0.0,
                'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0, 'total_costs': 0.0
            }

        wins = closed_trades[closed_trades['net_pnl'] > 0]
        losses = closed_trades[closed_trades['net_pnl'] <= 0]

        total = len(closed_trades)
        win_rate = (len(wins) / total) * 100 if total > 0 else 0.0

        gross_win = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1.0
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        expectancy = closed_trades['net_pnl'].mean()
        net_pnl = closed_trades['net_pnl'].sum()
        total_costs = closed_trades['total_costs'].sum() if 'total_costs' in closed_trades.columns else 0.0

        # Drawdown calculation
        equity_curve = [config.INITIAL_CAPITAL]
        running_eq = config.INITIAL_CAPITAL
        max_dd = 0.0
        peak = running_eq
        for pnl in closed_trades['net_pnl']:
            running_eq += pnl
            peak = max(peak, running_eq)
            dd = (peak - running_eq) / peak * 100
            max_dd = max(max_dd, dd)
            equity_curve.append(running_eq)

        # Sharpe ratio
        returns = closed_trades['return_pct'].values if 'return_pct' in closed_trades.columns else np.array([])
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(250 * 2)

        return {
            'total_trades': int(total),
            'wins': int(len(wins)),
            'losses': int(len(losses)),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(pf, 3),
            'expectancy': round(expectancy, 2),
            'net_pnl': round(net_pnl, 2),
            'total_costs': round(total_costs, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2)
        }

    def audit_breakdown(self, closed_trades, group_col):
        if closed_trades.empty or group_col not in closed_trades.columns:
            return pd.DataFrame()

        records = []
        for name, grp in closed_trades.groupby(group_col):
            perf = self.audit_overall_performance(grp)
            perf[group_col] = name
            records.append(perf)

        df = pd.DataFrame(records)
        cols = [group_col, 'total_trades', 'wins', 'win_rate', 'profit_factor', 'expectancy', 'net_pnl', 'max_drawdown_pct']
        return df[[c for c in cols if c in df.columns]].sort_values('profit_factor', ascending=False)

    def audit_execution_quality(self, closed_trades):
        if closed_trades.empty:
            return {}

        # Slippage on entry: (fill_entry_price - entry_price) / entry_price
        entry_slip = (closed_trades['fill_entry_price'] - closed_trades['entry_price']) / closed_trades['entry_price'] * 100
        # Slippage on exit: (exit_price - fill_exit_price) / exit_price
        exit_slip = (closed_trades['exit_price'] - closed_trades['fill_exit_price']) / closed_trades['exit_price'] * 100

        return {
            'avg_entry_slippage_pct': round(float(entry_slip.mean()), 4),
            'avg_exit_slippage_pct': round(float(exit_slip.mean()), 4),
            'avg_total_slippage_pct': round(float(entry_slip.mean() + exit_slip.mean()), 4),
            'modeled_slippage_pct': round(config.SLIPPAGE_PCT * 2 * 100, 4),
            'avg_transaction_cost_per_trade_rs': round(float(closed_trades['total_costs'].mean()), 2)
        }

    def audit_model_calibration(self, closed_trades):
        if closed_trades.empty or 'xgb_probability' not in closed_trades.columns:
            return pd.DataFrame()

        bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
        labels = ['<0.40', '0.40-0.50', '0.50-0.60', '0.60-0.70', '0.70-0.80', '0.80+']
        closed_trades = closed_trades.copy()
        closed_trades['p_bucket'] = pd.cut(closed_trades['xgb_probability'], bins=bins, labels=labels, right=False)

        records = []
        for bucket, grp in closed_trades.groupby('p_bucket', observed=False):
            n = len(grp)
            if n == 0:
                continue
            wins = (grp['net_pnl'] > 0).sum()
            actual_win_rate = (wins / n) * 100
            mean_predicted_p = grp['xgb_probability'].mean() * 100
            records.append({
                'ProbabilityBucket': bucket,
                'Trades': n,
                'MeanPredictedP%': round(mean_predicted_p, 1),
                'ActualWinRate%': round(actual_win_rate, 1),
                'CalibrationDelta%': round(actual_win_rate - mean_predicted_p, 1),
                'ProfitFactor': round(self.audit_overall_performance(grp)['profit_factor'], 3)
            })

        return pd.DataFrame(records)

    def audit_counterfactuals(self, closed_trades, rejected_df):
        total_accepted = len(closed_trades)
        total_rejected = len(rejected_df)

        rejection_reasons = {}
        if not rejected_df.empty and 'rejection_reason' in rejected_df.columns:
            rejection_reasons = rejected_df['rejection_reason'].value_counts().to_dict()

        return {
            'total_setups_generated': total_accepted + total_rejected,
            'setups_accepted': total_accepted,
            'setups_rejected': total_rejected,
            'acceptance_rate_pct': round(total_accepted / (total_accepted + total_rejected) * 100, 2) if (total_accepted + total_rejected) > 0 else 0.0,
            'rejection_reasons_breakdown': rejection_reasons
        }

    def generate_full_audit_report(self):
        trades_df, rejected_df = self.load_data()
        closed_trades = trades_df[trades_df['status'] == 'CLOSED'] if not trades_df.empty and 'status' in trades_df.columns else pd.DataFrame()

        print("=" * 100)
        print("V8 FORWARD VALIDATION SCIENTIFIC AUDIT REPORT")
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print("=" * 100)

        # 1. Overall Performance
        overall = self.audit_overall_performance(closed_trades)
        print("\n--- 1. Overall Forward Strategy Performance ---")
        for k, v in overall.items():
            print(f"  {k:25s}: {v}")

        # 2. Setup Breakdown
        print("\n--- 2. Setup-Wise Performance Breakdown ---")
        setup_df = self.audit_breakdown(closed_trades, 'setup_type')
        if not setup_df.empty:
            print(setup_df.to_string(index=False))
        else:
            print("  No closed trades logged yet.")

        # 3. Stock Breakdown
        print("\n--- 3. Stock-Wise Performance Breakdown ---")
        stock_df = self.audit_breakdown(closed_trades, 'symbol')
        if not stock_df.empty:
            print(stock_df.to_string(index=False))
        else:
            print("  No closed trades logged yet.")

        # 4. Regime Breakdown
        print("\n--- 4. Market Regime Performance Breakdown ---")
        regime_df = self.audit_breakdown(closed_trades, 'regime')
        if not regime_df.empty:
            print(regime_df.to_string(index=False))
        else:
            print("  No closed trades logged yet.")

        # 5. Execution Quality
        print("\n--- 5. Execution Quality & Slippage Divergence ---")
        exec_qual = self.audit_execution_quality(closed_trades)
        for k, v in exec_qual.items():
            print(f"  {k:35s}: {v}")

        # 6. Model Calibration
        print("\n--- 6. Probability Calibration Curve ---")
        calib_df = self.audit_model_calibration(closed_trades)
        if not calib_df.empty:
            print(calib_df.to_string(index=False))
        else:
            print("  No closed trades logged yet.")

        # 7. Counterfactual Analysis
        print("\n--- 7. Counterfactual & Rejection Audit ---")
        counterfactual = self.audit_counterfactuals(closed_trades, rejected_df)
        for k, v in counterfactual.items():
            if k == 'rejection_reasons_breakdown':
                print(f"  {k}:")
                for rk, rv in v.items():
                    print(f"    - {rk}: {rv}")
            else:
                print(f"  {k:30s}: {v}")

        # 8. Evidence Gates Status
        print("\n--- 8. Minimum Evidence Gates Status ---")
        n_trades = overall['total_trades']
        n_days = 0
        if not trades_df.empty:
            n_days = len(trades_df['created_at'].dt.date.unique())

        print(f"  Trade Count Gate (>=100 trades):     {n_trades}/100 ({'MET' if n_trades >= 100 else 'PENDING'})")
        print(f"  Duration Gate (>=8 weeks / 40 days): {n_days}/40 days ({'MET' if n_days >= 40 else 'PENDING'})")
        print(f"  Current Status:                      IN PROGRESS (DO NOT RETRAIN OR TUNE)")

        # Compile report
        report_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
            'overall_performance': overall,
            'setup_breakdown': setup_df.to_dict('records') if not setup_df.empty else [],
            'stock_breakdown': stock_df.to_dict('records') if not stock_df.empty else [],
            'regime_breakdown': regime_df.to_dict('records') if not regime_df.empty else [],
            'execution_quality': exec_qual,
            'model_calibration': calib_df.to_dict('records') if not calib_df.empty else [],
            'counterfactual_audit': counterfactual,
            'evidence_gates': {
                'total_trades': n_trades,
                'trade_gate_met': bool(n_trades >= 100),
                'active_trading_days': n_days,
                'duration_gate_met': bool(n_days >= 40)
            }
        }

        with open(FORWARD_REPORT_JSON, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[AUDIT] Report successfully saved to: {FORWARD_REPORT_JSON}")

        return report_data


if __name__ == "__main__":
    auditor = ForwardValidationAuditor()
    auditor.generate_full_audit_report()
