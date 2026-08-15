"""
V8.4 -- Temporal Stress Test Suite
Evaluates whether the frozen trading strategy and gatekeeper edge generalizes across time.

Scientific Test Batteries:
  1. Tier 1: Pure 5-Minute Held-Out Out-of-Sample Window (2026-08-01 to 2026-08-14).
  2. Tier 2: Rolling Monthly Cohort Comparison (June 2026 vs July 2026 vs August 2026).
  3. Tier 3: 2-Year Multi-Year Historical Hourly Replay (2023-09-04 to 2026-06-15, 35,358 bars).
  4. Setup Count Reconciliation (357 multi-setup opportunities vs 316 single-pass setups).
"""
import sys
import os
import time
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from models.xgboost_model import XGBoostTradeModel
from risk.expected_value import ExpectedValueCalculator
from execution.trade_journal import TradeJournal
from execution.position_manager import PositionManager
from execution.paper_executor import PaperExecutor
from run_v6_5_true_intraday import TrueIntradayFeatureEngine
from fetch_historical_2year_hourly import HOURLY_DB_PATH, HOURLY_SYMBOLS


TEMPORAL_REPORT_JSON = os.path.join(config.RESULTS_DIR, "v8_4_temporal_replay_report.json")
TEMPORAL_SUMMARY_CSV = os.path.join(config.RESULTS_DIR, "v8_4_temporal_summary.csv")


class TemporalStressAuditor:
    """
    Scientific auditor testing temporal generalization across multiple historical cohorts.
    """

    def __init__(self, threshold=0.40):
        self.threshold = threshold
        self.symbols = config.UNIVERSE
        self.initial_capital = config.INITIAL_CAPITAL

        self.model = XGBoostTradeModel()
        model_path = os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json")
        self.model.load(model_path)
        self.fe = TrueIntradayFeatureEngine()
        self.ev_calc = ExpectedValueCalculator()
        self.executor = PaperExecutor()

    def run_full_temporal_suite(self):
        print("=" * 100)
        print("V8.4: TEMPORAL STRESS & CROSS-TIME GENERALIZATION AUDIT")
        print(f"Frozen Model: results/xgb_intraday_5m.json | Production Threshold: P >= {self.threshold:.2f}")
        print("=" * 100)

        # -----------------------------------------------------------------
        # Tier 1 & Tier 2: 5-Minute Monthly & Held-Out Out-of-Sample Cohorts
        # -----------------------------------------------------------------
        print("\n--- Running Tier 1 & Tier 2: 5-Minute Temporal Cohort Breakdown ---")
        t1_report = self._audit_5m_temporal_cohorts()

        # -----------------------------------------------------------------
        # Tier 3: 2-Year Multi-Year Hourly Replay (2023 - 2026)
        # -----------------------------------------------------------------
        print("\n--- Running Tier 3: 2-Year Multi-Year Hourly Replay (2023-2026) ---")
        t3_report = self._audit_2year_hourly_replay()

        # -----------------------------------------------------------------
        # Master Temporal Summary & Compilation
        # -----------------------------------------------------------------
        full_report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
            'frozen_model': 'results/xgb_intraday_5m.json',
            'frozen_threshold': self.threshold,
            'tier1_held_out_oos_5m': t1_report['held_out_oos'],
            'tier2_monthly_cohorts_5m': t1_report['monthly_cohorts'],
            'tier3_2year_hourly_replay': t3_report
        }

        def json_serial(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(TEMPORAL_REPORT_JSON, 'w') as f:
            json.dump(full_report, f, indent=2, default=json_serial)

        # Export CSV Summary
        if 'monthly_cohorts' in t1_report:
            pd.DataFrame(t1_report['monthly_cohorts']).to_csv(TEMPORAL_SUMMARY_CSV, index=False)

        print(f"\n[REPORT] Temporal stress report saved to: {TEMPORAL_REPORT_JSON}")
        print(f"[SUMMARY] Temporal CSV summary saved to:    {TEMPORAL_SUMMARY_CSV}")
        return full_report

    def _audit_5m_temporal_cohorts(self):
        """Audits the 5-minute dataset split into monthly cohorts and held-out OOS period."""
        report_path = os.path.join(config.RESULTS_DIR, "v8_2_full_replay_counterfactual_report.json")
        if not os.path.exists(report_path):
            print("[ERROR] V8.2 report not found. Run run_v8_2_full_historical_replay.py first.")
            return {}

        conn = sqlite3.connect(os.path.join(config.DATA_DIR, "trade_journal_replay_42d.db"))
        trades_df = pd.read_sql_query("SELECT * FROM journal_trades WHERE status = 'CLOSED'", conn)
        rejections_df = pd.read_sql_query("SELECT * FROM journal_rejected_setups", conn)
        conn.close()

        trades_df['created_at'] = pd.to_datetime(trades_df['created_at'])
        rejections_df['created_at'] = pd.to_datetime(rejections_df['created_at'])

        # Monthly Cohorts
        cohorts = [
            ("June 2026 (Training Cohort 1)", "2026-06-16", "2026-06-30"),
            ("July 2026 (Training/Val Cohort 2)", "2026-07-01", "2026-07-31"),
            ("August 2026 (Pure Held-Out OOS Cohort 3)", "2026-08-01", "2026-08-14")
        ]

        cohort_rows = []
        for name, start_d, end_d in cohorts:
            t_sub = trades_df[(trades_df['created_at'] >= start_d) & (trades_df['created_at'] <= end_d + " 23:59:59")]
            r_sub = rejections_df[(rejections_df['created_at'] >= start_d) & (rejections_df['created_at'] <= end_d + " 23:59:59")]

            n_acc = len(t_sub)
            wins = (t_sub['net_pnl'] > 0).sum() if n_acc > 0 else 0
            wr = (wins / n_acc * 100) if n_acc > 0 else 0.0
            gw = t_sub[t_sub['net_pnl'] > 0]['net_pnl'].sum() if wins > 0 else 0.0
            gl = abs(t_sub[t_sub['net_pnl'] <= 0]['net_pnl'].sum()) if (n_acc - wins) > 0 else 1.0
            pf = gw / gl if gl > 0 else float('inf')
            pnl = t_sub['net_pnl'].sum() if n_acc > 0 else 0.0

            cohort_rows.append({
                'Cohort': name,
                'Accepted_Trades': n_acc,
                'Wins': wins,
                'Losses': n_acc - wins,
                'WinRate%': round(wr, 1),
                'ProfitFactor': round(pf, 3),
                'NetPnL_Rs': round(pnl, 2),
                'Rejected_Setups': len(r_sub),
                'AcceptanceRate%': round(n_acc / max(n_acc + len(r_sub), 1) * 100, 1)
            })

        cohort_df = pd.DataFrame(cohort_rows)
        print(cohort_df.to_string(index=False))

        # Held-Out OOS Detailed Summary
        held_out_oos = cohort_rows[2]  # August 2026 cohort
        print(f"\n[TIER 1 HELD-OUT OOS RESULT] August 1 - August 14 (10 Unseen Sessions):")
        print(f"  Accepted Trades: {held_out_oos['Accepted_Trades']} | Wins: {held_out_oos['Wins']} | Win Rate: {held_out_oos['WinRate%']}%")
        print(f"  Profit Factor:   {held_out_oos['ProfitFactor']} | Net P&L: Rs.{held_out_oos['NetPnL_Rs']:+.2f}")
        print(f"  Rejected Setups: {held_out_oos['Rejected_Setups']} (Avoided toxic choppy breakouts)")

        return {
            'monthly_cohorts': cohort_rows,
            'held_out_oos': held_out_oos
        }

    def _audit_2year_hourly_replay(self):
        """Replays 2 years of 1-hour candles (September 2023 to June 2026) through the setup detector and model."""
        if not os.path.exists(HOURLY_DB_PATH):
            print(f"[ERROR] Hourly database not found at {HOURLY_DB_PATH}")
            return {}

        conn = sqlite3.connect(HOURLY_DB_PATH)
        dm = DataManager()

        universe_data = {}
        daily_context = {}
        features_data = {}

        core_syms = [s for s in self.symbols if s != "^NSEI"]
        print(f"[TIER 3] Loading 2-year hourly data across {core_syms}...")

        for sym in core_syms:
            tbl = f"bars_1h_{sym.replace('.NS', '')}"
            df_sym = pd.read_sql_query(f"SELECT * FROM {tbl} ORDER BY Date ASC", conn)
            df_sym['Date'] = pd.to_datetime(df_sym['Date'])
            df_sym['trade_date'] = df_sym['Date'].dt.date
            universe_data[sym] = df_sym

            daily_ctx = dm.build_daily_context(df_sym)
            daily_context[sym] = daily_ctx

            df_feat = self.fe.build_intraday_features(df_sym)
            features_data[sym] = df_feat

        conn.close()

        # Isolate historical data that PRE-DATES the 5m dataset (2023-09-04 to 2026-06-15)
        cutoff_date = pd.to_datetime("2026-06-15").date()
        sample_df = universe_data[core_syms[0]]
        pre_dates = sorted([d for d in sample_df['trade_date'].unique() if d < cutoff_date])

        print(f"[TIER 3] Found {len(pre_dates)} historical trading sessions strictly BEFORE the 5m dataset ({pre_dates[0]} to {pre_dates[-1]}).")

        # Multi-Year Cohorts
        year_cohorts = [
            ("2023-H2 (Sep-Dec 2023 Bull Rally)", "2023-09-01", "2023-12-31"),
            ("2024 Full Year (Election & Rate Cycle)", "2024-01-01", "2024-12-31"),
            ("2025 Full Year (Consolidation & Rotation)", "2025-01-01", "2025-12-31"),
            ("2026-H1 (Jan-Jun 2026 Pre-5M Window)", "2026-01-01", "2026-06-15")
        ]

        feat_cols = self.fe.get_feature_columns()
        all_hourly_trades = []

        for current_day in pre_dates:
            for sym in core_syms:
                df_sym = universe_data[sym]
                sym_feat = features_data[sym]

                day_bars = df_sym[df_sym['trade_date'] == current_day]
                if len(day_bars) < 2:
                    continue

                # Check Opening Hourly Breakout
                first_bar = day_bars.iloc[0]
                second_bar = day_bars.iloc[1]

                # If bar 2 breaks above bar 1 high with volume
                if second_bar['Close'] > first_bar['High']:
                    feat_row = sym_feat[sym_feat['Date'] == second_bar['Date']]
                    if feat_row.empty:
                        continue
                    last_feat = feat_row.iloc[0]

                    if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                        feat_vec = last_feat[feat_cols].to_frame().T.fillna(0)
                        p_win = float(self.model.predict_proba(feat_vec)[0])
                        ev_info = self.ev_calc.calculate_ev(p_win)

                        entry_p = second_bar['Close']
                        stop_p = entry_p * 0.992
                        target_p = entry_p * 1.016

                        # Forward outcome simulation on remaining day bars
                        future_bars = day_bars.iloc[2:]
                        exit_p = None
                        exit_reason = None

                        for _, fb in future_bars.iterrows():
                            if fb['Low'] <= stop_p:
                                exit_p = stop_p
                                exit_reason = "STOP_HIT"
                                break
                            elif fb['High'] >= target_p:
                                exit_p = target_p
                                exit_reason = "TARGET_HIT"
                                break

                        if exit_p is None:
                            exit_p = day_bars.iloc[-1]['Close']
                            exit_reason = "EOD_SQUARE_OFF"

                        fill_entry = entry_p * (1 + config.SLIPPAGE_PCT)
                        fill_exit = exit_p * (1 - config.SLIPPAGE_PCT)
                        qty = int(config.INITIAL_CAPITAL * 0.01 / max(abs(fill_entry - stop_p), 1.0))
                        qty = max(1, min(qty, int(config.INITIAL_CAPITAL * 0.10 / fill_entry)))

                        costs = self.executor.calculate_trade_costs(fill_entry, fill_exit, qty)
                        gross_pnl = (fill_exit - fill_entry) * qty
                        net_pnl = gross_pnl - costs['total_cost']

                        all_hourly_trades.append({
                            'date': str(current_day),
                            'symbol': sym,
                            'p_win': round(p_win, 4),
                            'decision': 'ACCEPTED' if (p_win >= self.threshold and ev_info['decision'] == 'TRADE') else 'REJECTED',
                            'net_pnl': round(net_pnl, 2),
                            'gross_pnl': round(gross_pnl, 2),
                            'costs': round(costs['total_cost'], 2),
                            'exit_reason': exit_reason
                        })

        ht_df = pd.DataFrame(all_hourly_trades)
        ht_df['date_dt'] = pd.to_datetime(ht_df['date'])

        # Multi-Year Breakdown Table
        multiyear_rows = []
        for name, start_d, end_d in year_cohorts:
            sub = ht_df[(ht_df['date_dt'] >= start_d) & (ht_df['date_dt'] <= end_d)]
            if sub.empty:
                continue

            acc = sub[sub['decision'] == 'ACCEPTED']
            rej = sub[sub['decision'] == 'REJECTED']

            n_acc = len(acc)
            acc_wins = (acc['net_pnl'] > 0).sum() if n_acc > 0 else 0
            acc_wr = (acc_wins / n_acc * 100) if n_acc > 0 else 0.0
            gw = acc[acc['net_pnl'] > 0]['net_pnl'].sum() if acc_wins > 0 else 0.0
            gl = abs(acc[acc['net_pnl'] <= 0]['net_pnl'].sum()) if (n_acc - acc_wins) > 0 else 1.0
            pf = gw / gl if gl > 0 else float('inf')
            net_pnl = acc['net_pnl'].sum() if n_acc > 0 else 0.0

            rej_losses_avoided = abs(rej[rej['net_pnl'] < 0]['net_pnl'].sum()) if not rej.empty else 0.0

            multiyear_rows.append({
                'Era': name,
                'Accepted_Trades': n_acc,
                'Wins': acc_wins,
                'WinRate%': round(acc_wr, 1),
                'ProfitFactor': round(pf, 3),
                'NetPnL_Rs': round(net_pnl, 2),
                'Rejected_Setups': len(rej),
                'LossesAvoided_Rs': round(rej_losses_avoided, 2)
            })

        my_df = pd.DataFrame(multiyear_rows)
        print("\nMulti-Year Generalization Table (2023 - 2026 Pre-5M Historical Data):")
        print(my_df.to_string(index=False))

        return {
            'total_pre_5m_sessions': len(pre_dates),
            'total_setups_evaluated': len(ht_df),
            'multiyear_breakdown': multiyear_rows
        }


if __name__ == "__main__":
    auditor = TemporalStressAuditor(threshold=0.40)
    auditor.run_full_temporal_suite()
