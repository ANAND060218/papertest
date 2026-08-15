"""
V8.3 -- Out-of-Sample Generalization Replay on Unseen Stocks Universe
Evaluates the SAME frozen XGBoost model and SAME 0.40 threshold on a completely
unseen basket of 6 liquid stocks (AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC).

Scientific Objective:
  Tests whether the frozen model's ability to filter false breakouts generalizes
  to stocks and sectors that were never in its training or validation datasets.
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
from fetch_unseen_universe import UNSEEN_UNIVERSE, UNSEEN_DB_PATH


UNSEEN_JOURNAL_DB = os.path.join(config.DATA_DIR, "trade_journal_unseen_replay.db")
UNSEEN_REPORT_JSON = os.path.join(config.RESULTS_DIR, "v8_3_unseen_universe_replay_report.json")
UNSEEN_SUMMARY_CSV = os.path.join(config.RESULTS_DIR, "v8_3_unseen_universe_summary.csv")


class UnseenUniverseReplaySuite:
    """
    Simulates bar-by-bar market replay across 6 completely unseen stocks.
    """

    def __init__(self, threshold=0.40):
        self.threshold = threshold
        self.symbols = UNSEEN_UNIVERSE
        self.initial_capital = config.INITIAL_CAPITAL

        if os.path.exists(UNSEEN_JOURNAL_DB):
            os.remove(UNSEEN_JOURNAL_DB)

        self.journal = TradeJournal(UNSEEN_JOURNAL_DB)
        self.pos_mgr = PositionManager(initial_capital=self.initial_capital)
        self.executor = PaperExecutor()
        self.fe = TrueIntradayFeatureEngine()
        self.ev_calc = ExpectedValueCalculator()

        # Load frozen XGBoost model
        self.model = XGBoostTradeModel()
        model_path = os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json")
        self.model.load(model_path)
        print(f"[UNSEEN SUITE] Loaded frozen XGBoost model from {model_path}")
        print(f"[UNSEEN SUITE] Frozen threshold: P(win) >= {self.threshold:.2f} | EV > 0")

    def run_all_days(self):
        conn = sqlite3.connect(UNSEEN_DB_PATH)
        dm = DataManager()

        universe_data = {}
        daily_context = {}
        features_data = {}

        print("\n[UNSEEN SUITE] Ingesting unseen stocks and computing causal intraday features...")
        for sym in self.symbols:
            tbl = f"bars_5m_{sym.replace('.NS', '')}"
            df_sym = pd.read_sql_query(f"SELECT * FROM {tbl} ORDER BY Date ASC", conn)
            df_sym['Date'] = pd.to_datetime(df_sym['Date'])
            df_sym['trade_date'] = df_sym['Date'].dt.date
            universe_data[sym] = df_sym

            daily_ctx = dm.build_daily_context(df_sym)
            daily_context[sym] = daily_ctx

            df_feat = self.fe.build_intraday_features(df_sym)
            features_data[sym] = df_feat

        conn.close()

        sample_df = universe_data[self.symbols[0]]
        trading_dates = sorted(sample_df['trade_date'].unique())
        total_days = len(trading_dates)
        print(f"[UNSEEN SUITE] Replaying {total_days} sessions ({trading_dates[0]} to {trading_dates[-1]}) on UNSEEN stocks...")

        daily_results = []
        all_rejected_cfs = []
        feat_cols = self.fe.get_feature_columns()

        start_time = time.time()

        for day_idx, current_day in enumerate(trading_dates, 1):
            date_str = str(current_day)
            self.pos_mgr.start_new_day(current_date=current_day)

            day_ts_list = sample_df[sample_df['trade_date'] == current_day]['Date'].tolist()
            if not day_ts_list:
                continue

            orb_ranges = {}
            triggered_setups = set()
            day_setups_detected = 0
            day_trades_executed = 0
            day_setups_rejected = 0

            for current_ts in day_ts_list:
                current_time = current_ts.time()

                # A. Monitor Open Positions
                current_prices = {}
                for sym in self.symbols:
                    sym_df = universe_data[sym]
                    row_match = sym_df[sym_df['Date'] == current_ts]
                    if not row_match.empty:
                        current_prices[sym] = row_match.iloc[0]['Close']

                exits = self.pos_mgr.update_price_and_check_exits(current_prices, current_time=current_time)
                for pos, exit_price, reason in exits:
                    self._execute_position_exit(pos, exit_price, reason, current_ts)

                # B. Setup Detection (09:30 to 14:30)
                allow_new_entries = (current_time >= dtime(9, 30)) and (current_time <= dtime(14, 30))

                for sym in self.symbols:
                    sym_df = universe_data[sym]
                    sym_feat = features_data[sym]

                    df_today = sym_df[(sym_df['trade_date'] == current_day) & (sym_df['Date'] <= current_ts)]
                    if df_today.empty:
                        continue

                    cur_bar = df_today.iloc[-1]
                    cur_close = cur_bar['Close']

                    feat_row = sym_feat[sym_feat['Date'] == current_ts]
                    if feat_row.empty:
                        continue
                    last_feat = feat_row.iloc[0]

                    # ORB Range
                    if len(df_today) <= 3:
                        if sym not in orb_ranges:
                            orb_ranges[sym] = {'high': df_today['High'].max(), 'low': df_today['Low'].min(), 'complete': False}
                        else:
                            orb_ranges[sym]['high'] = max(orb_ranges[sym]['high'], df_today['High'].max())
                            orb_ranges[sym]['low'] = min(orb_ranges[sym]['low'], df_today['Low'].min())
                        if len(df_today) == 3:
                            orb_ranges[sym]['complete'] = True

                    if not allow_new_entries:
                        continue

                    candidate_setup = None

                    # 1. ORB Check
                    if sym in orb_ranges and orb_ranges[sym]['complete']:
                        orb_h = orb_ranges[sym]['high']
                        orb_key = (sym, 'ORB', current_day)
                        if cur_close > orb_h and orb_key not in triggered_setups:
                            if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                                triggered_setups.add(orb_key)
                                candidate_setup = {
                                    'sym': sym, 'setup_type': 'ORB',
                                    'entry_price': cur_close,
                                    'stop_price': cur_close * 0.996,
                                    'target_price': cur_close * 1.008
                                }

                    # 2. VWAP Breakout Check
                    if candidate_setup is None:
                        vwap = last_feat.get('feat_vwap', np.nan)
                        vwap_key = (sym, 'VWAP_BREAKOUT', current_day)
                        if pd.notna(vwap) and len(df_today) >= 2 and vwap_key not in triggered_setups:
                            prev_close = df_today.iloc[-2]['Close']
                            if prev_close < vwap and cur_close > vwap:
                                if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                                    triggered_setups.add(vwap_key)
                                    candidate_setup = {
                                        'sym': sym, 'setup_type': 'VWAP_BREAKOUT',
                                        'entry_price': cur_close,
                                        'stop_price': cur_close * 0.996,
                                        'target_price': cur_close * 1.007
                                    }

                    # 3. Previous Day High Breakout Check
                    if candidate_setup is None:
                        pdh_key = (sym, 'PREV_DAY_HIGH_BREAKOUT', current_day)
                        daily_df = daily_context[sym]
                        day_ctx_match = daily_df[daily_df['trade_date'] == current_day]
                        if not day_ctx_match.empty and pdh_key not in triggered_setups:
                            prev_day_high = day_ctx_match.iloc[0]['prev_day_high']
                            if cur_close > prev_day_high:
                                if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                                    triggered_setups.add(pdh_key)
                                    candidate_setup = {
                                        'sym': sym, 'setup_type': 'PREV_DAY_HIGH_BREAKOUT',
                                        'entry_price': cur_close,
                                        'stop_price': cur_close * 0.995,
                                        'target_price': cur_close * 1.010
                                    }

                    if candidate_setup is not None:
                        day_setups_detected += 1
                        setup_sym = candidate_setup['sym']
                        stype = candidate_setup['setup_type']
                        ep = candidate_setup['entry_price']
                        sp = candidate_setup['stop_price']
                        tp = candidate_setup['target_price']

                        feat_vec = last_feat[feat_cols].to_frame().T.fillna(0)
                        p_win = float(self.model.predict_proba(feat_vec)[0])
                        ev_info = self.ev_calc.calculate_ev(p_win)
                        is_ev_pos = (ev_info['decision'] == 'TRADE')

                        decision = "ACCEPT" if (p_win >= self.threshold and is_ev_pos) else "REJECT"

                        if decision == "ACCEPT":
                            can_trade, risk_reason = self.pos_mgr.can_open_position()
                            if can_trade:
                                day_trades_executed += 1
                                qty = self.pos_mgr.calculate_position_size(ep, sp)
                                order = self.executor.execute_market_buy(setup_sym, ep, qty)
                                trd_id = f"UNSEEN_TRD_{current_ts.strftime('%Y%m%d_%H%M%S')}_{setup_sym}"

                                self.pos_mgr.add_position(
                                    trade_id=trd_id, symbol=setup_sym, setup_type=stype, direction='LONG',
                                    entry_price=ep, stop_price=sp, target_price=tp, quantity=qty,
                                    fill_entry_price=order['fill_price']
                                )
                                self.journal.log_entry(
                                    trade_id=trd_id, symbol=setup_sym, setup_type=stype, direction='LONG',
                                    regime='INTRADAY', xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                                    entry_price=ep, stop_price=sp, target_price=tp, quantity=qty,
                                    fill_entry_price=order['fill_price'],
                                    features_dict=last_feat[feat_cols].to_dict(), timestamp=current_ts
                                )
                            else:
                                decision = "REJECT"

                        if decision == "REJECT":
                            day_setups_rejected += 1
                            reason = f"P(win) {p_win:.3f} < {self.threshold:.2f}" if p_win < self.threshold else f"EV {ev_info['ev_pct']:.3f}% <= 0"
                            rej_id = f"UNSEEN_REJ_{current_ts.strftime('%Y%m%d_%H%M%S')}_{setup_sym}"
                            self.journal.log_rejected_setup(
                                setup_id=rej_id, symbol=setup_sym, setup_type=stype, regime='INTRADAY',
                                current_ltp=ep, xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                                reason=reason, features_dict=last_feat[feat_cols].to_dict(),
                                timestamp=current_ts
                            )

                            cf_res = self._simulate_counterfactual_outcome(
                                sym_df=universe_data[setup_sym],
                                entry_ts=current_ts, current_day=current_day,
                                entry_price=ep, stop_price=sp, target_price=tp
                            )
                            cf_res.update({
                                'setup_id': rej_id, 'timestamp': current_ts,
                                'symbol': setup_sym, 'setup_type': stype,
                                'xgb_probability': round(p_win, 4), 'ev_pct': round(ev_info['ev_pct'], 4),
                                'rejection_reason': reason
                            })
                            all_rejected_cfs.append(cf_res)

            daily_results.append({
                'date': date_str,
                'setups_detected': day_setups_detected,
                'trades_executed': day_trades_executed,
                'setups_rejected': day_setups_rejected,
                'daily_pnl': round(self.pos_mgr.daily_pnl, 2),
                'ending_equity': round(self.pos_mgr.current_equity, 2)
            })

            if day_idx % 10 == 0 or day_idx == total_days:
                print(f"  [Progress] Replayed Day {day_idx:2d}/{total_days} ({date_str}) | Executed: {day_trades_executed} | Rejected: {day_setups_rejected} | Equity: Rs.{self.pos_mgr.current_equity:,.2f}")

        elapsed = time.time() - start_time
        print(f"\n[UNSEEN SUITE] Replay of unseen universe completed in {elapsed:.2f} seconds.")

        return self._generate_report(daily_results, all_rejected_cfs)

    def _execute_position_exit(self, pos, exit_price, reason, current_ts):
        trade_id = pos['trade_id']
        order = self.executor.execute_market_sell(pos['symbol'], exit_price, pos['quantity'])

        costs = self.executor.calculate_trade_costs(pos['fill_entry_price'], order['fill_price'], pos['quantity'])
        gross_pnl = (order['fill_price'] - pos['fill_entry_price']) * pos['quantity']
        net_pnl = gross_pnl - costs['total_cost']
        return_pct = (net_pnl / (pos['fill_entry_price'] * pos['quantity'])) * 100

        self.pos_mgr.close_position(trade_id, net_pnl)
        self.journal.log_exit(
            trade_id=trade_id, exit_timestamp=current_ts, exit_price=exit_price,
            fill_exit_price=order['fill_price'], gross_pnl=gross_pnl,
            total_costs=costs['total_cost'], net_pnl=net_pnl, return_pct=return_pct,
            exit_reason=reason, bars_held=pos['bars_held']
        )

    def _simulate_counterfactual_outcome(self, sym_df, entry_ts, current_day, entry_price, stop_price, target_price):
        future_bars = sym_df[(sym_df['trade_date'] == current_day) & (sym_df['Date'] > entry_ts)].sort_values('Date')
        fill_entry = entry_price * (1 + config.SLIPPAGE_PCT)
        qty = self.pos_mgr.calculate_position_size(entry_price, stop_price)

        exit_price = None
        exit_reason = None
        bars_held = 0

        for _, bar in future_bars.iterrows():
            bars_held += 1
            h, l, c = bar['High'], bar['Low'], bar['Close']
            bar_time = bar['Date'].time()

            if l <= stop_price:
                exit_price = stop_price
                exit_reason = "STOP_HIT"
                break
            elif h >= target_price:
                exit_price = target_price
                exit_reason = "TARGET_HIT"
                break
            elif bar_time >= dtime(15, 15):
                exit_price = c
                exit_reason = "INTRADAY_SQUARE_OFF"
                break

        if exit_price is None and not future_bars.empty:
            exit_price = future_bars.iloc[-1]['Close']
            exit_reason = "EOD_FORCE_CLOSE"
        elif exit_price is None:
            exit_price = entry_price
            exit_reason = "NO_FUTURE_BARS"

        fill_exit = exit_price * (1 - config.SLIPPAGE_PCT)
        costs = self.executor.calculate_trade_costs(fill_entry, fill_exit, qty)
        gross_pnl = (fill_exit - fill_entry) * qty
        net_pnl = gross_pnl - costs['total_cost']
        ret_pct = (net_pnl / (fill_entry * qty)) * 100

        return {
            'cf_entry_price': entry_price,
            'cf_fill_entry': round(fill_entry, 2),
            'cf_fill_exit': round(fill_exit, 2),
            'cf_gross_pnl': round(gross_pnl, 2),
            'cf_total_costs': round(costs['total_cost'], 2),
            'cf_net_pnl': round(net_pnl, 2),
            'cf_return_pct': round(ret_pct, 2),
            'cf_exit_reason': exit_reason,
            'cf_bars_held': bars_held,
            'cf_outcome': 'WIN' if net_pnl > 0 else 'LOSS'
        }

    def _generate_report(self, daily_results, rejected_cfs):
        conn = sqlite3.connect(UNSEEN_JOURNAL_DB)
        accepted_df = pd.read_sql_query("SELECT * FROM journal_trades WHERE status = 'CLOSED'", conn)
        conn.close()

        rejected_df = pd.DataFrame(rejected_cfs)

        def compute_stats(df, pnl_col):
            if df.empty or pnl_col not in df.columns:
                return {'count': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'gross_win_rs': 0.0, 'gross_loss_rs': 0.0, 'net_pnl_rs': 0.0, 'profit_factor': 0.0, 'expectancy_rs': 0.0}
            wins = df[df[pnl_col] > 0]
            losses = df[df[pnl_col] <= 0]
            total = len(df)
            gw = wins[pnl_col].sum() if not wins.empty else 0.0
            gl = abs(losses[pnl_col].sum()) if not losses.empty else 1.0
            pf = gw / gl if gl > 0 else float('inf')
            return {
                'count': total,
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': round((len(wins) / total) * 100, 2) if total > 0 else 0.0,
                'gross_win_rs': round(gw, 2),
                'gross_loss_rs': round(gl, 2),
                'net_pnl_rs': round(df[pnl_col].sum(), 2),
                'profit_factor': round(pf, 3),
                'expectancy_rs': round(df[pnl_col].mean(), 2)
            }

        acc_stats = compute_stats(accepted_df, 'net_pnl')
        rej_stats = compute_stats(rejected_df, 'cf_net_pnl')

        unfiltered_pnl = acc_stats['net_pnl_rs'] + rej_stats['net_pnl_rs']
        unfiltered_total = acc_stats['count'] + rej_stats['count']
        unfiltered_wins = acc_stats['wins'] + rej_stats['wins']
        unfiltered_wr = (unfiltered_wins / unfiltered_total * 100) if unfiltered_total > 0 else 0.0

        print("\n" + "=" * 100)
        print("V8.3 OUT-OF-SAMPLE GENERALIZATION REPLAY: UNSEEN STOCKS UNIVERSE")
        print("Stocks: AXISBANK, KOTAKBANK, BHARTIARTL, LT, WIPRO, ITC")
        print("=" * 100)

        matrix_data = [
            {"Stream": "ACCEPTED TRADES (P >= 0.40)", "Setups": acc_stats['count'], "Wins": acc_stats['wins'], "Losses": acc_stats['losses'], "WinRate%": f"{acc_stats['win_rate']:.1f}%", "ProfitFactor": f"{acc_stats['profit_factor']:.3f}", "NetPnL_Rs": f"Rs.{acc_stats['net_pnl_rs']:+,.2f}", "Expectancy_Rs": f"Rs.{acc_stats['expectancy_rs']:+.2f}"},
            {"Stream": "REJECTED SETUPS (P < 0.40)", "Setups": rej_stats['count'], "Wins": rej_stats['wins'], "Losses": rej_stats['losses'], "WinRate%": f"{rej_stats['win_rate']:.1f}%", "ProfitFactor": f"{rej_stats['profit_factor']:.3f}", "NetPnL_Rs": f"Rs.{rej_stats['net_pnl_rs']:+,.2f}", "Expectancy_Rs": f"Rs.{rej_stats['expectancy_rs']:+.2f}"},
            {"Stream": "UNFILTERED BASELINE (All)", "Setups": unfiltered_total, "Wins": unfiltered_wins, "Losses": unfiltered_total - unfiltered_wins, "WinRate%": f"{unfiltered_wr:.1f}%", "ProfitFactor": f"{(acc_stats['gross_win_rs'] + rej_stats['gross_win_rs']) / max(acc_stats['gross_loss_rs'] + rej_stats['gross_loss_rs'], 1):.3f}", "NetPnL_Rs": f"Rs.{unfiltered_pnl:+,.2f}", "Expectancy_Rs": f"Rs.{unfiltered_pnl/max(unfiltered_total,1):+.2f}"}
        ]
        matrix_df = pd.DataFrame(matrix_data)
        print(matrix_df.to_string(index=False))

        # Stock Breakdown
        print("\n--- Stock-Wise Breakdown on Unseen Universe ---")
        stock_comparison = []
        for sym in self.symbols:
            acc_sym = accepted_df[accepted_df['symbol'] == sym] if not accepted_df.empty else pd.DataFrame()
            rej_sym = rejected_df[rejected_df['symbol'] == sym] if not rejected_df.empty else pd.DataFrame()
            sym_acc = compute_stats(acc_sym, 'net_pnl')
            sym_rej = compute_stats(rej_sym, 'cf_net_pnl')
            stock_comparison.append({
                'Symbol': sym,
                'Accepted_Count': sym_acc['count'], 'Accepted_WinRate%': sym_acc['win_rate'], 'Accepted_PF': sym_acc['profit_factor'], 'Accepted_NetPnL': sym_acc['net_pnl_rs'],
                'Rejected_Count': sym_rej['count'], 'Rejected_WinRate%': sym_rej['win_rate'], 'Rejected_PF': sym_rej['profit_factor'], 'Rejected_NetPnL': sym_rej['net_pnl_rs']
            })
        stock_comp_df = pd.DataFrame(stock_comparison)
        print(stock_comp_df.to_string(index=False))

        capital_saved = abs(rej_stats['net_pnl_rs']) if rej_stats['net_pnl_rs'] < 0 else 0.0
        print(f"\n[SCIENTIFIC CONCLUSION] Capital Protected on Unseen Universe: Rs.{capital_saved:,.2f} of gross losses filtered out.")

        matrix_df.to_csv(UNSEEN_SUMMARY_CSV, index=False)
        report_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
            'dataset': 'UNSEEN_STOCKS_UNIVERSE',
            'symbols': self.symbols,
            'total_trading_days': len(daily_results),
            'accepted_trades_stats': acc_stats,
            'rejected_setups_stats': rej_stats,
            'capital_saved_rs': capital_saved,
            'stock_comparison': stock_comparison
        }

        with open(UNSEEN_REPORT_JSON, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n[REPORT] Saved unseen universe JSON report to: {UNSEEN_REPORT_JSON}")
        print(f"[SUMMARY] Saved unseen universe CSV summary to:   {UNSEEN_SUMMARY_CSV}")

        return report_data


if __name__ == "__main__":
    suite = UnseenUniverseReplaySuite(threshold=0.40)
    suite.run_all_days()
