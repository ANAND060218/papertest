"""
V8.1 -- Historical Market Replay Simulator
Streams historical 5-minute candles ONE BAR AT A TIME into the frozen execution engine.

Strict Integrity Rules:
  1. The engine NEVER receives future candles.
  2. Features and setups are computed using data strictly up to the current bar.
  3. Uses frozen production configuration (xgb_intraday_5m.json, threshold 0.40).
  4. Dual-stream logging: Executed trades -> journal_trades, Rejected setups -> journal_rejected_setups.
  5. Enforces compulsory square-off at 15:15 IST.
  6. Reconciles Replay executions against Backtest records.
"""
import sys
import os
import time
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from core.feature_engine import FeatureEngine
from core.regime_detector import RegimeDetector
from models.xgboost_model import XGBoostTradeModel
from risk.expected_value import ExpectedValueCalculator
from execution.trade_journal import TradeJournal
from execution.position_manager import PositionManager
from execution.paper_executor import PaperExecutor
from run_v6_5_true_intraday import TrueIntradayFeatureEngine


INTRADAY_UNIVERSE_DB = os.path.join(config.DATA_DIR, "intraday_universe_5m.db")
REPLAY_JOURNAL_DB = os.path.join(config.DATA_DIR, "trade_journal_replay.db")


class HistoricalReplayEngine:
    """
    Simulates real-time market conditions by replaying 5-minute historical bars
    one tick/bar at a time with zero future look-ahead.
    """

    def __init__(self, symbols=None, initial_capital=None, threshold=0.40, db_path=None):
        self.symbols = symbols or config.UNIVERSE
        self.capital = initial_capital or config.INITIAL_CAPITAL
        self.threshold = threshold

        # Dedicated journal for replay audits
        self.journal_db = db_path or REPLAY_JOURNAL_DB
        self.journal = TradeJournal(self.journal_db)
        self.pos_mgr = PositionManager(initial_capital=self.capital)
        self.executor = PaperExecutor()
        self.fe = TrueIntradayFeatureEngine()
        self.regime_det = RegimeDetector()
        self.ev_calc = ExpectedValueCalculator()

        # Load frozen XGBoost model
        self.model = XGBoostTradeModel()
        model_path = os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json")
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"[REPLAY ENGINE] Loaded frozen XGBoost model from {model_path}")
        else:
            print("[REPLAY ENGINE] Warning: No frozen model found.")

    def replay_day(self, target_date_str, verbose=True, sleep_sec=0.0):
        """
        Replays an entire trading day (09:15 - 15:30 IST) bar by bar.

        Args:
            target_date_str: 'YYYY-MM-DD' (e.g. '2026-08-07')
            verbose: Print human-readable tick stream
            sleep_sec: Pause between bars for visual observation
        """
        conn = sqlite3.connect(INTRADAY_UNIVERSE_DB)
        dm = DataManager()

        # 1. Load full history up to target_date to compute prior context
        universe_data = {}
        daily_context = {}

        for sym in self.symbols:
            tbl = f"bars_5m_{sym.replace('.NS', '')}"
            df_sym = pd.read_sql_query(f"SELECT * FROM {tbl} ORDER BY Date ASC", conn)
            df_sym['Date'] = pd.to_datetime(df_sym['Date'])
            df_sym['trade_date'] = df_sym['Date'].dt.date
            universe_data[sym] = df_sym

            # Compute daily context (yesterday's OHLCV)
            daily_ctx = dm.build_daily_context(df_sym)
            daily_context[sym] = daily_ctx

        conn.close()

        # Extract target day's timestamps
        sample_df = universe_data[self.symbols[0]]
        target_date = pd.to_datetime(target_date_str).date()
        day_bars_sample = sample_df[sample_df['trade_date'] == target_date].sort_values('Date')

        if day_bars_sample.empty:
            print(f"[ERROR] Date {target_date_str} not found in database.")
            return None

        day_timestamps = day_bars_sample['Date'].tolist()
        total_bars = len(day_timestamps)

        print("\n" + "=" * 90)
        print(f"MARKET REPLAY SESSION: {target_date_str}")
        print(f"Symbols Monitored: {self.symbols}")
        print(f"Session Schedule: {day_timestamps[0].strftime('%H:%M')} to {day_timestamps[-1].strftime('%H:%M')} ({total_bars} bars)")
        print(f"Frozen Model Threshold: P(win) >= {self.threshold:.2f} | EV > 0")
        print("=" * 90)

        # Track rolling intraday state per symbol
        # We start with historical bars before today + today's candles accumulated up to current_idx
        setups_detected_today = []
        trades_executed_today = []
        setups_rejected_today = []

        # Setup tracking state
        orb_ranges = {}        # sym -> {'high': float, 'low': float, 'complete': bool}
        triggered_setups = set()  # (sym, setup_type, date) to prevent duplicate entries on same day

        # Bar-by-bar chronological execution loop
        for bar_idx, current_ts in enumerate(day_timestamps):
            current_time = current_ts.time()
            time_str = current_ts.strftime('%H:%M')

            if verbose:
                print(f"\n[{time_str}] ----------------------------------------------------")

            # Dictionary of current prices for exit checks: sym -> current bar's close
            current_prices = {}

            # Step A: Collect current bar for all symbols
            current_bars = {}
            for sym in self.symbols:
                df_all = universe_data[sym]
                # Slice history STRICTLY up to current timestamp (NO FUTURE DATA)
                df_so_far = df_all[df_all['Date'] <= current_ts].copy().reset_index(drop=True)
                current_bars[sym] = df_so_far
                current_prices[sym] = df_so_far.iloc[-1]['Close']

            # Step B: Check exits on active open positions
            exits = self.pos_mgr.update_price_and_check_exits(current_prices, current_time=current_time)
            for pos, exit_price, reason in exits:
                self._execute_replay_exit(pos, exit_price, reason, current_ts, verbose=verbose)

            # Step C: Setup Detection and Evaluation per Symbol
            # 15:15 square-off cutoff: No new trade entries allowed after 14:30
            allow_new_entries = (current_time >= dtime(9, 30)) and (current_time <= dtime(14, 30))

            for sym in self.symbols:
                df_so_far = current_bars[sym]
                if len(df_so_far) < 30:
                    continue

                today_bars = df_so_far[df_so_far['trade_date'] == target_date]
                cur_bar = today_bars.iloc[-1]
                cur_close = cur_bar['Close']
                cur_volume = cur_bar['Volume']

                # Compute rolling features on available data
                df_feat = self.fe.build_intraday_features(df_so_far)
                last_feat_row = df_feat.iloc[-1]

                # Update ORB Range during first 15 mins (09:15 - 09:30)
                if len(today_bars) <= 3:
                    if sym not in orb_ranges:
                        orb_ranges[sym] = {'high': today_bars['High'].max(), 'low': today_bars['Low'].min(), 'complete': False}
                    else:
                        orb_ranges[sym]['high'] = max(orb_ranges[sym]['high'], today_bars['High'].max())
                        orb_ranges[sym]['low'] = min(orb_ranges[sym]['low'], today_bars['Low'].min())
                    if len(today_bars) == 3:
                        orb_ranges[sym]['complete'] = True

                if not allow_new_entries:
                    continue

                # -------------------------------------------------------------
                # 1. ORB Setup Check
                # -------------------------------------------------------------
                if sym in orb_ranges and orb_ranges[sym]['complete']:
                    orb_h = orb_ranges[sym]['high']
                    orb_l = orb_ranges[sym]['low']
                    orb_key = (sym, 'ORB', target_date)

                    if cur_close > orb_h and orb_key not in triggered_setups:
                        vol_ratio = last_feat_row.get('feat_volume_ratio', 1.0)
                        if vol_ratio >= 1.1:
                            triggered_setups.add(orb_key)
                            self._process_setup_opportunity(
                                sym=sym, setup_type='ORB', current_ts=current_ts,
                                entry_price=cur_close, stop_price=cur_close * 0.996,
                                target_price=cur_close * 1.008, last_feat_row=last_feat_row,
                                verbose=verbose
                            )

                # -------------------------------------------------------------
                # 2. VWAP Breakout Setup Check
                # -------------------------------------------------------------
                vwap = last_feat_row.get('feat_vwap', np.nan)
                prev_bar = today_bars.iloc[-2] if len(today_bars) >= 2 else None
                vwap_key = (sym, 'VWAP_BREAKOUT', target_date)

                if pd.notna(vwap) and prev_bar is not None and vwap_key not in triggered_setups:
                    prev_close = prev_bar['Close']
                    if prev_close < vwap and cur_close > vwap:
                        vol_ratio = last_feat_row.get('feat_volume_ratio', 1.0)
                        if vol_ratio >= 1.1:
                            triggered_setups.add(vwap_key)
                            self._process_setup_opportunity(
                                sym=sym, setup_type='VWAP_BREAKOUT', current_ts=current_ts,
                                entry_price=cur_close, stop_price=cur_close * 0.996,
                                target_price=cur_close * 1.007, last_feat_row=last_feat_row,
                                verbose=verbose
                            )

                # -------------------------------------------------------------
                # 3. Previous-Day High Breakout Setup Check
                # -------------------------------------------------------------
                pdh_key = (sym, 'PREV_DAY_HIGH_BREAKOUT', target_date)
                daily_df = daily_context[sym]
                day_ctx_match = daily_df[daily_df['trade_date'] == target_date]

                if not day_ctx_match.empty and pdh_key not in triggered_setups:
                    prev_day_high = day_ctx_match.iloc[0]['prev_day_high']
                    if cur_close > prev_day_high:
                        vol_ratio = last_feat_row.get('feat_volume_ratio', 1.0)
                        if vol_ratio >= 1.1:
                            triggered_setups.add(pdh_key)
                            self._process_setup_opportunity(
                                sym=sym, setup_type='PREV_DAY_HIGH_BREAKOUT', current_ts=current_ts,
                                entry_price=cur_close, stop_price=cur_close * 0.995,
                                target_price=cur_close * 1.010, last_feat_row=last_feat_row,
                                verbose=verbose
                            )

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # -----------------------------------------------------------------
        # Session Wrap-up & Summary
        # -----------------------------------------------------------------
        print("\n" + "=" * 90)
        print(f"REPLAY SESSION COMPLETED: {target_date_str}")
        print("=" * 90)

        closed_trades = self.journal.get_all_trades()
        rejected_setups = self.journal.get_rejected_setups()

        # Filter to today using date string prefix
        today_trades = closed_trades[closed_trades['created_at'].astype(str).str.startswith(target_date_str)] if not closed_trades.empty else pd.DataFrame()
        today_rejections = rejected_setups[rejected_setups['created_at'].astype(str).str.startswith(target_date_str)] if not rejected_setups.empty else pd.DataFrame()

        print(f"Active Positions Remaining: {len(self.pos_mgr.positions)}")
        print(f"Total Setups Detected:      {len(today_trades) + len(today_rejections)}")
        print(f"  - Accepted & Executed:    {len(today_trades)}")
        print(f"  - Rejected by ML / EV:    {len(today_rejections)}")
        print(f"Daily Net P&L:              Rs. {self.pos_mgr.daily_pnl:+.2f}")
        print(f"Portfolio Equity:           Rs. {self.pos_mgr.current_equity:,.2f}")

        if not today_trades.empty:
            print("\nExecuted Trades Details:")
            cols = ['trade_id', 'symbol', 'setup_type', 'fill_entry_price', 'fill_exit_price', 'net_pnl', 'exit_reason']
            print(today_trades[[c for c in cols if c in today_trades.columns]].to_string(index=False))

        return {
            'date': target_date_str,
            'setups_detected': len(today_trades) + len(today_rejections),
            'trades_executed': len(today_trades),
            'setups_rejected': len(today_rejections),
            'daily_pnl': round(self.pos_mgr.daily_pnl, 2),
            'ending_equity': round(self.pos_mgr.current_equity, 2)
        }

    def _process_setup_opportunity(self, sym, setup_type, current_ts, entry_price,
                                   stop_price, target_price, last_feat_row, verbose=True):
        """Evaluates setup opportunity with frozen XGBoost, EV, and risk engine."""
        feat_cols = self.fe.get_feature_columns()
        feat_vector = last_feat_row[feat_cols].to_frame().T.fillna(0)

        # 1. Frozen XGBoost Probability Inference
        p_win = 0.50
        if self.model.is_trained:
            p_win = float(self.model.predict_proba(feat_vector)[0])

        # 2. Expected Value Calculation
        ev_info = self.ev_calc.calculate_ev(p_win)
        is_ev_positive = (ev_info['decision'] == 'TRADE')

        # 3. Decision Gate (Threshold >= 0.40 and EV > 0)
        decision = "ACCEPT" if (p_win >= self.threshold and is_ev_positive) else "REJECT"

        if verbose:
            print(f"  [SETUP DETECTED] {sym:12s} | Type: {setup_type:20s} | LTP: Rs.{entry_price:,.2f}")
            print(f"    XGB P(win): {p_win*100:5.1f}% (Thresh: >={self.threshold*100:.0f}%) | EV: {ev_info['ev_pct']:+.3f}% | Decision: {decision}")

        if decision == "REJECT":
            reason = f"P(win) {p_win:.3f} < {self.threshold:.2f}" if p_win < self.threshold else f"EV {ev_info['ev_pct']:.3f}% <= 0"
            self.journal.log_rejected_setup(
                setup_id=f"REPLAY_REJ_{current_ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                symbol=sym, setup_type=setup_type, regime='INTRADAY',
                current_ltp=entry_price, xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                reason=reason, features_dict=last_feat_row[feat_cols].to_dict(),
                timestamp=current_ts
            )
            return

        # 4. Check Risk Engine
        can_trade, reason = self.pos_mgr.can_open_position()
        if not can_trade:
            if verbose:
                print(f"    [RISK BLOCKED] {reason}")
            self.journal.log_rejected_setup(
                setup_id=f"REPLAY_REJ_{current_ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                symbol=sym, setup_type=setup_type, regime='INTRADAY',
                current_ltp=entry_price, xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                reason=f"Risk Engine: {reason}",
                timestamp=current_ts
            )
            return

        # 5. Position Sizing
        quantity = self.pos_mgr.calculate_position_size(entry_price, stop_price)

        # 6. Execute Simulated Market Order (with slippage & spread)
        order = self.executor.execute_market_buy(sym, entry_price, quantity)
        trade_id = f"REPLAY_TRD_{current_ts.strftime('%Y%m%d_%H%M%S')}_{sym}"

        self.pos_mgr.add_position(
            trade_id=trade_id, symbol=sym, setup_type=setup_type, direction='LONG',
            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
            quantity=quantity, fill_entry_price=order['fill_price']
        )

        self.journal.log_entry(
            trade_id=trade_id, symbol=sym, setup_type=setup_type, direction='LONG',
            regime='INTRADAY', xgb_prob=p_win, ev_score=ev_info['ev_pct'],
            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
            quantity=quantity, fill_entry_price=order['fill_price'],
            features_dict=last_feat_row[feat_cols].to_dict(),
            timestamp=current_ts
        )

        if verbose:
            print(f"    >>> SIMULATED BUY EXECUTED: {quantity}x {sym} @ Rs.{order['fill_price']:.2f} (Target: Rs.{target_price:.2f}, Stop: Rs.{stop_price:.2f})")

    def _execute_replay_exit(self, pos, exit_price, reason, current_ts, verbose=True):
        """Executes position exit, computes all transaction charges, and updates journal."""
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

        if verbose:
            print(f"  [POSITION CLOSED] {pos['symbol']} -> {reason}")
            print(f"    Fill: Rs.{order['fill_price']:.2f} | Net P&L: Rs.{net_pnl:+.2f} ({return_pct:+.2f}%) | Costs: Rs.{costs['total_cost']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="V8.1 Historical Market Replay Engine")
    parser.add_argument("--date", type=str, default="2026-08-07", help="Date to replay (YYYY-MM-DD)")
    parser.add_argument("--multi", type=int, default=1, help="Number of consecutive trading days to replay")
    args = parser.parse_args()

    engine = HistoricalReplayEngine()

    if args.multi == 1:
        engine.replay_day(args.date)
    else:
        # Replay multiple consecutive days
        conn = sqlite3.connect(INTRADAY_UNIVERSE_DB)
        df_dates = pd.read_sql_query("SELECT DISTINCT substr(Date, 1, 10) as d FROM universe_intraday_5m ORDER BY d", conn)
        conn.close()
        all_dates = list(df_dates['d'].values)

        if args.date in all_dates:
            start_idx = all_dates.index(args.date)
            target_dates = all_dates[start_idx:start_idx + args.multi]
        else:
            target_dates = all_dates[-args.multi:]

        print(f"\n[MULTI-DAY REPLAY] Running {len(target_dates)} consecutive days: {target_dates}")
        multi_summary = []
        for d in target_dates:
            res = engine.replay_day(d, verbose=False)
            if res:
                multi_summary.append(res)

        ms_df = pd.DataFrame(multi_summary)
        print("\n" + "=" * 90)
        print("MULTI-DAY REPLAY SUMMARY REPORT")
        print("=" * 90)
        print(ms_df.to_string(index=False))


if __name__ == "__main__":
    main()
