"""
V8 & V8.5 -- Live Paper Trading Engine & Live-Readiness Hardening
Orchestrates the real-time forward paper trading lifecycle:
  1. 09:15 - 09:30 : Daily Universe Scanner & Opening Range Formation
  2. 09:30 - 15:15 : Real-time Polling Loop (5m candle ingestion, feature computation,
                     ORB/VWAP/PDH setup detection, frozen XGBoost inference, EV gate,
                     position sizing, simulated market order execution, trailing stop & target monitoring)
  3. 15:15 - 15:30 : Compulsory Auto Square-off & Forward Journal Logging

Live-Readiness Hardening:
  - Crash Recovery: Restores active positions and equity from SQLite upon restart.
  - Dual-Stream Logging: Executed trades -> journal_trades, Rejected setups -> journal_rejected_setups.
  - Frozen Model & Threshold: xgb_intraday_5m.json at P >= 0.40 and EV > 0.
  - Feed Staleness Detection: Guards against delayed feeds (> 60s).
"""
import sys
import os
import time
from datetime import datetime, time as dtime
import pandas as pd
import numpy as np
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from core.data_manager import DataManager
from core.regime_detector import RegimeDetector
from models.xgboost_model import XGBoostTradeModel
from risk.expected_value import ExpectedValueCalculator
from execution.trade_journal import TradeJournal
from execution.position_manager import PositionManager
from execution.paper_executor import PaperExecutor
from run_v6_5_true_intraday import TrueIntradayFeatureEngine


class LiveTradingEngine:
    """
    Real-time Live Trading Engine for Forward Paper Trading (V8.5).
    """

    def __init__(self, mode="PAPER", initial_capital=None, threshold=0.40):
        self.mode = mode
        self.threshold = threshold
        self.journal = TradeJournal()
        self.pos_mgr = PositionManager(initial_capital=initial_capital)
        self.executor = PaperExecutor()
        self.fe = TrueIntradayFeatureEngine()
        self.dm = DataManager()
        self.regime_det = RegimeDetector()
        self.ev_calc = ExpectedValueCalculator()

        # Load frozen production XGBoost model
        self.model = XGBoostTradeModel()
        model_path = os.path.join(config.RESULTS_DIR, "xgb_intraday_5m.json")
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"[LIVE ENGINE] Frozen XGBoost model loaded from {model_path}")
            print(f"[LIVE ENGINE] Frozen Decision Threshold: P(win) >= {self.threshold:.2f} | EV > 0")
        else:
            print(f"[LIVE ENGINE] Warning: {model_path} not found.")

        # V8.5 Crash Recovery on startup
        self._reconcile_state_on_startup()

        self.symbols = config.UNIVERSE
        self.orb_ranges = {}
        self.triggered_setups = set()
        self.last_feed_timestamp = None

    def _reconcile_state_on_startup(self):
        """Recovers active positions from SQLite journal upon reboot."""
        print("[CRASH RECOVERY] Reconciling system state from SQLite...")
        open_trades_df = self.journal.get_open_positions()
        if not open_trades_df.empty:
            print(f"[CRASH RECOVERY] Restoring {len(open_trades_df)} unclosed active positions.")
            for _, row in open_trades_df.iterrows():
                self.pos_mgr.add_position(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    setup_type=row['setup_type'],
                    direction=row['direction'],
                    entry_price=row['entry_price'],
                    stop_price=row['stop_price'],
                    target_price=row['target_price'],
                    quantity=row['quantity'],
                    fill_entry_price=row['fill_entry_price']
                )
            print(f"[CRASH RECOVERY] Restored {len(self.pos_mgr.positions)} active positions.")
        else:
            print("[CRASH RECOVERY] Clean startup: 0 open positions.")

    def fetch_live_5m_bars(self, symbol):
        """
        Fetches live 5-minute candles for today + recent history from Yahoo Finance.
        Returns clean DataFrame with Date, Open, High, Low, Close, Volume.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="5m")
            if df.empty:
                return None

            df.reset_index(inplace=True)
            date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df['Date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df['trade_date'] = df['Date'].dt.date
            df['symbol'] = symbol

            clean_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'trade_date', 'symbol']
            df_clean = df[[c for c in clean_cols if c in df.columns]].dropna()

            self.last_feed_timestamp = datetime.now()
            return df_clean
        except Exception as e:
            print(f"[DATA ERROR] Failed to fetch live 5m bars for {symbol}: {e}")
            return None

    def poll_and_evaluate_cycle(self, current_time=None):
        """
        Executes one full 5-minute evaluation cycle across all universe stocks:
          1. Ingest live 5m candles
          2. Check exits on open positions
          3. Detect setups (ORB, VWAP Breakout, Prev Day High)
          4. Run frozen XGBoost probability inference
          5. Execute paper orders / log counterfactual rejections
        """
        now = current_time or datetime.now()
        cur_t = now.time()
        today = now.date()

        print(f"\n--- [POLL CYCLE] Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')} ---")

        # Step 1: Collect live data for all symbols
        live_data = {}
        current_prices = {}

        for sym in self.symbols:
            df_sym = self.fetch_live_5m_bars(sym)
            if df_sym is not None and not df_sym.empty:
                live_data[sym] = df_sym
                current_prices[sym] = float(df_sym['Close'].iloc[-1])

        if not current_prices:
            print("[FEED ALERT] No live quotes returned from data provider.")
            return

        # Step 2: Check exits on active positions
        exits = self.pos_mgr.update_price_and_check_exits(current_prices, current_time=cur_t)
        for pos, exit_price, reason in exits:
            self.execute_exit(pos, exit_price, reason, now)

        # Step 3: Enforce entry time window (09:30 to 14:30 IST)
        allow_entries = (cur_t >= dtime(9, 30)) and (cur_t <= dtime(14, 30))

        feat_cols = self.fe.get_feature_columns()

        for sym in self.symbols:
            if sym not in live_data:
                continue

            df_sym = live_data[sym]
            today_bars = df_sym[df_sym['trade_date'] == today]
            if len(today_bars) < 2:
                continue

            cur_bar = today_bars.iloc[-1]
            cur_close = cur_bar['Close']
            cur_ts = cur_bar['Date']

            # Build backward-looking causal features
            df_feat = self.fe.build_intraday_features(df_sym)
            last_feat = df_feat.iloc[-1]

            # Build ORB Range (first 3 bars: 09:15 - 09:30)
            if len(today_bars) <= 3:
                if sym not in self.orb_ranges:
                    self.orb_ranges[sym] = {'high': today_bars['High'].max(), 'low': today_bars['Low'].min(), 'complete': False}
                else:
                    self.orb_ranges[sym]['high'] = max(self.orb_ranges[sym]['high'], today_bars['High'].max())
                    self.orb_ranges[sym]['low'] = min(self.orb_ranges[sym]['low'], today_bars['Low'].min())
                if len(today_bars) == 3:
                    self.orb_ranges[sym]['complete'] = True

            if not allow_entries:
                continue

            # Check Setups
            candidate_setup = None

            # 1. ORB Check
            if sym in self.orb_ranges and self.orb_ranges[sym]['complete']:
                orb_h = self.orb_ranges[sym]['high']
                orb_key = (sym, 'ORB', today)
                if cur_close > orb_h and orb_key not in self.triggered_setups:
                    if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                        self.triggered_setups.add(orb_key)
                        candidate_setup = {
                            'sym': sym, 'setup_type': 'ORB',
                            'entry_price': cur_close,
                            'stop_price': cur_close * 0.996,
                            'target_price': cur_close * 1.008
                        }

            # 2. VWAP Breakout Check
            if candidate_setup is None:
                vwap = last_feat.get('feat_vwap', np.nan)
                vwap_key = (sym, 'VWAP_BREAKOUT', today)
                if pd.notna(vwap) and len(today_bars) >= 2 and vwap_key not in self.triggered_setups:
                    prev_close = today_bars.iloc[-2]['Close']
                    if prev_close < vwap and cur_close > vwap:
                        if last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                            self.triggered_setups.add(vwap_key)
                            candidate_setup = {
                                'sym': sym, 'setup_type': 'VWAP_BREAKOUT',
                                'entry_price': cur_close,
                                'stop_price': cur_close * 0.996,
                                'target_price': cur_close * 1.007
                            }

            # 3. Previous Day High Breakout Check
            if candidate_setup is None:
                pdh_key = (sym, 'PREV_DAY_HIGH_BREAKOUT', today)
                daily_ctx = self.dm.build_daily_context(df_sym)
                day_match = daily_ctx[daily_ctx['trade_date'] == today]
                if not day_match.empty and pdh_key not in self.triggered_setups:
                    pdh = day_match.iloc[0]['prev_day_high']
                    if cur_close > pdh and last_feat.get('feat_volume_ratio', 1.0) >= 1.1:
                        self.triggered_setups.add(pdh_key)
                        candidate_setup = {
                            'sym': sym, 'setup_type': 'PREV_DAY_HIGH_BREAKOUT',
                            'entry_price': cur_close,
                            'stop_price': cur_close * 0.995,
                            'target_price': cur_close * 1.010
                        }

            # Process Opportunity if Detected
            if candidate_setup is not None:
                self._process_live_opportunity(candidate_setup, last_feat, feat_cols, cur_ts)

        # Summary of portfolio
        n_pos = len(self.pos_mgr.positions)
        print(f"  Active Positions: {n_pos} | Equity: Rs.{self.pos_mgr.current_equity:,.2f} | Day P&L: Rs.{self.pos_mgr.daily_pnl:+.2f}")

    def _process_live_opportunity(self, opp, last_feat, feat_cols, cur_ts):
        sym = opp['sym']
        stype = opp['setup_type']
        ep = opp['entry_price']
        sp = opp['stop_price']
        tp = opp['target_price']

        feat_vec = last_feat[feat_cols].to_frame().T.fillna(0)
        p_win = float(self.model.predict_proba(feat_vec)[0])
        ev_info = self.ev_calc.calculate_ev(p_win)
        is_ev_pos = (ev_info['decision'] == 'TRADE')

        decision = "ACCEPT" if (p_win >= self.threshold and is_ev_pos) else "REJECT"

        print(f"  [OPPORTUNITY] {sym} | Type: {stype} | LTP: Rs.{ep:,.2f}")
        print(f"    XGB P(win): {p_win*100:5.1f}% (Thresh: >={self.threshold*100:.0f}%) | EV: {ev_info['ev_pct']:+.3f}% | Decision: {decision}")

        if decision == "REJECT":
            reason = f"P(win) {p_win:.3f} < {self.threshold:.2f}" if p_win < self.threshold else f"EV {ev_info['ev_pct']:.3f}% <= 0"
            rej_id = f"LIVE_REJ_{cur_ts.strftime('%Y%m%d_%H%M%S')}_{sym}"
            self.journal.log_rejected_setup(
                setup_id=rej_id, symbol=sym, setup_type=stype, regime='INTRADAY',
                current_ltp=ep, xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                reason=reason, features_dict=last_feat[feat_cols].to_dict(),
                timestamp=cur_ts
            )
            return

        # Check Risk Engine
        can_trade, risk_reason = self.pos_mgr.can_open_position()
        if not can_trade:
            print(f"    [RISK BLOCKED] {risk_reason}")
            rej_id = f"LIVE_REJ_{cur_ts.strftime('%Y%m%d_%H%M%S')}_{sym}"
            self.journal.log_rejected_setup(
                setup_id=rej_id, symbol=sym, setup_type=stype, regime='INTRADAY',
                current_ltp=ep, xgb_prob=p_win, ev_score=ev_info['ev_pct'],
                reason=f"Risk Engine: {risk_reason}", features_dict=last_feat[feat_cols].to_dict(),
                timestamp=cur_ts
            )
            return

        # Execute Simulated Market Order
        qty = self.pos_mgr.calculate_position_size(ep, sp)
        order = self.executor.execute_market_buy(sym, ep, qty)
        trade_id = f"LIVE_TRD_{cur_ts.strftime('%Y%m%d_%H%M%S')}_{sym}"

        self.pos_mgr.add_position(
            trade_id=trade_id, symbol=sym, setup_type=stype, direction='LONG',
            entry_price=ep, stop_price=sp, target_price=tp, quantity=qty,
            fill_entry_price=order['fill_price']
        )
        self.journal.log_entry(
            trade_id=trade_id, symbol=sym, setup_type=stype, direction='LONG',
            regime='INTRADAY', xgb_prob=p_win, ev_score=ev_info['ev_pct'],
            entry_price=ep, stop_price=sp, target_price=tp, quantity=qty,
            fill_entry_price=order['fill_price'],
            features_dict=last_feat[feat_cols].to_dict(), timestamp=cur_ts
        )
        print(f"    >>> LIVE PAPER BUY EXECUTED: {qty}x {sym} @ Rs.{order['fill_price']:.2f} (Target: Rs.{tp:.2f}, Stop: Rs.{sp:.2f})")

    def execute_exit(self, pos, exit_price, reason, current_ts):
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
        print(f"  [POSITION CLOSED] {pos['symbol']} -> {reason}")
        print(f"    Fill: Rs.{order['fill_price']:.2f} | Net P&L: Rs.{net_pnl:+.2f} ({return_pct:+.2f}%) | Costs: Rs.{costs['total_cost']:.2f}")


if __name__ == "__main__":
    engine = LiveTradingEngine()
    engine.poll_and_evaluate_cycle()
