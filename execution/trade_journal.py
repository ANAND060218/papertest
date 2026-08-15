"""
V8 -- Trade Journal
Logs every single trade decision, order fill, EV score, regime, and outcome to a persistent SQLite database.

Table Schema:
  trades:
    trade_id, timestamp, symbol, setup_type, direction, regime,
    xgb_probability, ev_score, entry_price, stop_price, target_price,
    quantity, fill_entry_price, exit_timestamp, exit_price,
    fill_exit_price, gross_pnl, brokerage, stt, stamp_duty, gst,
    slippage_cost, total_costs, net_pnl, return_pct, exit_reason,
    bars_held, features_snapshot_json, status
"""
import sys
import os
import sqlite3
import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


JOURNAL_DB_PATH = os.path.join(config.DATA_DIR, "trade_journal.db")


class TradeJournal:
    """
    Persistent SQLite Trade Journal for live paper trading and live audit trail.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or JOURNAL_DB_PATH
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal_trades (
            trade_id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            symbol TEXT,
            setup_type TEXT,
            direction TEXT,
            regime TEXT,
            xgb_probability REAL,
            ev_score REAL,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            quantity INTEGER,
            fill_entry_price REAL,
            exit_timestamp TIMESTAMP,
            exit_price REAL,
            fill_exit_price REAL,
            gross_pnl REAL,
            total_costs REAL,
            net_pnl REAL,
            return_pct REAL,
            exit_reason TEXT,
            bars_held INTEGER,
            features_json TEXT,
            status TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal_rejected_setups (
            setup_id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            symbol TEXT,
            setup_type TEXT,
            regime TEXT,
            current_ltp REAL,
            xgb_probability REAL,
            ev_score REAL,
            rejection_reason TEXT,
            features_json TEXT
        );
        """)

        conn.commit()
        conn.close()

    def log_rejected_setup(self, setup_id, symbol, setup_type, regime,
                            current_ltp, xgb_prob, ev_score, reason, features_dict=None, timestamp=None):
        """
        Logs a detected setup that was REJECTED by ML probability threshold or EV rule.
        Enables counterfactual evaluation during forward testing.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        feat_json = json.dumps(features_dict) if features_dict else "{}"
        ts_str = str(timestamp) if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
        INSERT OR REPLACE INTO journal_rejected_setups (
            setup_id, created_at, symbol, setup_type, regime,
            current_ltp, xgb_probability, ev_score, rejection_reason, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            setup_id, ts_str, symbol, setup_type, regime,
            float(current_ltp), float(xgb_prob) if xgb_prob else 0.0,
            float(ev_score) if ev_score else 0.0, reason, feat_json
        ))

        conn.commit()
        conn.close()

    def get_rejected_setups(self):
        """Returns all logged rejected setups."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM journal_rejected_setups ORDER BY created_at DESC", conn)
        conn.close()
        return df

    def log_entry(self, trade_id, symbol, setup_type, direction, regime,
                  xgb_prob, ev_score, entry_price, stop_price, target_price,
                  quantity, fill_entry_price, features_dict=None, timestamp=None):
        """
        Logs a new open position entry.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        feat_json = json.dumps(features_dict) if features_dict else "{}"
        ts_str = str(timestamp) if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
        INSERT OR REPLACE INTO journal_trades (
            trade_id, created_at, symbol, setup_type, direction, regime,
            xgb_probability, ev_score, entry_price, stop_price, target_price,
            quantity, fill_entry_price, status, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (
            trade_id, ts_str, symbol, setup_type, direction, regime,
            float(xgb_prob) if xgb_prob else 0.0, float(ev_score) if ev_score else 0.0,
            float(entry_price), float(stop_price), float(target_price),
            int(quantity), float(fill_entry_price), feat_json
        ))

        conn.commit()
        conn.close()

    def log_exit(self, trade_id, exit_timestamp, exit_price, fill_exit_price,
                 gross_pnl, total_costs, net_pnl, return_pct, exit_reason, bars_held=1):
        """
        Logs the close/exit of an open position with final P&L and costs.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        exit_ts_str = str(exit_timestamp)

        cursor.execute("""
        UPDATE journal_trades SET
            exit_timestamp = ?,
            exit_price = ?,
            fill_exit_price = ?,
            gross_pnl = ?,
            total_costs = ?,
            net_pnl = ?,
            return_pct = ?,
            exit_reason = ?,
            bars_held = ?,
            status = 'CLOSED'
        WHERE trade_id = ?
        """, (
            exit_ts_str, float(exit_price), float(fill_exit_price),
            float(gross_pnl), float(total_costs), float(net_pnl),
            float(return_pct), exit_reason, int(bars_held), trade_id
        ))

        conn.commit()
        conn.close()

    def get_open_positions(self):
        """Returns all currently OPEN trades."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM journal_trades WHERE status = 'OPEN'", conn)
        conn.close()
        return df

    def get_all_trades(self):
        """Returns entire trade history."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM journal_trades ORDER BY created_at DESC", conn)
        conn.close()
        return df

    def save_state(self, key, value):
        """Persists a system state key/value (e.g. current equity, kill switch status)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)
        """, (key, str(value), now_str))
        conn.commit()
        conn.close()

    def load_state(self, key, default=None):
        """Loads a persisted system state value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default
