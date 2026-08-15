import sqlite3
import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'paper_trading.db')

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Strategies Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id TEXT PRIMARY KEY,
            initial_capital REAL NOT NULL,
            current_cash REAL NOT NULL,
            status TEXT DEFAULT 'ACTIVE'
        )
    ''')
    
    # Orders Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'BUY' or 'SELL'
            quantity INTEGER NOT NULL,
            fill_price REAL NOT NULL,
            gross_value REAL NOT NULL,
            brokerage REAL NOT NULL,
            exchange_fee REAL NOT NULL,
            gst REAL NOT NULL,
            stt REAL NOT NULL,
            stamp_duty REAL NOT NULL,
            sebi_fee REAL NOT NULL,
            total_costs REAL NOT NULL,
            net_value REAL NOT NULL,
            notes TEXT,
            FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
        )
    ''')
    
    # Positions Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL, -- positive for long, negative for short
            avg_entry_price REAL NOT NULL,
            realized_pnl REAL DEFAULT 0.0,
            FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
        )
    ''')
    
    # V19 Specific: Pair Monitor Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS v19_pairs (
            pair_id TEXT PRIMARY KEY, -- e.g. "HDFCBANK/KOTAKBANK"
            stock_a TEXT,
            stock_b TEXT,
            hedge_ratio REAL,
            half_life REAL,
            last_z_score REAL,
            status TEXT, -- 'ACTIVE' or 'DISABLED'
            updated_at TEXT
        )
    ''')
    
    # Equity Curve Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS equity_curve (
            date TEXT,
            strategy_id TEXT,
            cash REAL,
            holdings_value REAL,
            total_equity REAL,
            drawdown_pct REAL,
            PRIMARY KEY (date, strategy_id),
            FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# ─── Data Access Helpers ────────────────────────────────────────────────

def register_strategy(strategy_id, initial_capital):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO strategies (strategy_id, initial_capital, current_cash, status)
        VALUES (?, ?, ?, 'ACTIVE')
    ''', (strategy_id, initial_capital, initial_capital))
    conn.commit()
    conn.close()

def get_strategy_cash(strategy_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT current_cash FROM strategies WHERE strategy_id = ?', (strategy_id,))
    row = c.fetchone()
    conn.close()
    return row['current_cash'] if row else 0.0

def update_strategy_cash(strategy_id, cash_delta):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE strategies SET current_cash = current_cash + ? WHERE strategy_id = ?', 
              (cash_delta, strategy_id))
    conn.commit()
    conn.close()

def get_positions(strategy_id):
    conn = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM positions WHERE strategy_id = ? AND quantity != 0', 
                           conn, params=(strategy_id,))
    conn.close()
    return df

def get_position(strategy_id, symbol):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM positions WHERE strategy_id = ? AND symbol = ?', (strategy_id, symbol))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def update_position(strategy_id, symbol, qty_change, fill_price):
    """
    Updates position. Calculates realized P&L on closing trades.
    qty_change is positive for buy, negative for sell.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT * FROM positions WHERE strategy_id = ? AND symbol = ?', (strategy_id, symbol))
    row = c.fetchone()
    
    if not row:
        # New position
        c.execute('''
            INSERT INTO positions (strategy_id, symbol, quantity, avg_entry_price, realized_pnl)
            VALUES (?, ?, ?, ?, 0.0)
        ''', (strategy_id, symbol, qty_change, fill_price))
    else:
        current_qty = row['quantity']
        avg_price = row['avg_entry_price']
        realized_pnl = row['realized_pnl']
        
        new_qty = current_qty + qty_change
        
        # Determine if we are adding to position or closing/reversing
        is_closing = False
        if (current_qty > 0 and qty_change < 0) or (current_qty < 0 and qty_change > 0):
            is_closing = True
            
        if is_closing:
            # Closing trade: calculate realized P&L
            closed_qty = min(abs(current_qty), abs(qty_change))
            if current_qty > 0:
                # Long closing
                pnl = (fill_price - avg_price) * closed_qty
            else:
                # Short closing
                pnl = (avg_price - fill_price) * closed_qty
            realized_pnl += pnl
            
            # If reversed, update avg price
            if (current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0):
                avg_price = fill_price
            elif new_qty == 0:
                avg_price = 0.0
        else:
            # Adding to position: calculate new avg price
            if current_qty == 0:
                avg_price = fill_price
            else:
                total_value = (abs(current_qty) * avg_price) + (abs(qty_change) * fill_price)
                avg_price = total_value / abs(new_qty)
        
        c.execute('''
            UPDATE positions 
            SET quantity = ?, avg_entry_price = ?, realized_pnl = ?
            WHERE position_id = ?
        ''', (new_qty, avg_price, realized_pnl, row['position_id']))
        
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print(f"Database initialized at {DB_PATH}")
