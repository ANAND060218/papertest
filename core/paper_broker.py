import sys, os
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.database import (
    get_db_connection, get_strategy_cash, update_strategy_cash, update_position
)

def calculate_equity_costs(price, qty, is_buy, is_intraday=False):
    """
    Computes precise 2026 transaction costs for an Indian equity order.
    """
    turnover = price * qty
    brokerage = 20.0
    exchange = turnover * 0.00035
    gst = (brokerage + exchange) * 0.18
    
    if is_intraday:
        stt = turnover * 0.00025 if not is_buy else 0.0
    else:
        stt = turnover * 0.001
        
    stamp = turnover * 0.00015 if is_buy else 0.0
    sebi = turnover * 0.000001
    
    total = brokerage + exchange + gst + stt + stamp + sebi
    return {
        'brokerage': brokerage,
        'exchange_fee': exchange,
        'gst': gst,
        'stt': stt,
        'stamp_duty': stamp,
        'sebi_fee': sebi,
        'total_costs': total
    }

class VirtualBroker:
    """
    Executes virtual trades, applies realistic slippage and costs, 
    and updates the SQLite ledger.
    """
    def __init__(self, default_slippage_pct=0.0005): # 0.05% slippage
        self.default_slippage = default_slippage_pct
        
    def submit_order(self, strategy_id, symbol, side, qty, current_market_price, 
                     is_intraday=False, notes=""):
        """
        Executes a paper trade.
        side: 'BUY' or 'SELL'
        qty: positive integer
        """
        assert side in ['BUY', 'SELL']
        assert qty > 0
        
        # Apply slippage
        if side == 'BUY':
            fill_price = current_market_price * (1 + self.default_slippage)
        else:
            fill_price = current_market_price * (1 - self.default_slippage)
            
        gross_val = fill_price * qty
        costs = calculate_equity_costs(fill_price, qty, is_buy=(side=='BUY'), is_intraday=is_intraday)
        
        # Cash impact (buy takes cash, sell gives cash)
        # However, costs ALWAYS reduce cash
        if side == 'BUY':
            net_val = gross_val + costs['total_costs']
            cash_impact = -net_val
            pos_qty_change = qty
        else:
            net_val = gross_val - costs['total_costs']
            cash_impact = net_val
            pos_qty_change = -qty
            
        # 1. Update Database Order Log
        conn = get_db_connection()
        c = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        c.execute('''
            INSERT INTO orders (
                strategy_id, timestamp, symbol, side, quantity, fill_price, gross_value,
                brokerage, exchange_fee, gst, stt, stamp_duty, sebi_fee, total_costs, net_value, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id, now_str, symbol, side, qty, fill_price, gross_val,
            costs['brokerage'], costs['exchange_fee'], costs['gst'], costs['stt'],
            costs['stamp_duty'], costs['sebi_fee'], costs['total_costs'], net_val, notes
        ))
        conn.commit()
        conn.close()
        
        # 2. Update Position
        update_position(strategy_id, symbol, pos_qty_change, fill_price)
        
        # 3. Update Cash
        update_strategy_cash(strategy_id, cash_impact)
        
        return {
            'status': 'FILLED',
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'fill_price': fill_price,
            'costs': costs['total_costs'],
            'net_cash_impact': cash_impact
        }

    def get_portfolio_snapshot(self, strategy_id, current_prices_dict):
        """
        Calculates the live MTM (Mark-To-Market) value of the portfolio.
        current_prices_dict: {'RELIANCE.NS': 2500, 'TCS.NS': 3200, ...}
        """
        cash = get_strategy_cash(strategy_id)
        
        conn = get_db_connection()
        df_pos = pd.read_sql_query('SELECT * FROM positions WHERE strategy_id = ? AND quantity != 0', 
                                   conn, params=(strategy_id,))
        conn.close()
        
        holdings_value = 0.0
        positions_details = []
        
        for _, row in df_pos.iterrows():
            sym = row['symbol']
            qty = row['quantity']
            avg_px = row['avg_entry_price']
            
            c_price = current_prices_dict.get(sym, avg_px)
            
            # Unrealized PnL (Long or Short)
            if qty > 0:
                upnl = (c_price - avg_px) * qty
            else:
                upnl = (avg_px - c_price) * abs(qty)
                
            val = abs(qty) * c_price
            holdings_value += val
            
            positions_details.append({
                'symbol': sym,
                'qty': qty,
                'avg_price': avg_px,
                'current_price': c_price,
                'value': val,
                'unrealized_pnl': upnl
            })
            
        total_equity = cash + holdings_value
        
        # Determine drawdown
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT MAX(total_equity) as peak FROM equity_curve WHERE strategy_id = ?', (strategy_id,))
        row = c.fetchone()
        peak_eq = row['peak'] if (row and row['peak']) else total_equity
        conn.close()
        
        peak_eq = max(peak_eq, total_equity)
        drawdown_pct = ((total_equity - peak_eq) / peak_eq) * 100 if peak_eq > 0 else 0
        
        return {
            'strategy_id': strategy_id,
            'cash': cash,
            'holdings_value': holdings_value,
            'total_equity': total_equity,
            'drawdown_pct': drawdown_pct,
            'positions': positions_details
        }
        
    def save_eod_equity(self, strategy_id, current_prices_dict, date_str=None):
        """Saves End-Of-Day snapshot to equity curve history."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
            
        snap = self.get_portfolio_snapshot(strategy_id, current_prices_dict)
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO equity_curve (date, strategy_id, cash, holdings_value, total_equity, drawdown_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date_str, strategy_id, snap['cash'], snap['holdings_value'], snap['total_equity'], snap['drawdown_pct']))
        conn.commit()
        conn.close()
        
        return snap
