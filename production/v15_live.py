import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.paper_broker import VirtualBroker
from core.database import get_db_connection

class V15PaperEngine:
    """
    Executes the frozen V15.2 Top-10 Dual Momentum model, using the Virtual Broker.
    Runs once a month to rebalance.
    """
    def __init__(self, broker: VirtualBroker, strategy_id="V15.2_MOMENTUM"):
        self.broker = broker
        self.strategy_id = strategy_id
        
        # Load Frozen Configuration
        config_path = os.path.join(BASE_DIR, "data", "production_config.json")
        with open(config_path, "r") as f:
            self.prod_cfg = json.load(f)
            
    def run_monthly_rebalance(self, current_date_str):
        print(f"\n[V15.2 PAPER ENGINE] Running Monthly Rebalance for {current_date_str}")
        
        # 1. Ingest Data (simulate with local DB up to current_date_str)
        db_path = os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
        conn_mkt = sqlite3.connect(db_path)
        df_all = pd.read_sql_query(f"SELECT Date, Symbol, Close FROM stock_daily_10y WHERE Date <= '{current_date_str}' ORDER BY Date ASC", conn_mkt)
        conn_mkt.close()

        df_all['Date'] = pd.to_datetime(df_all['Date'])
        stocks_df = df_all[~df_all['Symbol'].isin(['^NSEI', '^NSEBANK'])].copy()

        price_matrix = stocks_df.pivot(index='Date', columns='Symbol', values='Close').ffill()
        monthly_df = price_matrix.resample('ME').last().ffill()

        # 2. Compute Signals
        l1 = self.prod_cfg['signal_parameters']['momentum_lookback_1_months']
        l2 = self.prod_cfg['signal_parameters']['momentum_lookback_2_months']
        sma_p = self.prod_cfg['signal_parameters']['absolute_trend_filter_months']
        top_n = self.prod_cfg['portfolio_construction']['holdings_count_top_n']

        mom_1 = monthly_df.pct_change(l1).iloc[-1]
        mom_2 = monthly_df.pct_change(l2).iloc[-1]
        combined_mom = 0.5 * mom_1 + 0.5 * mom_2
        sma_filter = monthly_df.rolling(sma_p).mean().iloc[-1]
        latest_prices = price_matrix.iloc[-1]

        ranked_data = []
        for sym in price_matrix.columns:
            c_price = latest_prices[sym]
            m_score = combined_mom.get(sym, np.nan)
            sma_val = sma_filter.get(sym, np.nan)
            is_above_sma = c_price > sma_val if pd.notna(sma_val) else False

            if pd.notna(m_score):
                ranked_data.append({
                    'symbol': sym,
                    'current_price': round(float(c_price), 2),
                    'score': float(m_score),
                    'is_above_10m_sma': is_above_sma
                })

        df_ranked = pd.DataFrame(ranked_data).sort_values(by='score', ascending=False).reset_index(drop=True)
        selected_leaders = df_ranked[df_ranked['is_above_10m_sma']].head(top_n).copy()

        # 3. Fetch Current Positions from Virtual Broker
        conn = get_db_connection()
        pos_df = pd.read_sql_query("SELECT symbol, quantity FROM positions WHERE strategy_id = ? AND quantity != 0", conn, params=(self.strategy_id,))
        
        # Calculate total strategy equity to determine allocation size
        current_prices_dict = latest_prices.to_dict()
        snapshot = self.broker.get_portfolio_snapshot(self.strategy_id, current_prices_dict)
        total_equity = snapshot['total_equity']
        
        current_holdings = dict(zip(pos_df['symbol'], pos_df['quantity'])) if not pos_df.empty else {}
        target_holdings = {}
        
        alloc_per_stock = total_equity / top_n
        
        for _, row in selected_leaders.iterrows():
            sym = row['symbol']
            c_p = row['current_price']
            qty = int(alloc_per_stock / c_p)
            target_holdings[sym] = qty

        print(f"  Target Portfolio: {list(target_holdings.keys())}")

        # 4. Generate diff orders (Sells first to free cash, then Buys)
        sells = []
        buys = []

        # Find stocks to sell or trim
        for sym, curr_qty in current_holdings.items():
            tgt_qty = target_holdings.get(sym, 0)
            diff = tgt_qty - curr_qty
            if diff < 0:
                sells.append({'symbol': sym, 'qty': abs(diff)})
                
        # Find stocks to buy
        for sym, tgt_qty in target_holdings.items():
            curr_qty = current_holdings.get(sym, 0)
            diff = tgt_qty - curr_qty
            if diff > 0:
                buys.append({'symbol': sym, 'qty': diff})

        # Execute Sells
        for order in sells:
            sym = order['symbol']
            qty = order['qty']
            price = current_prices_dict[sym]
            print(f"  [SIGNAL] {sym}: SELL {qty} shares")
            self.broker.submit_order(self.strategy_id, sym, 'SELL', qty, price, notes="Monthly Rebalance Exit/Trim")

        # Execute Buys
        for order in buys:
            sym = order['symbol']
            qty = order['qty']
            price = current_prices_dict[sym]
            print(f"  [SIGNAL] {sym}: BUY {qty} shares")
            self.broker.submit_order(self.strategy_id, sym, 'BUY', qty, price, notes="Monthly Rebalance Entry")

        # 5. Save EOD Snapshot
        self.broker.save_eod_equity(self.strategy_id, current_prices_dict, current_date_str)
        conn.close()

if __name__ == "__main__":
    from core.database import init_db, register_strategy
    init_db()
    
    # Initialize with 10L capital
    register_strategy("V15.2_MOMENTUM", 1000000.0)
    
    broker = VirtualBroker(default_slippage_pct=0.0005)
    engine = V15PaperEngine(broker)
    
    # Simulate today's run
    engine.run_monthly_rebalance("2026-08-15")
