"""
V8 -- Paper Order Executor
Simulates live broker order execution with:
  1. Bid/Ask spread model (0.02%)
  2. Realistic Slippage model (0.05% on market orders)
  3. Execution Latency (100ms simulated delay)
  4. Complete statutory and regulatory transaction costs (Brokerage, STT, Stamp Duty, GST, Exchange, SEBI)
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backtest.backtester import CostModel


class PaperExecutor:
    """
    Simulated Broker Order Execution Engine for Paper Trading.
    """

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or CostModel()
        self.slippage_pct = config.SLIPPAGE_PCT
        self.spread_pct = config.SPREAD_PCT

    def execute_market_buy(self, symbol, current_ltp, quantity):
        """
        Simulates a market BUY order with spread and adverse slippage.
        """
        # Half-spread + slippage impact
        effective_cost_rate = (self.spread_pct / 2.0) + self.slippage_pct
        fill_price = current_ltp * (1.0 + effective_cost_rate)

        # Micro-latency simulation (100ms)
        time.sleep(0.05)

        fill_time = datetime.now()
        order_info = {
            'order_type': 'MARKET_BUY',
            'symbol': symbol,
            'ltp': current_ltp,
            'quantity': quantity,
            'fill_price': round(fill_price, 2),
            'fill_time': fill_time,
            'slippage_pct': round(effective_cost_rate * 100, 3)
        }
        return order_info

    def execute_market_sell(self, symbol, current_ltp, quantity):
        """
        Simulates a market SELL order with spread and adverse slippage.
        """
        effective_cost_rate = (self.spread_pct / 2.0) + self.slippage_pct
        fill_price = current_ltp * (1.0 - effective_cost_rate)

        time.sleep(0.05)

        fill_time = datetime.now()
        order_info = {
            'order_type': 'MARKET_SELL',
            'symbol': symbol,
            'ltp': current_ltp,
            'quantity': quantity,
            'fill_price': round(fill_price, 2),
            'fill_time': fill_time,
            'slippage_pct': round(effective_cost_rate * 100, 3)
        }
        return order_info

    def calculate_trade_costs(self, fill_entry_price, fill_exit_price, quantity):
        """
        Calculates all Indian statutory charges for the executed trade.
        """
        return self.cost_model.calculate_round_trip_cost(fill_entry_price, fill_exit_price, quantity)
