"""
V8 -- Position & Risk Manager
Manages open positions, position sizing, trailing stop-losses, and portfolio risk limits:
  1. Position Sizing: Max 10% capital per position, 1% portfolio risk per trade
  2. Max Concurrent Positions: 3
  3. Max Daily Drawdown: 2% of total capital (locks trading for the rest of the day)
  4. Intraday Auto Square-Off: 15:15 IST compulsory close
  5. Trailing Stop Management: Lock profits when price reaches 1x ATR in favor
"""
import sys
import os
from datetime import datetime, time as dtime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PositionManager:
    """
    Manages active trading positions and enforces strict risk rules.
    """

    def __init__(self, initial_capital=None):
        self.capital = initial_capital or config.INITIAL_CAPITAL
        self.current_equity = self.capital
        self.peak_equity = self.capital
        self.daily_starting_equity = self.capital
        self.max_concurrent = config.MAX_CONCURRENT_POSITIONS
        self.max_daily_loss_pct = config.MAX_DAILY_LOSS_PCT
        self.max_position_pct = config.MAX_POSITION_PCT

        # In-memory position dictionary: trade_id -> position dict
        self.positions = {}
        self.daily_pnl = 0.0
        self.is_kill_switch_active = False

    def start_new_day(self, current_date=None):
        """Resets daily risk metrics at the start of each session."""
        self.daily_starting_equity = self.current_equity
        self.daily_pnl = 0.0
        self.is_kill_switch_active = False

    def can_open_position(self, proposed_risk_amount=None):
        """
        Validates whether risk rules allow opening a new position.
        """
        # 1. Kill switch check
        if self.is_kill_switch_active:
            return False, "Kill switch is ACTIVE. Trading suspended."

        # 2. Max daily loss check
        current_daily_loss = self.daily_starting_equity - self.current_equity
        if current_daily_loss >= (self.daily_starting_equity * self.max_daily_loss_pct):
            self.is_kill_switch_active = True
            return False, f"Daily max loss reached ({current_daily_loss:.2f} >= 2%). Trading locked."

        # 3. Max concurrent positions
        if len(self.positions) >= self.max_concurrent:
            return False, f"Max concurrent positions reached ({len(self.positions)}/{self.max_concurrent})."

        return True, "OK"

    def calculate_position_size(self, entry_price, stop_price):
        """
        Calculates safe quantity based on 1% portfolio risk and max position value limits.
        """
        risk_per_trade_rs = self.current_equity * 0.01  # 1% equity risk
        risk_per_share = max(abs(entry_price - stop_price), entry_price * 0.005)

        qty_by_risk = int(risk_per_trade_rs / risk_per_share)
        max_position_value = self.current_equity * self.max_position_pct
        qty_by_capital = int(max_position_value / entry_price)

        final_qty = max(1, min(qty_by_risk, qty_by_capital))
        return final_qty

    def add_position(self, trade_id, symbol, setup_type, direction,
                     entry_price, stop_price, target_price, quantity, fill_entry_price):
        """
        Adds a new active position.
        """
        pos = {
            'trade_id': trade_id,
            'symbol': symbol,
            'setup_type': setup_type,
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'quantity': quantity,
            'fill_entry_price': fill_entry_price,
            'entry_time': datetime.now(),
            'highest_price': fill_entry_price,
            'lowest_price': fill_entry_price,
            'bars_held': 0,
            'trailing_stop': stop_price
        }
        self.positions[trade_id] = pos
        return pos

    def update_price_and_check_exits(self, current_prices_dict, current_time=None):
        """
        Updates highest/lowest prices, checks trailing stops, targets, and intraday time limits.
        Returns list of positions that need to be closed: [(pos, exit_price, reason)]
        """
        exits_to_execute = []
        now_time = current_time or datetime.now().time()
        square_off_time = dtime(15, 15)

        for trade_id, pos in list(self.positions.items()):
            sym = pos['symbol']
            if sym not in current_prices_dict:
                continue

            current_ltp = current_prices_dict[sym]
            pos['highest_price'] = max(pos['highest_price'], current_ltp)
            pos['lowest_price'] = min(pos['lowest_price'], current_ltp)
            pos['bars_held'] += 1

            # 1. Compulsory intraday square-off at 15:15 IST
            if now_time >= square_off_time:
                exits_to_execute.append((pos, current_ltp, 'INTRADAY_SQUARE_OFF'))
                continue

            # 2. Target Hit
            if current_ltp >= pos['target_price']:
                exits_to_execute.append((pos, pos['target_price'], 'TARGET_HIT'))
                continue

            # 3. Stop Hit (or Trailing Stop)
            if current_ltp <= pos['trailing_stop']:
                exits_to_execute.append((pos, pos['trailing_stop'], 'STOP_HIT'))
                continue

            # 4. Profit Trailing Logic: If price moves > 50% toward target, move stop to Breakeven
            move_to_target = (current_ltp - pos['fill_entry_price']) / (pos['target_price'] - pos['fill_entry_price'])
            if move_to_target >= 0.5 and pos['trailing_stop'] < pos['fill_entry_price']:
                pos['trailing_stop'] = pos['fill_entry_price']  # Breakeven stop

        return exits_to_execute

    def close_position(self, trade_id, net_pnl):
        """
        Removes a closed position and updates equity.
        """
        if trade_id in self.positions:
            del self.positions[trade_id]
            self.current_equity += net_pnl
            self.daily_pnl += net_pnl
            self.peak_equity = max(self.peak_equity, self.current_equity)
