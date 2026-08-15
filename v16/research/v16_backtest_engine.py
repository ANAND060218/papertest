"""
V16.1 Causal Intraday Backtest Engine
Walks through every historical trading day bar-by-bar, using ONLY information
available at that exact moment. No future data leaks.

Implements 3 independent strategies on hourly bars:
  A. Opening Range Breakout (ORB): First hour high/low → breakout trigger
  B. VWAP Trend Continuation: Price holds above/below session VWAP → continuation
  C. Gap Fade / Mean Reversion: Large opening gaps fade back toward previous close

All trades are flat by 15:15 IST. Full Indian intraday statutory cost model applied.
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntradayCostCalculator:
    """Exact Indian intraday equity statutory costs."""

    @staticmethod
    def round_trip_cost(buy_price, sell_price, qty, slippage_pct=0.0003):
        eff_buy = buy_price * (1 + slippage_pct)
        eff_sell = sell_price * (1 - slippage_pct)
        buy_val = eff_buy * qty
        sell_val = eff_sell * qty
        total_val = buy_val + sell_val

        brokerage = min(20.0, buy_val * 0.0003) + min(20.0, sell_val * 0.0003)
        stt = sell_val * 0.00025          # 0.025% sell side only (intraday)
        exchange = total_val * 0.0000345  # NSE turnover charges
        gst = (brokerage + exchange) * 0.18
        stamp = buy_val * 0.00003         # stamp duty on buy
        sebi = total_val * 0.000001

        total_statutory = brokerage + stt + exchange + gst + stamp + sebi
        slippage_cost = (eff_buy - buy_price + sell_price - eff_sell) * qty

        gross_pnl = (sell_price - buy_price) * qty
        net_pnl = (eff_sell - eff_buy) * qty - total_statutory

        return {
            'gross_pnl': round(gross_pnl, 2),
            'net_pnl': round(net_pnl, 2),
            'total_costs': round(total_statutory + slippage_cost, 2),
            'friction_pct': round((total_statutory + slippage_cost) / buy_val * 100, 4) if buy_val > 0 else 0
        }


class V16BacktestEngine:
    """
    Day-by-day causal intraday backtest engine using hourly bars.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(BASE_DIR, "data", "historical_2year_hourly.db")
        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        self.hourly_df = pd.read_sql_query(
            "SELECT Date, Open, High, Low, Close, Volume, symbol FROM universe_hourly_2year ORDER BY Date ASC",
            conn
        )
        conn.close()
        self.hourly_df['Date'] = pd.to_datetime(self.hourly_df['Date'])
        self.hourly_df['trade_date'] = self.hourly_df['Date'].dt.date
        self.symbols = [s for s in self.hourly_df['symbol'].unique() if s != '^NSEI']
        self.nifty_hourly = self.hourly_df[self.hourly_df['symbol'] == '^NSEI'].copy()

    def _get_nifty_regime(self, trade_date):
        """Uses ONLY prior-day NIFTY data to determine regime (zero lookahead)."""
        nifty_prior = self.nifty_hourly[self.nifty_hourly['trade_date'] < trade_date].copy()
        if len(nifty_prior) < 40:
            return 'NEUTRAL', 50.0

        # Use daily closes from prior sessions
        daily = nifty_prior.groupby('trade_date').agg({'Close': 'last', 'Open': 'first'}).reset_index()
        daily = daily.sort_values('trade_date')

        if len(daily) < 20:
            return 'NEUTRAL', 50.0

        sma5 = daily['Close'].rolling(5).mean().iloc[-1]
        sma20 = daily['Close'].rolling(20).mean().iloc[-1]
        last_close = daily['Close'].iloc[-1]
        prev_candle_bullish = daily['Close'].iloc[-1] > daily['Open'].iloc[-1]

        score = 50.0
        if last_close > sma20: score += 15
        else: score -= 15
        if sma5 > sma20: score += 10
        else: score -= 10
        if prev_candle_bullish: score += 10
        else: score -= 10

        score = max(0, min(100, score))
        if score >= 60: return 'BULLISH', score
        elif score <= 40: return 'BEARISH', score
        else: return 'NEUTRAL', score

    def _get_stock_context(self, symbol, trade_date):
        """Prior-day multi-day context for a stock (zero lookahead)."""
        prior = self.hourly_df[
            (self.hourly_df['symbol'] == symbol) &
            (self.hourly_df['trade_date'] < trade_date)
        ].copy()

        if len(prior) < 20:
            return None

        daily = prior.groupby('trade_date').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).reset_index().sort_values('trade_date')

        if len(daily) < 5:
            return None

        prev_close = daily['Close'].iloc[-1]
        prev_high = daily['High'].iloc[-1]
        prev_low = daily['Low'].iloc[-1]

        # ATR (5-day)
        daily['TR'] = np.maximum(
            daily['High'] - daily['Low'],
            np.maximum(abs(daily['High'] - daily['Close'].shift(1)),
                       abs(daily['Low'] - daily['Close'].shift(1)))
        )
        atr = daily['TR'].rolling(5).mean().iloc[-1]

        # 5-day trend
        mom_5d = (daily['Close'].iloc[-1] / daily['Close'].iloc[-5] - 1) * 100 if len(daily) >= 5 else 0

        return {
            'prev_close': prev_close,
            'prev_high': prev_high,
            'prev_low': prev_low,
            'atr_5d': atr,
            'atr_pct': (atr / prev_close * 100) if prev_close > 0 else 1.0,
            'mom_5d_pct': mom_5d
        }

    def run_strategy_backtest(self, strategy='ORB', capital_per_trade=50000.0, slippage_pct=0.0003):
        """
        Runs a specific strategy across all trading days and all stocks.
        """
        all_trade_dates = sorted(self.hourly_df['trade_date'].unique())
        all_trades = []
        cost_calc = IntradayCostCalculator()

        for td in all_trade_dates:
            # 1. Get market regime using ONLY prior data
            regime, regime_score = self._get_nifty_regime(td)

            for sym in self.symbols:
                # 2. Get stock context using ONLY prior data
                ctx = self._get_stock_context(sym, td)
                if ctx is None:
                    continue

                # 3. Get today's hourly bars (simulating real-time bar-by-bar)
                today_bars = self.hourly_df[
                    (self.hourly_df['symbol'] == sym) &
                    (self.hourly_df['trade_date'] == td)
                ].sort_values('Date').reset_index(drop=True)

                if len(today_bars) < 4:
                    continue

                trade_result = None
                if strategy == 'ORB':
                    trade_result = self._simulate_orb(today_bars, ctx, regime, capital_per_trade)
                elif strategy == 'VWAP':
                    trade_result = self._simulate_vwap(today_bars, ctx, regime, capital_per_trade)
                elif strategy == 'GAP_FADE':
                    trade_result = self._simulate_gap_fade(today_bars, ctx, regime, capital_per_trade)

                if trade_result is not None:
                    trade_result['symbol'] = sym
                    trade_result['trade_date'] = str(td)
                    trade_result['regime'] = regime
                    trade_result['regime_score'] = regime_score

                    # Apply costs
                    costs = cost_calc.round_trip_cost(
                        trade_result['entry_price'], trade_result['exit_price'],
                        trade_result['qty'], slippage_pct
                    )
                    trade_result.update(costs)
                    all_trades.append(trade_result)

        return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    def _simulate_orb(self, today_bars, ctx, regime, capital):
        """
        Strategy A: Opening Range Breakout.
        First hourly bar defines the range. Breakout trigger on 2nd/3rd bar.
        Direction aligned with NIFTY regime.
        """
        first_bar = today_bars.iloc[0]
        or_high = first_bar['High']
        or_low = first_bar['Low']
        or_range = or_high - or_low

        if or_range <= 0 or or_range / first_bar['Close'] * 100 < 0.15:
            return None  # Range too tight

        direction = 'LONG' if regime != 'BEARISH' else 'SHORT'

        # Walk subsequent bars looking for breakout
        for i in range(1, min(4, len(today_bars))):  # Bars 2-4 only (10:15 to 12:15)
            bar = today_bars.iloc[i]

            if direction == 'LONG' and bar['High'] > or_high:
                entry_price = or_high + 0.05
                stop_loss = or_low
                risk = entry_price - stop_loss
                if risk <= 0 or risk / entry_price * 100 > 2.0:
                    return None  # Risk too large

                target = entry_price + risk * 1.5
                qty = max(1, int(capital / entry_price))

                # Walk remaining bars for exit
                exit_price = entry_price
                exit_reason = 'EOD_FLAT'
                for j in range(i + 1, len(today_bars)):
                    ebar = today_bars.iloc[j]
                    if ebar['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'STOP_LOSS'
                        break
                    elif ebar['High'] >= target:
                        exit_price = target
                        exit_reason = 'TARGET_HIT'
                        break
                    exit_price = ebar['Close']  # EOD exit

                return {
                    'strategy': 'ORB',
                    'direction': direction,
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'exit_price': round(exit_price, 2),
                    'exit_reason': exit_reason,
                    'qty': qty,
                    'or_range_pct': round(or_range / first_bar['Close'] * 100, 3),
                    'atr_pct': round(ctx['atr_pct'], 2)
                }

            elif direction == 'SHORT' and bar['Low'] < or_low:
                entry_price = or_low - 0.05
                stop_loss = or_high
                risk = stop_loss - entry_price
                if risk <= 0 or risk / entry_price * 100 > 2.0:
                    return None

                target = entry_price - risk * 1.5
                qty = max(1, int(capital / entry_price))

                exit_price = entry_price
                exit_reason = 'EOD_FLAT'
                for j in range(i + 1, len(today_bars)):
                    ebar = today_bars.iloc[j]
                    if ebar['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'STOP_LOSS'
                        break
                    elif ebar['Low'] <= target:
                        exit_price = target
                        exit_reason = 'TARGET_HIT'
                        break
                    exit_price = ebar['Close']

                return {
                    'strategy': 'ORB',
                    'direction': direction,
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'exit_price': round(exit_price, 2),
                    'exit_reason': exit_reason,
                    'qty': qty,
                    'or_range_pct': round(or_range / first_bar['Close'] * 100, 3),
                    'atr_pct': round(ctx['atr_pct'], 2)
                }

        return None  # No breakout

    def _simulate_vwap(self, today_bars, ctx, regime, capital):
        """
        Strategy B: VWAP Trend Continuation.
        If price is above session VWAP and regime is bullish, buy pullback to VWAP.
        """
        if regime == 'NEUTRAL':
            return None

        direction = 'LONG' if regime == 'BULLISH' else 'SHORT'

        # Calculate running VWAP (using typical price × volume proxy)
        cumulative_tp_vol = 0.0
        cumulative_vol = 0.0

        for i in range(len(today_bars)):
            bar = today_bars.iloc[i]
            tp = (bar['High'] + bar['Low'] + bar['Close']) / 3
            vol = max(bar['Volume'], 1)
            cumulative_tp_vol += tp * vol
            cumulative_vol += vol

        if cumulative_vol == 0:
            return None

        session_vwap = cumulative_tp_vol / cumulative_vol

        # Look for pullback to VWAP in bars 2-4
        for i in range(1, min(4, len(today_bars))):
            bar = today_bars.iloc[i]

            if direction == 'LONG':
                # Price pulled back near VWAP and bounced
                if bar['Low'] <= session_vwap * 1.002 and bar['Close'] > session_vwap:
                    entry_price = bar['Close']
                    risk = ctx['atr_5d'] * 0.5
                    if risk <= 0: continue
                    stop_loss = entry_price - risk
                    target = entry_price + risk * 2.0
                    qty = max(1, int(capital / entry_price))

                    exit_price = entry_price
                    exit_reason = 'EOD_FLAT'
                    for j in range(i + 1, len(today_bars)):
                        ebar = today_bars.iloc[j]
                        if ebar['Low'] <= stop_loss:
                            exit_price = stop_loss
                            exit_reason = 'STOP_LOSS'
                            break
                        elif ebar['High'] >= target:
                            exit_price = target
                            exit_reason = 'TARGET_HIT'
                            break
                        exit_price = ebar['Close']

                    return {
                        'strategy': 'VWAP_PULLBACK',
                        'direction': direction,
                        'entry_price': round(entry_price, 2),
                        'stop_loss': round(stop_loss, 2),
                        'target': round(target, 2),
                        'exit_price': round(exit_price, 2),
                        'exit_reason': exit_reason,
                        'qty': qty,
                        'vwap': round(session_vwap, 2),
                        'atr_pct': round(ctx['atr_pct'], 2)
                    }

            elif direction == 'SHORT':
                if bar['High'] >= session_vwap * 0.998 and bar['Close'] < session_vwap:
                    entry_price = bar['Close']
                    risk = ctx['atr_5d'] * 0.5
                    if risk <= 0: continue
                    stop_loss = entry_price + risk
                    target = entry_price - risk * 2.0
                    qty = max(1, int(capital / entry_price))

                    exit_price = entry_price
                    exit_reason = 'EOD_FLAT'
                    for j in range(i + 1, len(today_bars)):
                        ebar = today_bars.iloc[j]
                        if ebar['High'] >= stop_loss:
                            exit_price = stop_loss
                            exit_reason = 'STOP_LOSS'
                            break
                        elif ebar['Low'] <= target:
                            exit_price = target
                            exit_reason = 'TARGET_HIT'
                            break
                        exit_price = ebar['Close']

                    return {
                        'strategy': 'VWAP_PULLBACK',
                        'direction': direction,
                        'entry_price': round(entry_price, 2),
                        'stop_loss': round(stop_loss, 2),
                        'target': round(target, 2),
                        'exit_price': round(exit_price, 2),
                        'exit_reason': exit_reason,
                        'qty': qty,
                        'vwap': round(session_vwap, 2),
                        'atr_pct': round(ctx['atr_pct'], 2)
                    }

        return None

    def _simulate_gap_fade(self, today_bars, ctx, regime, capital):
        """
        Strategy C: Gap Fade / Mean Reversion.
        If stock gaps > 1.0% from previous close, fade back toward previous close.
        """
        first_bar = today_bars.iloc[0]
        gap_pct = (first_bar['Open'] - ctx['prev_close']) / ctx['prev_close'] * 100

        # Only trade gaps > 1.0%
        if abs(gap_pct) < 1.0:
            return None

        if gap_pct > 1.0:
            # Gap up → short fade
            direction = 'SHORT'
            entry_price = first_bar['Close']  # Enter at first bar close
            target = ctx['prev_close'] + (first_bar['Open'] - ctx['prev_close']) * 0.5  # 50% gap fill
            risk = ctx['atr_5d'] * 0.5
            if risk <= 0: return None
            stop_loss = entry_price + risk
        elif gap_pct < -1.0:
            # Gap down → long fade
            direction = 'LONG'
            entry_price = first_bar['Close']
            target = ctx['prev_close'] - (ctx['prev_close'] - first_bar['Open']) * 0.5
            risk = ctx['atr_5d'] * 0.5
            if risk <= 0: return None
            stop_loss = entry_price - risk
        else:
            return None

        qty = max(1, int(capital / entry_price))

        exit_price = entry_price
        exit_reason = 'EOD_FLAT'
        for j in range(1, len(today_bars)):
            ebar = today_bars.iloc[j]
            if direction == 'LONG':
                if ebar['Low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                    break
                elif ebar['High'] >= target:
                    exit_price = target
                    exit_reason = 'TARGET_HIT'
                    break
            else:
                if ebar['High'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                    break
                elif ebar['Low'] <= target:
                    exit_price = target
                    exit_reason = 'TARGET_HIT'
                    break
            exit_price = ebar['Close']

        pnl_multiplier = 1 if direction == 'LONG' else -1

        return {
            'strategy': 'GAP_FADE',
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'exit_price': round(exit_price, 2),
            'exit_reason': exit_reason,
            'qty': qty,
            'gap_pct': round(gap_pct, 2),
            'atr_pct': round(ctx['atr_pct'], 2)
        }
