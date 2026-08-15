"""
V17.1 Causal 5-Minute Intraday Backtest Engine
Implements:
  - 08:45-09:15 Pre-market Regime
  - Candidate Ranking (Relative Strength, Gap)
  - 5-Minute Bar Streaming
  - Strategy A (ORB + VWAP)
  - Strategy B (VWAP Pullback)
  - Full Indian Intraday Cost Model
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def intraday_costs(buy_p, sell_p, qty, slip=0.0003):
    eb = buy_p * (1 + slip)
    es = sell_p * (1 - slip)
    bv, sv = eb * qty, es * qty
    tv = bv + sv
    brk = min(20, bv * 0.0003) + min(20, sv * 0.0003)
    stt = sv * 0.00025
    exc = tv * 0.0000345
    gst = (brk + exc) * 0.18
    stamp = bv * 0.00003
    sebi = tv * 0.000001
    stat = brk + stt + exc + gst + stamp + sebi
    slip_cost = (eb - buy_p + sell_p - es) * qty
    gross = (sell_p - buy_p) * qty
    net = (es - eb) * qty - stat
    return gross, net, stat + slip_cost


class V17BacktestEngine:
    def __init__(self):
        self.daily_db = os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
        self._load_data()

    def _load_data(self):
        # 1. Load Daily Data for Regime Context
        conn_daily = sqlite3.connect(self.daily_db)
        self.daily_df = pd.read_sql_query("SELECT * FROM stock_daily_10y", conn_daily)
        conn_daily.close()
        self.daily_df['Date'] = pd.to_datetime(self.daily_df['Date']).dt.date
        
        self.nifty_daily = self.daily_df[self.daily_df['Symbol'] == '^NSEI'].copy().sort_values('Date')
        self.nifty_daily['sma5'] = self.nifty_daily['Close'].rolling(5).mean()
        self.nifty_daily['sma20'] = self.nifty_daily['Close'].rolling(20).mean()

        # 2. Load 5m Intraday Data from Multi-Year DB
        multi_year_db = os.path.join(BASE_DIR, "data", "v17_multi_year.db")
        conn_intra = sqlite3.connect(multi_year_db)
        self.m5_df = pd.read_sql_query("SELECT datetime as Date, open as Open, high as High, low as Low, close as Close, volume as Volume, symbol FROM intraday_5m", conn_intra)
        conn_intra.close()

        self.m5_df['Date'] = pd.to_datetime(self.m5_df['Date'])
        self.m5_df['td'] = self.m5_df['Date'].dt.date
        
        # We need NIFTY 5m separately for relative strength and gap calculations
        self.nifty_m5 = self.m5_df[self.m5_df['symbol'] == '^NSEI'].copy().sort_values('Date')
        self.stock_symbols = [s for s in self.m5_df['symbol'].unique() if s != '^NSEI']

    def get_regime(self, trade_date):
        """08:45-09:15 Pre-Market Engine: Zero lookahead daily context"""
        prior = self.nifty_daily[self.nifty_daily['Date'] < trade_date]
        if len(prior) < 20:
            return 'NO_TRADE', 50
        
        last = prior.iloc[-1]
        
        score = 50
        if last['Close'] > last['sma20']: score += 15
        else: score -= 15
        if last['sma5'] > last['sma20']: score += 10
        else: score -= 10
        if last['Close'] > last['Open']: score += 10
        else: score -= 10
            
        if score >= 60: return 'BULLISH', score
        elif score <= 40: return 'BEARISH', score
        return 'NO_TRADE', score

    def get_stock_context(self, sym, trade_date):
        prior = self.daily_df[(self.daily_df['Symbol'] == sym) & (self.daily_df['Date'] < trade_date)].copy()
        if len(prior) < 5:
            return None
        prior = prior.sort_values('Date')
        
        # Calculate ATR
        prior['TR'] = np.maximum(prior['High'] - prior['Low'],
                      np.maximum(abs(prior['High'] - prior['Close'].shift(1)),
                                 abs(prior['Low'] - prior['Close'].shift(1))))
        atr = prior['TR'].rolling(5).mean().iloc[-1]
        prev_close = prior['Close'].iloc[-1]
        
        if pd.isna(atr) or prev_close == 0:
            return None
            
        return {
            'prev_close': prev_close,
            'atr': atr
        }

    def run_strategy(self, capital=50000):
        all_dates = sorted(self.m5_df['td'].unique())
        trades = []
        
        print(f"Starting V17.1 5m Backtest on {len(all_dates)} days...")
        
        regime_counts = {'BULLISH': 0, 'BEARISH': 0, 'NO_TRADE': 0}

        for td in all_dates:
            regime, score = self.get_regime(td)
            regime_counts[regime] += 1
            if regime == 'NO_TRADE':
                continue # Strict no-trade filter

            direction = 'LONG' if regime == 'BULLISH' else 'SHORT'

            # 1. 09:15 - 09:45: Observe Opening Range and Rank Candidates
            nifty_today = self.nifty_m5[self.nifty_m5['td'] == td].sort_values('Date').reset_index(drop=True)
            if len(nifty_today) < 6: # Need at least 30 mins (6 bars)
                print(f"  [{td}] Skipped: Nifty bars < 6 ({len(nifty_today)})")
                continue
                
            nifty_open = nifty_today.iloc[0]['Open']
            
            candidates = []
            for sym in self.stock_symbols:
                ctx = self.get_stock_context(sym, td)
                if not ctx: continue
                
                sym_today = self.m5_df[(self.m5_df['symbol'] == sym) & (self.m5_df['td'] == td)].sort_values('Date').reset_index(drop=True)
                if len(sym_today) < 75: # Need a full day of data
                    continue
                    
                # First 30 mins (6 bars)
                or_bars = sym_today.iloc[0:6]
                or_high = or_bars['High'].max()
                or_low = or_bars['Low'].min()
                or_close = or_bars.iloc[-1]['Close']
                
                # Relative Strength vs NIFTY in first 30m
                nifty_or_close = nifty_today.iloc[5]['Close']
                nifty_ret = (nifty_or_close - nifty_open) / nifty_open
                sym_ret = (or_close - sym_today.iloc[0]['Open']) / sym_today.iloc[0]['Open']
                rs = sym_ret - nifty_ret
                
                # Gap %
                gap_pct = (sym_today.iloc[0]['Open'] - ctx['prev_close']) / ctx['prev_close'] * 100
                
                candidates.append({
                    'sym': sym,
                    'rs': rs,
                    'gap_pct': gap_pct,
                    'or_high': or_high,
                    'or_low': or_low,
                    'sym_today': sym_today,
                    'atr': ctx['atr']
                })
            
            if not candidates:
                print(f"  [{td}] Skipped: No candidates found (likely sym_today < 75 bars).")
                continue
            
            print(f"  [{td}] {regime} - Found {len(candidates)} candidates.")
            
            # Rank: Pick strongest for Longs, weakest for Shorts
            candidates.sort(key=lambda x: x['rs'], reverse=(direction == 'LONG'))
            top_candidates = candidates[:2] # Trade top 2

            for cand in top_candidates:
                sym_today = cand['sym_today']
                
                # Pre-calculate VWAP for the whole day (causally used)
                cum_tp_vol = 0.0
                cum_vol = 0.0
                vwaps = []
                for i in range(len(sym_today)):
                    b = sym_today.iloc[i]
                    tp = (b['High'] + b['Low'] + b['Close']) / 3
                    v = max(b['Volume'], 1)
                    cum_tp_vol += tp * v
                    cum_vol += v
                    vwaps.append(cum_tp_vol / cum_vol if cum_vol > 0 else b['Close'])
                
                sym_today['vwap'] = vwaps

                # ==========================================
                # Strategy A: ORB + VWAP Confirmation
                # ==========================================
                self._run_strat_A(td, cand, sym_today, direction, regime, capital, trades)
                
                # ==========================================
                # Strategy B: VWAP Pullback
                # ==========================================
                self._run_strat_B(td, cand, sym_today, direction, regime, capital, trades)

        print(f"Regime Breakdown: {regime_counts}")
        return pd.DataFrame(trades)

    def _run_strat_A(self, td, cand, sym_today, direction, regime, capital, trades):
        """
        ORB + VWAP: Breakout of 30-min Opening Range.
        Must be on the correct side of VWAP for confirmation.
        """
        sym = cand['sym']
        or_h = cand['or_high']
        or_l = cand['or_low']
        
        for i in range(6, len(sym_today) - 6): # Look for entry between 09:45 and 14:45
            bar = sym_today.iloc[i]
            vwap = bar['vwap']
            
            if direction == 'LONG':
                # Breakout above OR high AND price > VWAP
                if bar['High'] > or_h and bar['Close'] > vwap:
                    entry_p = or_h + 0.05
                    sl = or_l # Stop below OR low
                    risk = entry_p - sl
                    if risk <= 0 or risk/entry_p > 0.02: break
                    tgt = entry_p + risk * 2.0
                    
                    self._manage_trade(sym_today, i, entry_p, sl, tgt, capital, 'Strat_A_ORB_VWAP', direction, td, sym, regime, trades)
                    break
                    
            elif direction == 'SHORT':
                if bar['Low'] < or_l and bar['Close'] < vwap:
                    entry_p = or_l - 0.05
                    sl = or_h
                    risk = sl - entry_p
                    if risk <= 0 or risk/entry_p > 0.02: break
                    tgt = entry_p - risk * 2.0
                    
                    self._manage_trade(sym_today, i, entry_p, sl, tgt, capital, 'Strat_A_ORB_VWAP', direction, td, sym, regime, trades)
                    break

    def _run_strat_B(self, td, cand, sym_today, direction, regime, capital, trades):
        """
        VWAP Pullback: Price trends away, pulls back to VWAP, and bounces.
        """
        sym = cand['sym']
        
        # Wait for trend to establish (09:45 - 11:00)
        # Look for pullback after 10:00
        for i in range(9, len(sym_today) - 6):
            bar = sym_today.iloc[i]
            prev_bar = sym_today.iloc[i-1]
            vwap = bar['vwap']
            
            if direction == 'LONG':
                # Pulled back to VWAP and bounced
                if prev_bar['Low'] <= vwap * 1.002 and bar['Close'] > vwap and bar['Close'] > bar['Open']:
                    entry_p = bar['Close']
                    sl = entry_p - cand['atr'] * 0.4
                    tgt = entry_p + cand['atr'] * 0.8
                    self._manage_trade(sym_today, i, entry_p, sl, tgt, capital, 'Strat_B_VWAP_Pullback', direction, td, sym, regime, trades)
                    break
            
            elif direction == 'SHORT':
                if prev_bar['High'] >= vwap * 0.998 and bar['Close'] < vwap and bar['Close'] < bar['Open']:
                    entry_p = bar['Close']
                    sl = entry_p + cand['atr'] * 0.4
                    tgt = entry_p - cand['atr'] * 0.8
                    self._manage_trade(sym_today, i, entry_p, sl, tgt, capital, 'Strat_B_VWAP_Pullback', direction, td, sym, regime, trades)
                    break


    def _manage_trade(self, sym_today, i, entry_p, sl, tgt, capital, strat_name, direction, td, sym, regime, trades):
        qty = max(1, int(capital / entry_p))
        exit_p = entry_p
        reason = 'EOD_1515'
        
        entry_dt = sym_today.iloc[i]['Date']
        exit_dt = sym_today.iloc[-1]['Date'] # default EOD
        for j in range(i + 1, len(sym_today)):
            eb = sym_today.iloc[j]
            # 15:15 Force Exit (index 72 out of 75 for 5m bars 09:15-15:30)
            if eb['Date'].hour == 15 and eb['Date'].minute >= 15:
                exit_p = eb['Close']
                exit_dt = eb['Date']
                break
                
            if direction == 'LONG':
                if eb['Low'] <= sl: exit_p, reason, exit_dt = sl, 'SL', eb['Date']; break
                if eb['High'] >= tgt: exit_p, reason, exit_dt = tgt, 'TGT', eb['Date']; break
            else:
                if eb['High'] >= sl: exit_p, reason, exit_dt = sl, 'SL', eb['Date']; break
                if eb['Low'] <= tgt: exit_p, reason, exit_dt = tgt, 'TGT', eb['Date']; break
                
            exit_p = eb['Close']
            exit_dt = eb['Date']
            
        g, n, c = intraday_costs(min(entry_p, exit_p), max(entry_p, exit_p), qty)
        if direction == 'SHORT':
            g = (entry_p - exit_p) * qty
            n = g - c
            
        trades.append({
            'entry_time': str(entry_dt), 'exit_time': str(exit_dt),
            'date': str(td), 'sym': sym, 'strat': strat_name, 'dir': direction, 'regime': regime,
            'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
            'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2)
        })

if __name__ == "__main__":
    engine = V17BacktestEngine()
    df_trades = engine.run_strategy()
    
    out_path = os.path.join(BASE_DIR, "results", "v17_multi_year_trades.csv")
    df_trades.to_csv(out_path, index=False)
    print(f"Generated {len(df_trades)} trades. Saved to {out_path}")
