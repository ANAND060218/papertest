"""
V16 Pre-Market & Opening Liquidity Scanner
Scans the liquid stock universe across 1M, 1W, and 1D timeframes + opening gap %
to rank today's highest-probability intraday trading candidates.
"""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


class PremarketScanner:
    """
    Multi-timeframe liquidity and relative strength scanner.
    """

    def __init__(self, stocks_daily_df, nifty_daily_df):
        self.stocks_df = stocks_daily_df.copy()
        self.nifty_df = nifty_daily_df.copy()

    def scan_for_date(self, target_date, top_n=5, market_bias='BULLISH'):
        """
        Scans all liquid stocks at market open on target_date.
        """
        target_dt = pd.to_datetime(target_date)
        candidates = []

        symbols = [s for s in self.stocks_df['Symbol'].unique() if s not in ['^NSEI', '^NSEBANK']]

        for sym in symbols:
            df_sym = self.stocks_df[(self.stocks_df['Symbol'] == sym) & (self.stocks_df['Date'] <= target_dt)].sort_values('Date').reset_index(drop=True)
            if len(df_sym) < 30:
                continue

            today = df_sym.iloc[-1]
            prev = df_sym.iloc[-2]

            # 1. Multi-Timeframe Trend Checks
            # 1-Month Momentum (~21 bars)
            mom_1m = (today['Close'] - df_sym.iloc[-21]['Close']) / df_sym.iloc[-21]['Close'] * 100 if len(df_sym) >= 21 else 0.0
            # 1-Week Momentum (~5 bars)
            mom_1w = (today['Close'] - df_sym.iloc[-5]['Close']) / df_sym.iloc[-5]['Close'] * 100 if len(df_sym) >= 5 else 0.0
            # 1-Day Change
            mom_1d = (prev['Close'] - prev['Open']) / prev['Open'] * 100

            # 2. Opening Gap %
            gap_pct = (today['Open'] - prev['Close']) / prev['Close'] * 100

            # 3. Volatility & ATR %
            df_sym['TR'] = np.maximum(
                df_sym['High'] - df_sym['Low'],
                np.maximum(abs(df_sym['High'] - df_sym['Close'].shift(1)), abs(df_sym['Low'] - df_sym['Close'].shift(1)))
            )
            atr_14 = df_sym['TR'].rolling(14).mean().iloc[-1]
            atr_pct = (atr_14 / today['Close']) * 100 if today['Close'] > 0 else 1.0

            # 4. 20-Day SMA Trend
            sma_20 = df_sym['Close'].rolling(20).mean().iloc[-1]
            above_sma20 = today['Close'] > sma_20

            # 5. Composite Intraday Opportunity Scoring (0 to 100)
            score = 50.0

            if market_bias == 'BULLISH':
                # Reward stocks leading higher
                if mom_1m > 3.0: score += 12.0
                if mom_1w > 1.5: score += 12.0
                if above_sma20: score += 10.0
                if gap_pct > 0.20: score += 10.0
                elif gap_pct < -0.50: score -= 15.0
            elif market_bias == 'BEARISH':
                # Reward stocks breaking down
                if mom_1m < -3.0: score += 12.0
                if mom_1w < -1.5: score += 12.0
                if not above_sma20: score += 10.0
                if gap_pct < -0.20: score += 10.0
                elif gap_pct > 0.50: score -= 15.0
            else:
                # Range market: Reward high ATR expansion candidates
                if atr_pct > 1.5: score += 15.0

            score = max(0.0, min(100.0, score))

            candidates.append({
                'symbol': sym,
                'current_price': round(float(today['Close']), 2),
                'open_price': round(float(today['Open']), 2),
                'gap_pct': round(float(gap_pct), 2),
                'mom_1w_pct': round(float(mom_1w), 2),
                'mom_1m_pct': round(float(mom_1m), 2),
                'atr_pct': round(float(atr_pct), 2),
                'above_20d_sma': bool(above_sma20),
                'opportunity_score': round(score, 1),
                'trend_1m': 'BULLISH' if mom_1m > 0 else 'BEARISH',
                'trend_1w': 'BULLISH' if mom_1w > 0 else 'BEARISH'
            })

        df_cand = pd.DataFrame(candidates).sort_values(by='opportunity_score', ascending=False).reset_index(drop=True)
        return df_cand.head(top_n), df_cand
