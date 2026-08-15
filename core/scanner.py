"""
V7 -- Daily Stock Opportunity Scanner
Selects the Top N candidate instruments each morning before market open.

Architecture (from roadmap):
  Universe -> Liquidity Filter -> Volatility/ATR Filter -> Momentum/Relative Strength -> Setup Readiness -> Ranked Candidates

Transparent Scoring:
  1. Liquidity Score: Avg 20-day daily turnover
  2. Volatility Score: ATR% adequate for intraday targets (>1.2%)
  3. Momentum Score: 5-day / 20-day relative strength vs Benchmark (^NSEI)
  4. Setup Score: Proximity to breakout, inside bar consolidation, volume anomaly
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.data_manager import DataManager
from core.feature_engine import FeatureEngine
from core.regime_detector import RegimeDetector


class DailyStockScanner:
    """
    Scans the defined universe to rank and pick the best candidates for the trading session.
    """

    def __init__(self, min_turnover_cr=20.0, min_atr_pct=1.0):
        self.min_turnover_cr = min_turnover_cr  # Minimum 20 Cr daily turnover
        self.min_atr_pct = min_atr_pct          # Minimum 1.0% ATR

    def scan_universe(self, universe=None, as_of_date=None):
        """
        Scans all symbols in the universe as of a specific date (or the latest available date).
        Returns a ranked DataFrame of trading candidates.
        """
        dm = DataManager()
        fe = FeatureEngine()
        regime_det = RegimeDetector()

        symbols = universe or config.UNIVERSE
        benchmark_sym = "^NSEI"

        # Load benchmark for relative strength calculation
        bench_df = dm.load_daily(benchmark_sym)
        bench_ret_5 = 0.0
        if bench_df is not None and len(bench_df) >= 5:
            if as_of_date:
                bench_df = bench_df[bench_df['Date'] <= as_of_date]
            bench_ret_5 = (bench_df['Close'].iloc[-1] - bench_df['Close'].iloc[-5]) / bench_df['Close'].iloc[-5] * 100

        candidates = []

        for sym in symbols:
            df = dm.load_daily(sym)
            if df is None or len(df) < 50:
                continue

            if as_of_date:
                df = df[df['Date'] <= as_of_date].copy()
                if len(df) < 50:
                    continue

            df = fe.build_features(df)
            df = regime_det.classify_regimes(df)

            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            close = last_row['Close']
            volume = last_row['Volume']

            # 1. Liquidity check: 20-day average turnover in Crores
            df['turnover_cr'] = (df['Close'] * df['Volume']) / 1e7
            avg_turnover_cr = df['turnover_cr'].rolling(20).mean().iloc[-1]

            if avg_turnover_cr < self.min_turnover_cr:
                continue  # Filter out illiquid stocks

            # 2. Volatility: ATR%
            atr_pct = last_row['feat_atr_pct'] * 100
            if atr_pct < self.min_atr_pct:
                continue  # Filter out flat, low-volatility stocks

            # 3. Momentum & Relative Strength vs NIFTY
            ret_5d = (close - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
            rel_strength = ret_5d - bench_ret_5

            # 4. Setup Readiness Scores
            # Inside bar setup ready
            is_inside_bar = (last_row['High'] < prev_row['High']) and (last_row['Low'] > prev_row['Low'])

            # Proximity to 20-day high (within 1.5% of breakout)
            high_20 = df['High'].rolling(20).max().iloc[-1]
            dist_to_breakout_pct = ((high_20 - close) / close) * 100

            # Volume surge ratio
            vol_ratio = last_row['feat_volume_ratio']

            # Composite Opportunity Score (0 to 100)
            # Weights: Liquidity (20%), Volatility (25%), Relative Strength (25%), Setup Readiness (30%)
            score = 0.0

            # Liquidity score (cap at 100 Cr)
            score += min(avg_turnover_cr / 100.0, 1.0) * 20.0

            # Volatility score (optimal ATR 1.5% to 3.5%)
            if 1.2 <= atr_pct <= 4.0:
                score += (atr_pct / 3.0) * 25.0
            else:
                score += 10.0

            # Relative Strength (positive outperformance awarded)
            if rel_strength > 0:
                score += min(rel_strength / 3.0, 1.0) * 25.0
            else:
                score += max(0, (rel_strength + 3.0) / 3.0) * 10.0

            # Setup Readiness score
            setup_bonus = 0.0
            if is_inside_bar:
                setup_bonus += 15.0
            if 0.0 <= dist_to_breakout_pct <= 1.5:
                setup_bonus += 10.0
            if vol_ratio > 1.3:
                setup_bonus += 5.0
            score += min(setup_bonus, 30.0)

            candidates.append({
                'Symbol': sym,
                'LTP': round(close, 2),
                'Regime': last_row['regime'],
                'AvgTurnoverCr': round(avg_turnover_cr, 1),
                'ATR%': round(atr_pct, 2),
                'Return5D%': round(ret_5d, 2),
                'RelStrength%': round(rel_strength, 2),
                'InsideBar': 'YES' if is_inside_bar else 'NO',
                'DistTo20DHigh%': round(dist_to_breakout_pct, 2),
                'VolRatio': round(vol_ratio, 2) if pd.notna(vol_ratio) else 1.0,
                'OpportunityScore': round(score, 1),
            })

        cand_df = pd.DataFrame(candidates)
        if not cand_df.empty:
            cand_df = cand_df.sort_values('OpportunityScore', ascending=False).reset_index(drop=True)
            cand_df['Rank'] = range(1, len(cand_df) + 1)
            # Reorder columns
            cols = ['Rank', 'Symbol', 'OpportunityScore', 'Regime', 'LTP', 'ATR%', 'RelStrength%', 'InsideBar', 'DistTo20DHigh%', 'AvgTurnoverCr']
            cand_df = cand_df[cols]

        return cand_df


if __name__ == "__main__":
    scanner = DailyStockScanner()
    print("=" * 100)
    print("V7 DAILY STOCK OPPORTUNITY SCANNER")
    print("=" * 100)
    top_candidates = scanner.scan_universe()
    print(top_candidates.to_string(index=False))
