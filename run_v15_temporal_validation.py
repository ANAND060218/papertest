"""
V15 Temporal Out-of-Sample Validation Benchmark
Evaluates Dual Momentum across 3 strict temporal slices:
  - In-Sample Period (2017 - 2022): 6 Years
  - Out-of-Sample Period (2023 - 2024): 2 Years
  - Untouched Recent OOS (2025 - 2026): 1.6 Years
"""
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.dual_momentum_engine import DualMomentumEngine


def evaluate_temporal_slices():
    db_path = os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
    history, trades = DualMomentumEngine.run_backtest(db_path, top_n=5, initial_capital=100000.0)

    df_hist = pd.DataFrame(history)
    df_hist['date'] = pd.to_datetime(df_hist['date'])

    slices = [
        ("In-Sample (2017-2022)", "2017-08-31", "2022-12-31"),
        ("OOS Validation (2023-2024)", "2023-01-31", "2024-12-31"),
        ("Final Untouched OOS (2025-2026)", "2025-01-31", "2026-08-14"),
        ("Full 10-Year Period (2017-2026)", "2017-08-31", "2026-08-14")
    ]

    print("\n" + "=" * 115)
    print("V15 TEMPORAL COHORT BREAKDOWN (CROSS-SECTIONAL DUAL MOMENTUM)")
    print("=" * 115)
    print(f"{'Cohort Period':<32} | {'Start Eq (Rs)':<14} | {'End Eq (Rs)':<14} | {'Return %':<10} | {'CAGR %':<10} | {'Trades':<8} | {'PF':<6}")
    print("-" * 115)

    for name, s_date, e_date in slices:
        sub_hist = df_hist[(df_hist['date'] >= pd.to_datetime(s_date)) & (df_hist['date'] <= pd.to_datetime(e_date))].reset_index(drop=True)
        if sub_hist.empty:
            continue

        start_eq = sub_hist.iloc[0]['portfolio_equity']
        end_eq = sub_hist.iloc[-1]['portfolio_equity']
        ret_pct = ((end_eq - start_eq) / start_eq) * 100
        days = (sub_hist.iloc[-1]['date'] - sub_hist.iloc[0]['date']).days
        years = days / 365.25
        cagr = (((end_eq / start_eq) ** (1 / years)) - 1) * 100 if years > 0 else ret_pct

        # Filter trades for this slice
        sub_trades = [t for t in trades if pd.to_datetime(s_date) <= pd.to_datetime(t['entry_date']) <= pd.to_datetime(e_date)]
        wins = [t for t in sub_trades if t['is_win']]
        gw = sum(t['net_pnl_rs'] for t in wins)
        gl = abs(sum(t['net_pnl_rs'] for t in sub_trades if not t['is_win']))
        pf = round(gw / gl, 3) if gl > 0 else 99.0

        print(f"{name:<32} | Rs {start_eq:>10,.2f} | Rs {end_eq:>10,.2f} | {ret_pct:>8.2f}% | {cagr:>8.2f}% | {len(sub_trades):<8} | {pf:>6.3f}")
    print("=" * 115)


if __name__ == "__main__":
    evaluate_temporal_slices()
