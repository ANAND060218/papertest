"""
V16.1 Phase 2: Friction Stress Test + Walk-Forward OOS + Improvement Experiments
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
    return stat + slip_cost


def run_phase2():
    print("=" * 100)
    print("V16.1 PHASE 2: DEEP DIAGNOSTIC ON ALL 3 STRATEGIES")
    print("=" * 100)

    results_dir = os.path.join(BASE_DIR, "results")

    # Load all trade CSVs
    for strat in ['orb', 'vwap', 'gap_fade']:
        csv_path = os.path.join(results_dir, f"v16_{strat}_trades.csv")
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {strat} - no trades file found")
            continue

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"\n{'#' * 100}")
        print(f"STRATEGY: {strat.upper()} ({len(df)} trades)")
        print(f"{'#' * 100}")

        # Walk-Forward 6-month windows
        print(f"\n  --- Walk-Forward OOS (6-month windows) ---")
        print(f"  {'Period':<25} | {'Trades':<7} | {'Win %':<7} | {'PF':<7} | {'Net P&L':<12}")
        print(f"  {'-'*75}")
        min_dt = df['date'].min()
        max_dt = df['date'].max()
        cur = min_dt
        while cur < max_dt:
            end = cur + pd.DateOffset(months=6)
            w = df[(df['date'] >= cur) & (df['date'] < end)]
            if len(w) > 0:
                n = len(w)
                wr = (w['net'] > 0).sum() / n * 100
                gw = w[w['net'] > 0]['net'].sum()
                gl = abs(w[w['net'] <= 0]['net'].sum())
                pf = round(gw / gl, 3) if gl > 0 else 99.0
                label = f"{cur.strftime('%Y-%m')} to {end.strftime('%Y-%m')}"
                print(f"  {label:<25} | {n:<7} | {wr:>5.1f}% | {pf:>7.3f} | Rs {w['net'].sum():>9,.0f}")
            cur = end

        # Per-symbol breakdown
        print(f"\n  --- Per-Symbol Breakdown ---")
        print(f"  {'Symbol':<18} | {'Trades':<7} | {'Win %':<7} | {'PF':<7} | {'Net P&L':<12}")
        print(f"  {'-'*70}")
        for sym in sorted(df['sym'].unique()):
            sub = df[df['sym'] == sym]
            n = len(sub)
            wr = (sub['net'] > 0).sum() / n * 100
            gw = sub[sub['net'] > 0]['net'].sum()
            gl = abs(sub[sub['net'] <= 0]['net'].sum())
            pf = round(gw / gl, 3) if gl > 0 else 99.0
            print(f"  {sym:<18} | {n:<7} | {wr:>5.1f}% | {pf:>7.3f} | Rs {sub['net'].sum():>9,.0f}")

        # Gross vs Net analysis
        total_gross = df['gross'].sum()
        total_costs = df['costs'].sum()
        total_net = df['net'].sum()
        print(f"\n  --- Friction Impact ---")
        print(f"  Total Gross P&L:   Rs {total_gross:>10,.0f}")
        print(f"  Total Costs:       Rs {total_costs:>10,.0f}")
        print(f"  Total Net P&L:     Rs {total_net:>10,.0f}")
        print(f"  Cost / |Gross|:    {abs(total_costs)/max(1,abs(total_gross))*100:>6.1f}%")
        print(f"  Average cost/trade: Rs {total_costs/len(df):>8,.0f}")

        # Trade size analysis
        if 'entry' in df.columns:
            avg_entry = df['entry'].mean()
            avg_qty = df['qty'].mean()
            print(f"  Average entry price: Rs {avg_entry:>8,.0f}")
            print(f"  Average qty: {avg_qty:>8,.0f}")

        # Distribution of net P&L per trade
        print(f"\n  --- Net P&L Distribution ---")
        pcts = [10, 25, 50, 75, 90]
        for p in pcts:
            v = np.percentile(df['net'], p)
            print(f"  P{p:<3}: Rs {v:>8,.0f}")
        print(f"  Mean:  Rs {df['net'].mean():>8,.0f}")
        print(f"  Stdev: Rs {df['net'].std():>8,.0f}")

    # ============================================
    # COMBINED PORTFOLIO: What if we run ALL strategies together?
    # ============================================
    print(f"\n{'=' * 100}")
    print("COMBINED PORTFOLIO: All Strategies Running Together")
    print(f"{'=' * 100}")

    all_dfs = []
    for strat in ['orb', 'vwap', 'gap_fade']:
        csv_path = os.path.join(results_dir, f"v16_{strat}_trades.csv")
        if os.path.exists(csv_path):
            d = pd.read_csv(csv_path)
            d['strat'] = strat
            all_dfs.append(d)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        # Daily P&L
        daily = combined.groupby('date')['net'].sum().reset_index()
        daily = daily.sort_values('date')
        daily['cum'] = daily['net'].cumsum()
        daily['peak'] = daily['cum'].cummax()
        daily['dd'] = daily['cum'] - daily['peak']

        print(f"  Total trades: {len(combined)}")
        print(f"  Total Net P&L: Rs {combined['net'].sum():>10,.0f}")
        print(f"  Trading days with at least 1 trade: {len(daily)}")
        print(f"  Avg daily P&L: Rs {daily['net'].mean():>8,.0f}")
        print(f"  Max drawdown: Rs {daily['dd'].min():>10,.0f}")
        print(f"  Win days: {(daily['net'] > 0).sum()} / {len(daily)} ({(daily['net'] > 0).sum()/len(daily)*100:.1f}%)")

    print(f"\n{'=' * 100}")
    print("CONCLUSION: Is there a tradeable intraday edge?")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    run_phase2()
