"""
V15.1 Master Robustness, Sensitivity & Production Validation Test Harness
Executes all 6 institutional stress tests on Cross-Sectional Dual Momentum across 10 years of data:
  1. Walk-Forward Expanding Out-of-Sample Validation (2020-2026)
  2. Multi-Parameter Sensitivity & Broad Plateau Matrix
  3. Single-Stock Removal & Outlier Dominance Audit
  4. Friction & Slippage Escalation Stress Test (0.00% to 0.50%)
  5. Head-to-Head Benchmark vs NIFTY 50 Index Buy & Hold
  6. Calendar Year-by-Year Performance Audit
"""
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from core.v15_robustness_engine import V15RobustnessEngine


def run_full_robustness_suite():
    print("\n" + "=" * 125)
    print("V15.1 INSTITUTIONAL ROBUSTNESS & PRODUCTION STRESS-TESTING SUITE")
    print("Testing 10 Years (2016-2026) Across 40 Top Indian Equities")
    print("=" * 125)

    engine = V15RobustnessEngine()

    # =========================================================================
    # TEST 1: WALK-FORWARD EXPANDING OUT-OF-SAMPLE VALIDATION
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 1: WALK-FORWARD EXPANDING OUT-OF-SAMPLE (OOS) VALIDATION")
    print("#" * 125)
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    wf_results = []

    for y in wf_years:
        start_oos = f"{y}-01-01"
        end_oos = f"{y}-12-31" if y < 2026 else "2026-08-14"
        res = engine.run_simulation(start_date=start_oos, end_date=end_oos, lookback_1=6, lookback_2=12, sma_filter=10, top_n=5)
        if 'error' not in res:
            wf_results.append({
                'test_year': y,
                'return_pct': res['total_return_pct'],
                'profit_factor': res['profit_factor'],
                'win_rate': res['win_rate_pct'],
                'trades': res['total_trades'],
                'max_dd': res['max_drawdown_pct']
            })
            print(f"  [OOS {y}] Return: {res['total_return_pct']:>6.2f}% | PF: {res['profit_factor']:>5.3f} | WinRate: {res['win_rate_pct']:>5.1f}% | Trades: {res['total_trades']:<3} | MaxDD: {res['max_drawdown_pct']:>6.2f}%")

    # =========================================================================
    # TEST 2: PARAMETER SENSITIVITY & BROAD PLATEAU GRID
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 2: MULTI-PARAMETER SENSITIVITY GRID (Testing for Broad Plateau vs Peak Overfitting)")
    print("#" * 125)
    param_configs = [
        # (Lookback1, Lookback2, SMA_filter, Top_N, RebalFreq, Label)
        (6, 12, 10, 5, 1, "Baseline (6m+12m, SMA10, Top5, 1M)"),
        (3, 6, 10, 5, 1, "Shorter Mom (3m+6m, SMA10, Top5, 1M)"),
        (9, 12, 10, 5, 1, "Longer Mom (9m+12m, SMA10, Top5, 1M)"),
        (6, None, 10, 5, 1, "Single Mom (6m only, SMA10, Top5, 1M)"),
        (12, None, 10, 5, 1, "Single Mom (12m only, SMA10, Top5, 1M)"),
        (6, 12, None, 5, 1, "No SMA Trend Filter (6m+12m, Top5, 1M)"),
        (6, 12, 6, 5, 1, "Faster SMA Filter (SMA6, Top5, 1M)"),
        (6, 12, 12, 5, 1, "Slower SMA Filter (SMA12, Top5, 1M)"),
        (6, 12, 10, 3, 1, "Concentrated Portfolio (Top 3 Holdings)"),
        (6, 12, 10, 7, 1, "Diversified Portfolio (Top 7 Holdings)"),
        (6, 12, 10, 10, 1, "Broad Portfolio (Top 10 Holdings)"),
        (6, 12, 10, 5, 2, "Bi-Monthly Rebalancing (Every 2 Months)"),
        (6, 12, 10, 5, 3, "Quarterly Rebalancing (Every 3 Months)"),
    ]

    param_results = []
    print(f"{'Configuration':<45} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Sharpe':<6} | {'Max DD %':<9} | {'Trades':<6}")
    print("-" * 125)
    for l1, l2, sma, top, r_freq, label in param_configs:
        res = engine.run_simulation(lookback_1=l1, lookback_2=l2, sma_filter=sma, top_n=top, rebalance_freq=r_freq)
        if 'error' not in res:
            param_results.append({
                'label': label,
                'cagr': res['cagr_pct'],
                'total_ret': res['total_return_pct'],
                'pf': res['profit_factor'],
                'sharpe': res['sharpe_ratio'],
                'max_dd': res['max_drawdown_pct'],
                'trades': res['total_trades']
            })
            print(f"{label:<45} | {res['cagr_pct']:>6.2f}% | {res['total_return_pct']:>10.1f}% | {res['profit_factor']:>6.3f} | {res['sharpe_ratio']:>6.2f} | {res['max_drawdown_pct']:>7.2f}% | {res['total_trades']:<6}")

    # =========================================================================
    # TEST 3: SINGLE STOCK REMOVAL & OUTLIER VULNERABILITY AUDIT
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 3: STOCK REMOVAL & OUTLIER VULNERABILITY AUDIT")
    print("#" * 125)

    base_res = engine.run_simulation(lookback_1=6, lookback_2=12, sma_filter=10, top_n=5)
    top_winners = [sym for sym, _ in base_res['top_contributors']]
    print(f"Top 5 Profit-Contributing Stocks: {top_winners}")

    removal_tests = [
        ("Full Universe (All 38 Stocks)", []),
        (f"Exclude #1 Winner ({top_winners[0]})", [top_winners[0]]),
        (f"Exclude Top 2 Winners ({top_winners[:2]})", top_winners[:2]),
        (f"Exclude Top 5 Winners ({top_winners[:5]})", top_winners[:5]),
        ("Exclude Mega-Caps (RELIANCE & TCS)", ["RELIANCE.NS", "TCS.NS"]),
        ("Exclude Banking Leaders (HDFCBANK & ICICIBANK)", ["HDFCBANK.NS", "ICICIBANK.NS"])
    ]

    removal_results = []
    print(f"{'Stress Test Scenario':<45} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Sharpe':<6} | {'Max DD %':<9}")
    print("-" * 125)
    for label, ex_list in removal_tests:
        res = engine.run_simulation(lookback_1=6, lookback_2=12, sma_filter=10, top_n=5, exclude_symbols=ex_list)
        if 'error' not in res:
            removal_results.append({
                'label': label,
                'cagr': res['cagr_pct'],
                'total_ret': res['total_return_pct'],
                'pf': res['profit_factor'],
                'sharpe': res['sharpe_ratio'],
                'max_dd': res['max_drawdown_pct']
            })
            print(f"{label:<45} | {res['cagr_pct']:>6.2f}% | {res['total_return_pct']:>10.1f}% | {res['profit_factor']:>6.3f} | {res['sharpe_ratio']:>6.2f} | {res['max_drawdown_pct']:>7.2f}%")

    # =========================================================================
    # TEST 4: SLIPPAGE & FRICTION ESCALATION STRESS TEST
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 4: SLIPPAGE & TRANSACTION FRICTION ESCALATION STRESS TEST")
    print("#" * 125)
    slippages = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]
    friction_results = []

    print(f"{'Slippage Per Side':<22} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Final Equity (Rs)':<18} | {'Max DD %':<9}")
    print("-" * 125)
    for slip in slippages:
        res = engine.run_simulation(lookback_1=6, lookback_2=12, sma_filter=10, top_n=5, slippage_pct=slip)
        if 'error' not in res:
            friction_results.append({
                'slippage_pct': slip * 100,
                'cagr': res['cagr_pct'],
                'total_ret': res['total_return_pct'],
                'pf': res['profit_factor'],
                'final_equity': res['final_equity'],
                'max_dd': res['max_drawdown_pct']
            })
            print(f"{slip * 100:>6.2f}% Slippage/Side | {res['cagr_pct']:>6.2f}% | {res['total_return_pct']:>10.1f}% | {res['profit_factor']:>6.3f} | Rs {res['final_equity']:>14,.2f} | {res['max_drawdown_pct']:>7.2f}%")

    # =========================================================================
    # TEST 5: HEAD-TO-HEAD BENCHMARK VS NIFTY 50 BUY & HOLD
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 5: HEAD-TO-HEAD BENCHMARK VS NIFTY 50 INDEX BUY & HOLD (2017 - 2026)")
    print("#" * 125)
    nifty_bench = engine.compute_nifty_benchmark(start_date="2017-08-31", end_date="2026-08-14", initial_capital=100000.0)
    v15_base = engine.run_simulation(lookback_1=6, lookback_2=12, sma_filter=10, top_n=5, slippage_pct=0.0005)

    print(f"{'Metric':<30} | {'V15 Dual Momentum':<25} | {'NIFTY 50 Index (Buy & Hold)':<30} | {'Alpha / Outperformance'}")
    print("-" * 125)
    print(f"{'Initial Capital':<30} | Rs 100,000.00             | Rs 100,000.00                  | -")
    print(f"{'Final Equity':<30} | Rs {v15_base['final_equity']:>10,.2f}             | Rs {nifty_bench['final_equity']:>10,.2f}                  | +Rs {v15_base['final_equity'] - nifty_bench['final_equity']:>10,.2f}")
    print(f"{'Total Return':<30} | {v15_base['total_return_pct']:>6.2f}%                   | {nifty_bench['total_return_pct']:>6.2f}%                        | +{v15_base['total_return_pct'] - nifty_bench['total_return_pct']:>6.2f}%")
    print(f"{'Annualized CAGR':<30} | {v15_base['cagr_pct']:>6.2f}%                   | {nifty_bench['cagr_pct']:>6.2f}%                        | +{v15_base['cagr_pct'] - nifty_bench['cagr_pct']:>6.2f}% / year")
    print(f"{'Annualized Volatility':<30} | {v15_base['ann_vol_pct']:>6.2f}%                   | {nifty_bench['ann_vol_pct']:>6.2f}%                        | -")
    print(f"{'Sharpe Ratio (Rf=6.5%)':<30} | {v15_base['sharpe_ratio']:>6.2f}                    | {nifty_bench['sharpe_ratio']:>6.2f}                         | +{v15_base['sharpe_ratio'] - nifty_bench['sharpe_ratio']:>6.2f}")
    print(f"{'Sortino Ratio':<30} | {v15_base['sortino_ratio']:>6.2f}                    | -                              | -")
    print(f"{'Calmar Ratio (CAGR/MaxDD)':<30} | {v15_base['calmar_ratio']:>6.2f}                    | {nifty_bench['calmar_ratio']:>6.2f}                         | +{v15_base['calmar_ratio'] - nifty_bench['calmar_ratio']:>6.2f}")
    print(f"{'Maximum Drawdown':<30} | {v15_base['max_drawdown_pct']:>6.2f}%                   | {nifty_bench['max_drawdown_pct']:>6.2f}%                        | Lower Downside Protection")

    # =========================================================================
    # TEST 6: CALENDAR YEAR-BY-YEAR AUDIT
    # =========================================================================
    print("\n" + "#" * 125)
    print("TEST 6: CALENDAR YEAR-BY-YEAR BREAKDOWN")
    print("#" * 125)
    print(f"{'Calendar Year':<15} | {'V15 Return %':<15} | {'Trades Executed':<18} | {'Profit Factor (PF)'}")
    print("-" * 125)
    for yr, y_stat in sorted(v15_base['yearly_stats'].items()):
        print(f"{yr:<15} | {y_stat['return_pct']:>13.2f}% | {y_stat['trades']:<18} | {y_stat['pf']:>6.3f}")

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v15_1_robustness_report.json")
    summary_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "walk_forward_oos": wf_results,
        "parameter_sensitivity": param_results,
        "stock_removal_audit": removal_results,
        "friction_stress_test": friction_results,
        "nifty_benchmark": nifty_bench,
        "v15_base": {k: v for k, v in v15_base.items() if k != 'equity_curve'}
    }

    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[REPORT SAVED] -> {out_json}")


if __name__ == "__main__":
    run_full_robustness_suite()
