"""
V15.2 Master Verification, Survivorship-Bias Audit & Risk Architecture Suite
Executes the full institutional verification framework:
  1. Independent Replication & Point-in-Time Audit
  2. Survivorship-Bias Strict 2016 Universe Audit
  3. Simple 12M Naive Momentum Control Benchmark
  4. 2,000-Simulation Monte Carlo Trade Permutations
  5. 5-Way Portfolio Risk & Drawdown Mitigation Architecture
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
from core.v15_2_verification_engine import V15VerificationEngine


def run_full_verification():
    print("\n" + "=" * 125)
    print("V15.2 INSTITUTIONAL VERIFICATION, SURVIVORSHIP AUDIT & RISK SUITE")
    print("10-Year Cross-Sectional Dual Momentum System (2016 - 2026)")
    print("=" * 125)

    engine = V15VerificationEngine()

    # =========================================================================
    # PILLAR 1: INDEPENDENT REPLICATION & BASELINE AUDIT
    # =========================================================================
    print("\n" + "#" * 125)
    print("PILLAR 1: INDEPENDENT REPLICATION (Baseline V15.2 Dual Momentum)")
    print("#" * 125)
    base_res = engine.run_simulation(
        lookback_1=6, lookback_2=12, sma_filter=10, top_n=5,
        weighting_schema='EQUAL_WEIGHT', universe_filter='ALL', strategy_mode='DUAL_MOMENTUM'
    )
    print(f"  Final Equity       : Rs {base_res['final_equity']:>12,.2f} on Rs 100,000 capital")
    print(f"  Total Return       : +{base_res['total_return_pct']:.2f}%")
    print(f"  Annualized CAGR    : +{base_res['cagr_pct']:.2f}% per year")
    print(f"  Profit Factor (PF) : {base_res['profit_factor']:.3f}")
    print(f"  Sharpe Ratio (6.5%): {base_res['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio      : {base_res['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio       : {base_res['calmar_ratio']:.2f}")
    print(f"  Max Drawdown       : {base_res['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades       : {base_res['total_trades']} (Turnover ~16 trades/yr)")

    # =========================================================================
    # PILLAR 2: SURVIVORSHIP-BIAS AUDIT (Strict 2016-Eligible Constituents)
    # =========================================================================
    print("\n" + "#" * 125)
    print("PILLAR 2: SURVIVORSHIP-BIAS AUDIT (Banning All Post-2016 Index Additions)")
    print("#" * 125)
    surv_res = engine.run_simulation(
        lookback_1=6, lookback_2=12, sma_filter=10, top_n=5,
        weighting_schema='EQUAL_WEIGHT', universe_filter='SURVIVORSHIP_STRICT_2016', strategy_mode='DUAL_MOMENTUM'
    )
    print(f"{'Universe Specification':<45} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Sharpe':<6} | {'Max DD %':<9}")
    print("-" * 125)
    print(f"{'Full 40-Stock Universe':<45} | {base_res['cagr_pct']:>6.2f}% | {base_res['total_return_pct']:>10.1f}% | {base_res['profit_factor']:>6.3f} | {base_res['sharpe_ratio']:>6.2f} | {base_res['max_drawdown_pct']:>7.2f}%")
    print(f"{'Strict 2016 Point-in-Time (Survivorship-Free)':<45} | {surv_res['cagr_pct']:>6.2f}% | {surv_res['total_return_pct']:>10.1f}% | {surv_res['profit_factor']:>6.3f} | {surv_res['sharpe_ratio']:>6.2f} | {surv_res['max_drawdown_pct']:>7.2f}%")

    # =========================================================================
    # PILLAR 3: SIMPLE MOMENTUM CONTROL BENCHMARK
    # =========================================================================
    print("\n" + "#" * 125)
    print("PILLAR 3: SIMPLE MOMENTUM CONTROL BENCHMARK (Is Dual Momentum adding genuine alpha?)")
    print("#" * 125)
    naive_res = engine.run_simulation(
        top_n=5, weighting_schema='EQUAL_WEIGHT', universe_filter='ALL', strategy_mode='NAIVE_12M_CONTROL'
    )
    print(f"{'Strategy Architecture':<45} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Sharpe':<6} | {'Max DD %':<9}")
    print("-" * 125)
    print(f"{'V15.2 Dual Momentum (6M+12M + 10M SMA Filter)':<45} | {base_res['cagr_pct']:>6.2f}% | {base_res['total_return_pct']:>10.1f}% | {base_res['profit_factor']:>6.3f} | {base_res['sharpe_ratio']:>6.2f} | {base_res['max_drawdown_pct']:>7.2f}%")
    print(f"{'Naive 12-Month Momentum (No Dual / No SMA)':<45} | {naive_res['cagr_pct']:>6.2f}% | {naive_res['total_return_pct']:>10.1f}% | {naive_res['profit_factor']:>6.3f} | {naive_res['sharpe_ratio']:>6.2f} | {naive_res['max_drawdown_pct']:>7.2f}%")

    # =========================================================================
    # PILLAR 4: MONTE CARLO TRADE RESHUFFLING (2,000 PERMUTATIONS)
    # =========================================================================
    print("\n" + "#" * 125)
    print("PILLAR 4: MONTE CARLO RESHUFFLING (2,000 Iterations - Probability of Drawdown)")
    print("#" * 125)
    mc_res = V15VerificationEngine.run_monte_carlo(base_res['trades_list'], initial_capital=100000.0, n_simulations=2000)
    print(f"  CAGR Distribution   : 5th Pct: {mc_res['cagr_5th_pct']:>5.2f}% | Median: {mc_res['cagr_50th_pct']:>5.2f}% | 95th Pct: {mc_res['cagr_95th_pct']:>5.2f}%")
    print(f"  Max DD Distribution : 5th Pct: {mc_res['max_dd_5th_pct']:>5.2f}% | Median: {mc_res['max_dd_50th_pct']:>5.2f}% | 95th Pct: {mc_res['max_dd_95th_pct']:>5.2f}%")
    print(f"  Probability of Drawdown >= 30%: {mc_res['prob_dd_ge_30pct']}%")
    print(f"  Probability of Drawdown >= 40%: {mc_res['prob_dd_ge_40pct']}%")

    # =========================================================================
    # PILLAR 5: PORTFOLIO RISK & DRAWDOWN MITIGATION ARCHITECTURES
    # =========================================================================
    print("\n" + "#" * 125)
    print("PILLAR 5: PORTFOLIO RISK & DRAWDOWN MITIGATION ARCHITECTURES")
    print("#" * 125)

    risk_schemas = [
        ("Top 5 Equal Weight (20% each)", 5, 'EQUAL_WEIGHT'),
        ("Top 7 Equal Weight (14.3% each)", 7, 'EQUAL_WEIGHT'),
        ("Top 10 Equal Weight (10.0% each)", 10, 'EQUAL_WEIGHT'),
        ("Inverse Volatility Weighting (6M Vol)", 5, 'INVERSE_VOL'),
        ("Macro Regime Hedging (50% cash in Bear)", 5, 'MACRO_REGIME'),
        ("Macro Regime + Top 7 Diversification", 7, 'MACRO_REGIME'),
    ]

    risk_results = []
    print(f"{'Portfolio Risk Architecture':<45} | {'CAGR %':<8} | {'Total Ret %':<12} | {'PF':<6} | {'Sharpe':<6} | {'Sortino':<7} | {'Calmar':<6} | {'Max DD %':<9}")
    print("-" * 125)

    for label, n_top, schema in risk_schemas:
        r_res = engine.run_simulation(
            lookback_1=6, lookback_2=12, sma_filter=10, top_n=n_top,
            weighting_schema=schema, universe_filter='ALL', strategy_mode='DUAL_MOMENTUM'
        )
        if 'error' not in r_res:
            risk_results.append({
                'label': label,
                'cagr': r_res['cagr_pct'],
                'total_ret': r_res['total_return_pct'],
                'pf': r_res['profit_factor'],
                'sharpe': r_res['sharpe_ratio'],
                'sortino': r_res['sortino_ratio'],
                'calmar': r_res['calmar_ratio'],
                'max_dd': r_res['max_drawdown_pct']
            })
            print(f"{label:<45} | {r_res['cagr_pct']:>6.2f}% | {r_res['total_return_pct']:>10.1f}% | {r_res['profit_factor']:>6.3f} | {r_res['sharpe_ratio']:>6.2f} | {r_res['sortino_ratio']:>7.2f} | {r_res['calmar_ratio']:>6.2f} | {r_res['max_drawdown_pct']:>7.2f}%")

    # Save to disk
    out_json = os.path.join(config.RESULTS_DIR, "v15_2_verification_report.json")
    summary_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "pillar1_baseline": {k: v for k, v in base_res.items() if k not in ['trades_list', 'df_history']},
        "pillar2_survivorship": {k: v for k, v in surv_res.items() if k not in ['trades_list', 'df_history']},
        "pillar3_naive_control": {k: v for k, v in naive_res.items() if k not in ['trades_list', 'df_history']},
        "pillar4_monte_carlo": mc_res,
        "pillar5_risk_architectures": risk_results
    }

    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[REPORT SAVED] -> {out_json}")


if __name__ == "__main__":
    run_full_verification()
