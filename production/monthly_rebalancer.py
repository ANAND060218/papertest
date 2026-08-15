"""
V15.2.1 Canonical Monthly Rebalancer & Execution Engine
Orchestrates the frozen Top-10 Dual Momentum shadow/paper trading pipeline:
  1. Ingests 10-year daily database and identifies the exact latest available trading bar (2026-08-14).
  2. Pre-Trade RiskGuard Validation (Data Integrity & Structural Kill-Switches).
  3. Computes 6M & 12M Dual-Momentum Scores and 10M SMA Trend Filter.
  4. SignalReporter generates the transparent 'Why' Decision Audit Sheet.
  5. Immutable ExecutionLogger records audit trail in results/execution_log.csv.
  6. ShadowPortfolio updates mark-to-market positions and cash balance in results/shadow_positions.csv.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from production.risk_guard import RiskGuard
from production.signal_report import SignalReporter
from production.execution_log import ExecutionLogger
from production.shadow_portfolio import ShadowPortfolio


def run_monthly_rebalance(portfolio_capital=100000.0, execute_shadow=True):
    print("\n" + "=" * 125)
    print("V15.2.1 CANONICAL PRODUCTION MONTHLY REBALANCER (TOP 10 DIVERSIFIED MOMENTUM)")
    print("Point-in-Time Causal Execution & Pre-Trade Safety Protocol")
    print("=" * 125)

    # 1. Load Frozen Configuration
    config_path = os.path.join(BASE_DIR, "data", "production_config.json")
    with open(config_path, "r") as f:
        prod_cfg = json.load(f)

    # 2. Ingest Data
    db_path = os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
    conn = sqlite3.connect(db_path)
    df_all = pd.read_sql_query("SELECT Date, Symbol, Close FROM stock_daily_10y ORDER BY Date ASC", conn)
    conn.close()

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    stocks_df = df_all[~df_all['Symbol'].isin(['^NSEI', '^NSEBANK'])].copy()

    # Get exact latest market bar date
    true_latest_bar_date = df_all['Date'].max()

    price_matrix = stocks_df.pivot(index='Date', columns='Symbol', values='Close').ffill()
    monthly_df = price_matrix.resample('ME').last().ffill()

    # 3. Risk Guard Pre-Trade Data Validation (Kill-Switch Check)
    guard = RiskGuard(prod_cfg)
    data_valid, data_errors = guard.validate_market_data(monthly_df, monthly_df.index[-1])

    if not data_valid:
        print("\n[CRITICAL KILL-SWITCH TRIGGERED]")
        for err in data_errors:
            print(f"  ❌ {err}")
        print("\nAborting rebalance safely. No orders generated.")
        return False

    print(f"[DATA INTEGRITY PASSED] {len(price_matrix.columns)} symbols verified. Market Bar Date: {true_latest_bar_date.strftime('%Y-%m-%d')}.")

    # 4. Compute Signals
    l1 = prod_cfg['signal_parameters']['momentum_lookback_1_months']
    l2 = prod_cfg['signal_parameters']['momentum_lookback_2_months']
    sma_p = prod_cfg['signal_parameters']['absolute_trend_filter_months']
    top_n = prod_cfg['portfolio_construction']['holdings_count_top_n']

    mom_1 = monthly_df.pct_change(l1).iloc[-1]
    mom_2 = monthly_df.pct_change(l2).iloc[-1]
    combined_mom = 0.5 * mom_1 + 0.5 * mom_2
    sma_filter = monthly_df.rolling(sma_p).mean().iloc[-1]
    latest_prices = price_matrix.loc[true_latest_bar_date]

    ranked_data = []
    for sym in price_matrix.columns:
        c_price = latest_prices[sym]
        m_score = combined_mom.get(sym, np.nan)
        sma_val = sma_filter.get(sym, np.nan)
        is_above_sma = c_price > sma_val if pd.notna(sma_val) else False

        if pd.notna(m_score):
            ranked_data.append({
                'symbol': sym,
                'current_price': round(float(c_price), 2),
                'mom_6m_pct': round(float(mom_1.get(sym, 0) * 100), 2),
                'mom_12m_pct': round(float(mom_2.get(sym, 0) * 100), 2),
                'combined_momentum_score': round(float(m_score), 4),
                'sma_10m': round(float(sma_val), 2) if pd.notna(sma_val) else 0.0,
                'is_above_10m_sma': is_above_sma
            })

    df_ranked = pd.DataFrame(ranked_data).sort_values(by='combined_momentum_score', ascending=False).reset_index(drop=True)

    # 5. Generate Auditable 'Why' Report
    shadow = ShadowPortfolio(initial_capital=portfolio_capital)
    df_existing_pos = pd.read_csv(shadow.ledger_path) if os.path.exists(shadow.ledger_path) else pd.DataFrame()
    curr_holdings_dict = dict(zip(df_existing_pos['symbol'], df_existing_pos['shares_qty'])) if not df_existing_pos.empty else {}

    audit_report = SignalReporter.generate_detailed_report(df_ranked, top_n=top_n, current_holdings=curr_holdings_dict)

    # 6. Sizing & Position Allocation (Top 10 Equal Weight)
    selected_leaders = df_ranked[df_ranked['is_above_10m_sma']].head(top_n).copy()
    alloc_per_stock = portfolio_capital / top_n
    target_weight_pct = round(100.0 / top_n, 2)

    proposed_orders = []
    for idx, row in selected_leaders.iterrows():
        sym = row['symbol']
        c_p = row['current_price']
        qty = int(alloc_per_stock / c_p)
        val = qty * c_p
        action = "HOLD" if sym in curr_holdings_dict else "BUY"

        proposed_orders.append({
            'rank': idx + 1,
            'symbol': sym,
            'action': action,
            'current_price': c_p,
            'target_weight_pct': target_weight_pct,
            'target_value_inr': round(val, 2),
            'target_shares_qty': qty,
            'mom_6m_pct': row['mom_6m_pct'],
            'mom_12m_pct': row['mom_12m_pct'],
            'score': row['combined_momentum_score']
        })

    # 7. Risk Guard Pre-Trade Capital Validation
    alloc_valid, alloc_errors, alloc_warnings = guard.validate_allocations(proposed_orders, portfolio_capital)
    if not alloc_valid:
        print("\n[PRE-TRADE RISK VIOLATION]")
        for err in alloc_errors:
            print(f"  ❌ {err}")
        return False

    # 8. Print Executive Order Sheet
    print("\n" + "=" * 125)
    print(f"V15.2.1 CANONICAL REBALANCE ORDER SHEET (Market Bar Date: {true_latest_bar_date.strftime('%d-%b-%Y')})")
    print(f"Portfolio Capital: Rs {portfolio_capital:,.2f} | Strategy: Top {top_n} Dual Momentum Equal Weight (10% per Stock)")
    print("=" * 125)
    print(f"{'Rank':<5} | {'Action':<6} | {'Symbol':<16} | {'Price (Rs)':<11} | {'Mom 6M':<8} | {'Mom 12M':<9} | {'Score':<7} | {'Weight %':<9} | {'Target Value (Rs)':<18} | {'Qty'}")
    print("-" * 125)

    for o in proposed_orders:
        print(f"{o['rank']:<5} | {o['action']:<6} | {o['symbol']:<16} | Rs {o['current_price']:>7.2f} | {o['mom_6m_pct']:>6.2f}% | {o['mom_12m_pct']:>7.2f}% | {o['score']:>6.3f} | {o['target_weight_pct']:>7.2f}% | Rs {o['target_value_inr']:>14,.2f} | {o['target_shares_qty']:<6}")

    print("=" * 125)

    # 9. Print 'Why' Audit Rationale for Top Candidates
    print("\n" + "#" * 125)
    print("DECISION AUDIT TRAIL ('WHY' RATIONALE FOR TOP 12 RANKED STOCKS)")
    print("#" * 125)
    for _, row in audit_report.head(12).iterrows():
        print(f"  [{row['rank']}] {row['symbol']:<15} | Status: {row['status']:<20} | Trend: {row['trend_filter']} | {row['rationale_reason']}")

    # 10. Update Permanent Logs & Shadow Ledger
    logger = ExecutionLogger()
    df_logged = logger.log_rebalance_orders(proposed_orders, true_latest_bar_date, status="SHADOW_EXECUTED" if execute_shadow else "PENDING_APPROVAL")

    if execute_shadow:
        current_prices_dict = dict(zip(selected_leaders['symbol'], selected_leaders['current_price']))
        df_pos, tot_eq, cash_buf = shadow.update_from_rebalance(proposed_orders, current_prices_dict, true_latest_bar_date)
        print(f"\n[SHADOW LEDGER UPDATED] Total Portfolio Equity: Rs {tot_eq:,.2f} | Cash Buffer: Rs {cash_buf:,.2f} ({cash_buf / tot_eq * 100:.2f}%)")

    # Export artifacts
    orders_csv = os.path.join(BASE_DIR, "results", "monthly_orders.csv")
    audit_csv = os.path.join(BASE_DIR, "results", "signal_audit_report.csv")

    pd.DataFrame(proposed_orders).to_csv(orders_csv, index=False)
    audit_report.to_csv(audit_csv, index=False)

    print(f"\n[SAVED ORDER SHEET]  -> {orders_csv}")
    print(f"[SAVED AUDIT REPORT] -> {audit_csv}")
    print(f"[EXECUTION LOG]      -> {logger.log_path}")
    print(f"[SHADOW LEDGER]      -> {shadow.ledger_path}")

    return True


if __name__ == "__main__":
    run_monthly_rebalance(portfolio_capital=100000.0, execute_shadow=True)
