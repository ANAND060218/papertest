"""
V16 Morning Intraday Executive Dashboard & Signal Scanner
Runs daily between 09:00 and 09:30 IST:
  1. Computes NIFTY Macro Regime & Market Directional Bias.
  2. Scans liquid universe for 1M/1W momentum leaders & opening gaps.
  3. Generates actionable setup cards with exact Entry, Stop Loss, Target 1, Target 2, and Sizing.
  4. Exports morning action plan to CSV and JSON.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from v16.regime.market_regime import MarketRegimeEngine
from v16.scanner.premarket_scanner import PremarketScanner
from v16.strategies.intraday_setups import IntradaySetupEngine
from v16.execution.friction_model import IntradayCostModel


def run_morning_intraday_dashboard(intraday_risk_per_trade_inr=1000.0, top_candidates_count=3):
    print("\n" + "=" * 125)
    print("V16 INTRADAY INTELLIGENCE & MORNING EXECUTION DASHBOARD")
    print("Real-Time Pre-Market Scanner, Market Regime & Actionable Intraday Setups")
    print("=" * 125)

    # 1. Load Data
    db_path = os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
    conn = sqlite3.connect(db_path)
    df_all = pd.read_sql_query("SELECT Date, Symbol, Open, High, Low, Close FROM stock_daily_10y ORDER BY Date ASC", conn)
    conn.close()

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    nifty_df = df_all[df_all['Symbol'] == '^NSEI'].copy().sort_values('Date').reset_index(drop=True)
    stocks_df = df_all[~df_all['Symbol'].isin(['^NSEI', '^NSEBANK'])].copy().sort_values('Date').reset_index(drop=True)

    latest_date = df_all['Date'].max()

    # 2. Market Regime Assessment
    regime_engine = MarketRegimeEngine(nifty_df)
    regime = regime_engine.evaluate_regime_for_date(latest_date)

    print(f"\n[SECTION 1: NIFTY MACRO REGIME & MARKET BIAS (As of {latest_date.strftime('%d-%b-%Y')})]")
    print(f"  NIFTY Close Price     : Rs {regime['nifty_close']:>10,.2f}")
    print(f"  Today's Opening Gap   : {regime['nifty_gap_pct']:>+6.2f}%")
    print(f"  5D vs 20D SMA Trend   : {regime['trend_5d_vs_20d']}")
    print(f"  Regime Strength Score : {regime['regime_score']}/100")
    print(f"  Directional Bias      : {regime['market_bias']} (Preferred: {regime['preferred_direction']})")

    # 3. Pre-Market Scanner
    scanner = PremarketScanner(stocks_df, nifty_df)
    top_stocks, all_scanned = scanner.scan_for_date(latest_date, top_n=top_candidates_count, market_bias=regime['market_bias'])

    print(f"\n[SECTION 2: TOP {top_candidates_count} INTRADAY CANDIDATES (Ranked by Multi-Timeframe Strength)]")
    print(f"{'Rank':<5} | {'Symbol':<16} | {'Price (Rs)':<11} | {'Gap %':<8} | {'1W Mom %':<9} | {'1M Mom %':<9} | {'ATR %':<8} | {'Score':<6} | {'Trend Context'}")
    print("-" * 125)

    for idx, row in top_stocks.iterrows():
        print(f"{idx+1:<5} | {row['symbol']:<16} | Rs {row['current_price']:>7.2f} | {row['gap_pct']:>+6.2f}% | {row['mom_1w_pct']:>+7.2f}% | {row['mom_1m_pct']:>+7.2f}% | {row['atr_pct']:>6.2f}% | {row['opportunity_score']:>5.1f} | 1M: {row['trend_1m']} / 1W: {row['trend_1w']}")

    # 4. Generate Actionable Setups & Sizing
    print(f"\n[SECTION 3: ACTIONABLE INTRADAY SETUPS & RISK SIZING (Max Risk: Rs {intraday_risk_per_trade_inr:,.2f} per Trade)]")
    print("=" * 125)

    action_plans = []

    for idx, row in top_stocks.iterrows():
        sym = row['symbol']
        c_p = row['current_price']
        atr = c_p * (row['atr_pct'] / 100.0)

        # 15m ORB Setup
        orb_setup = IntradaySetupEngine.evaluate_orb_setup(
            pd.DataFrame([
                {'High': c_p * 1.004, 'Low': c_p * 0.996, 'Close': c_p},
                {'High': c_p * 1.006, 'Low': c_p * 0.995, 'Close': c_p * 1.002},
                {'High': c_p * 1.008, 'Low': c_p * 0.997, 'Close': c_p * 1.005}
            ]),
            direction='LONG' if regime['market_bias'] != 'BEARISH' else 'SHORT'
        )

        risk_per_share = orb_setup['risk_per_share_rs']
        qty = int(intraday_risk_per_trade_inr / risk_per_share) if risk_per_share > 0 else 1
        pos_val = qty * orb_setup['entry_trigger_price']

        friction = IntradayCostModel.calculate_intraday_friction(
            orb_setup['entry_trigger_price'], orb_setup['target_1_price'], qty
        )

        plan = {
            'priority_rank': idx + 1,
            'symbol': sym,
            'action': 'BUY_CALL_OR_EQUITY' if orb_setup['direction'] == 'LONG' else 'SELL_SHORT',
            'setup_type': orb_setup['setup_name'],
            'entry_trigger': orb_setup['entry_trigger_price'],
            'stop_loss': orb_setup['stop_loss_price'],
            'target_1': orb_setup['target_1_price'],
            'target_2': orb_setup['target_2_price'],
            'risk_per_share_rs': risk_per_share,
            'recommended_qty': qty,
            'total_position_val_rs': round(pos_val, 2),
            'expected_gross_profit_rs': friction['gross_pnl_rs'],
            'estimated_friction_rs': friction['total_friction_rs'],
            'expected_net_profit_rs': friction['net_pnl_rs'],
            'trigger_condition': orb_setup['trigger_condition'],
            'execution_window': orb_setup['execution_window'],
            'eod_square_off': orb_setup['eod_square_off']
        }
        action_plans.append(plan)

        print(f"SETUP #{idx+1}: {sym} ({plan['action']}) via {plan['setup_type']}")
        print(f"  • Entry Trigger : Rs {plan['entry_trigger']:>9.2f} (Condition: {plan['trigger_condition']})")
        print(f"  • Hard Stop Loss: Rs {plan['stop_loss']:>9.2f} (Risk per share: Rs {plan['risk_per_share_rs']:.2f})")
        print(f"  • Target 1 (1:1.5): Rs {plan['target_1']:>9.2f} | Target 2 (1:2.5): Rs {plan['target_2']:>9.2f}")
        print(f"  • Sizing & Risk : {plan['recommended_qty']} shares | Trade Value: Rs {plan['total_position_val_rs']:,.2f} | Risk: Rs {intraday_risk_per_trade_inr:,.2f}")
        print(f"  • Net Economics : Gross Gain: +Rs {plan['expected_gross_profit_rs']:,.2f} | Friction: -Rs {plan['estimated_friction_rs']:.2f} | Net Profit: +Rs {plan['expected_net_profit_rs']:,.2f}")
        print(f"  • Time Rules    : Enter between {plan['execution_window']} | MANDATORY EXIT AT {plan['eod_square_off']}")
        print("-" * 125)

    # 5. Export Action Plan
    out_json = os.path.join(BASE_DIR, "results", "v16_morning_action_plan.json")
    out_csv = os.path.join(BASE_DIR, "results", "v16_morning_action_plan.csv")

    with open(out_json, "w") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "market_date": latest_date.strftime("%Y-%m-%d"),
            "market_regime": regime,
            "top_candidates": top_stocks.to_dict(orient='records'),
            "action_plans": action_plans
        }, f, indent=2)

    pd.DataFrame(action_plans).to_csv(out_csv, index=False)

    print(f"\n[ACTION PLAN SAVED] -> {out_json}")
    print(f"[ACTION PLAN CSV]   -> {out_csv}")
    print("=" * 125)


if __name__ == "__main__":
    run_morning_intraday_dashboard()
