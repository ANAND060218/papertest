"""
Master Live Forward Paper Trading Runner (V8.5)
Runs the real-time forward paper trading execution loop during NSE market hours (09:15 - 15:30 IST).

Usage:
  python run_paper_trade.py              # Runs live loop (auto-detects market open/closed)
  python run_paper_trade.py --dry-run    # Runs 1 diagnostic poll cycle on latest data
"""
import sys
import os
import time
import argparse
from datetime import datetime, time as dtime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from execution.live_engine import LiveTradingEngine
from v8_forward_report import ForwardValidationAuditor


def is_market_open():
    now = datetime.now()
    # Monday=0, Sunday=6
    if now.weekday() >= 5:
        return False
    cur_time = now.time()
    return dtime(9, 15) <= cur_time <= dtime(15, 30)


def main():
    parser = argparse.ArgumentParser(description="V8.5 Live Forward Paper Trading Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 diagnostic polling cycle regardless of market hours")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds (default: 300s = 5m)")
    args = parser.parse_args()

    print("=" * 90)
    print("INTRADAY LIVE FORWARD PAPER TRADING RUNNER (V8.5)")
    print(f"Current System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"Monitored Core Universe: {config.UNIVERSE}")
    print("=" * 90)

    engine = LiveTradingEngine(mode="PAPER", initial_capital=config.INITIAL_CAPITAL)

    if args.dry_run:
        print("\n[DRY RUN MODE] Executing 1 diagnostic health-check cycle...")
        engine.poll_and_evaluate_cycle()
        print("\n[DRY RUN COMPLETE] Live engine and data pipeline verified.")
        return

    # Check Market Hours
    market_active = is_market_open()
    if not market_active:
        print("\n[STATUS: MARKET CLOSED]")
        print("  NSE Trading Hours: Monday to Friday, 09:15 to 15:30 IST.")
        print("  Current Time is outside active market hours.")
        print("\n  Executing a health-check poll against latest available session data...")
        engine.poll_and_evaluate_cycle()

        print("\n[READY] The system is configured and ready.")
        print("To run the live session tomorrow, launch:")
        print("  python run_paper_trade.py")
        return

    # Live Market Loop (09:15 to 15:30 IST)
    print(f"\n[STATUS: MARKET OPEN] Entering live 5-minute polling loop (Interval: {args.interval}s)...")
    cycle_num = 1

    try:
        while is_market_open():
            now = datetime.now()
            print(f"\n=== [Cycle #{cycle_num}] {now.strftime('%H:%M:%S IST')} ===")
            engine.poll_and_evaluate_cycle()
            cycle_num += 1

            # Sleep until next bar close
            time.sleep(args.interval)

        print("\n[MARKET CLOSED] Session completed. Performing end-of-day square-off audit...")
        # Auto square-off any positions left at 15:15
        engine.poll_and_evaluate_cycle(current_time=datetime.now().replace(hour=15, minute=16))

        # Generate Forward Validation Report
        print("\nGenerating end-of-session Forward Validation Report...")
        auditor = ForwardValidationAuditor()
        auditor.generate_full_audit_report()

    except KeyboardInterrupt:
        print("\n[USER TERMINATION] Paper trading session stopped by user.")
        print("All positions and logs are preserved in data/trade_journal.db.")


if __name__ == "__main__":
    main()
