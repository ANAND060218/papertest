import os
import sys
import datetime
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.database import init_db, register_strategy
from core.paper_broker import VirtualBroker
from production.v15_live import V15PaperEngine
from v19.production.v19_live import V19PaperEngine

def update_market_data():
    """
    Downloads latest market data using the existing download_market_data.py script
    """
    print("="*80)
    print("UPDATING MARKET DATA")
    print("="*80)
    
    script_path = os.path.join(BASE_DIR, 'scripts', 'download_market_data.py')
    if os.path.exists(script_path):
        # We assume the user has a script to download the daily EOD data to nifty_10year_stacked.csv
        # In a real environment we would call it. Here we'll just simulate it or call it if it exists.
        subprocess.run(["python", script_path], cwd=BASE_DIR)
    else:
        print("[WARN] Market data updater script not found. Proceeding with existing data.")

def run_daily_production():
    print("="*80)
    print("PAPER TRADING PRODUCTION CYCLE")
    print("="*80)
    
    # 1. Init Database & Register 4 Strategies
    init_db()
    
    # V15 Momentum
    register_strategy("V15_10K", 10000.0)
    register_strategy("V15_1M", 1000000.0)
    
    # V19 Stat Arb
    register_strategy("V19_10K", 10000.0)
    register_strategy("V19_1M", 1000000.0)
    
    # Common Broker
    broker = VirtualBroker(default_slippage_pct=0.0005)
    
    today = datetime.datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    
    print(f"\n[RUNNING FOR DATE: {today_str}]")
    
    # --- V15 Execution (Only runs logic if it's month end) ---
    # Technically V15 should only rebalance at month end, 
    # but the engine can safely be called daily and we can restrict it,
    # or let it run its monthly logic (it uses resample('ME')).
    # For now, we just force the rebalance check.
    # In a true daily loop, we check if today is the last trading day of the month.
    # For paper trading, we will just run the rebalance on the 1st of every month.
    if today.day == 1 or True: # Force run for testing, real life: today.day == 1
        print("\n--- Running V15 (10K) ---")
        v15_10k = V15PaperEngine(broker, strategy_id="V15_10K")
        v15_10k.run_monthly_rebalance(today_str)
        
        print("\n--- Running V15 (1M) ---")
        v15_1m = V15PaperEngine(broker, strategy_id="V15_1M")
        v15_1m.run_monthly_rebalance(today_str)
        
    # --- V19 Execution (Runs daily) ---
    print("\n--- Running V19 (10K) ---")
    v19_10k = V19PaperEngine(broker, strategy_id="V19_10K", capital_per_leg=5000.0)
    v19_10k.run_daily_cycle(today_str)
    
    print("\n--- Running V19 (1M) ---")
    v19_1m = V19PaperEngine(broker, strategy_id="V19_1M", capital_per_leg=100000.0)
    v19_1m.run_daily_cycle(today_str)
    
    print("\n" + "="*80)
    print("PRODUCTION CYCLE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    update_market_data()
    run_daily_production()
