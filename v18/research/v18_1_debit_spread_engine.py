import sys
import os
import pandas as pd
from datetime import datetime
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from v18.core.v18_data_parser import load_options_data_for_day, get_point_in_time_chain, find_atm_strike
from v18.core.v18_cost_model import calculate_spread_costs

def run_debit_spread_backtest_for_year(year_folder):
    """
    Simulates a daily directional debit spread at 09:30 AM, exiting at 15:15 PM for full year.
    """
    trades = []
    print(f"Running V18.1 Directional Debit Spread Engine for Full Year {year_folder}...")
    
    for month in range(1, 13):
        month_folder = str(month)
        opt_dir = f"E:\\v18_options_data\\banknifty_data\\banknifty_options\\{year_folder}\\{month_folder}"
        spot_dir = f"E:\\v18_options_data\\banknifty_data\\banknifty_spot\\{year_folder}\\{month_folder}"
        
        if not os.path.exists(opt_dir):
            continue
            
        opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.csv")))
        
        for opt_file in opt_files:
            basename = os.path.basename(opt_file)
            spot_basename = basename.replace("options_", "spot")
            spot_file = os.path.join(spot_dir, spot_basename)
            
            if not os.path.exists(spot_file):
                continue
                
            # 1. Load Data
            df_spot = pd.read_csv(spot_file)
            df_spot['datetime'] = pd.to_datetime(df_spot['date'] + ' ' + df_spot['time'])
            df_opt = load_options_data_for_day(opt_file)
            
            # 2. Daily Setup at 09:30
            entry_time = pd.to_datetime(df_spot['date'].iloc[0] + ' 09:30:00')
            exit_time = pd.to_datetime(df_spot['date'].iloc[0] + ' 15:15:00')
            
            spot_at_entry_row = df_spot[df_spot['datetime'] == entry_time]
            if spot_at_entry_row.empty:
                continue
            spot_entry = spot_at_entry_row.iloc[0]['close']
            
            spot_open = df_spot.iloc[0]['open']
            
            # Simple Regime: Gap Up = Bullish, Gap Down = Bearish
            is_bullish = spot_entry > spot_open 
            
            # 3. Construct Options Chain at Entry
            chain_entry = get_point_in_time_chain(df_opt, entry_time)
            if chain_entry.empty:
                continue
                
            atm_strike = find_atm_strike(spot_entry, 100)
            
            # Select closest expiry
            expiries = sorted(chain_entry['expiry'].unique())
            if not expiries:
                continue
            target_expiry = expiries[0]
            
            chain_entry = chain_entry[chain_entry['expiry'] == target_expiry]
            
            # 4. Construct Debit Spread
            try:
                if is_bullish:
                    buy_leg = chain_entry[(chain_entry['strike'] == atm_strike) & (chain_entry['type'] == 'CE')].iloc[0]
                    sell_leg = chain_entry[(chain_entry['strike'] == atm_strike + 200) & (chain_entry['type'] == 'CE')].iloc[0]
                else:
                    buy_leg = chain_entry[(chain_entry['strike'] == atm_strike) & (chain_entry['type'] == 'PE')].iloc[0]
                    sell_leg = chain_entry[(chain_entry['strike'] == atm_strike - 200) & (chain_entry['type'] == 'PE')].iloc[0]
            except IndexError:
                continue
                
            entry_premium_paid = buy_leg['close'] - sell_leg['close']
            
            # 5. Fast forward to Exit at 15:15
            chain_exit = get_point_in_time_chain(df_opt, exit_time, target_expiry)
            if chain_exit.empty:
                continue
                
            try:
                buy_leg_exit = chain_exit[(chain_exit['strike'] == buy_leg['strike']) & (chain_exit['type'] == buy_leg['type'])].iloc[0]
                sell_leg_exit = chain_exit[(chain_exit['strike'] == sell_leg['strike']) & (chain_exit['type'] == sell_leg['type'])].iloc[0]
            except IndexError:
                continue
                
            exit_premium_received = buy_leg_exit['close'] - sell_leg_exit['close']
            
            # 6. Apply 2026 Cost Engine
            qty = 15 # BankNifty lot size
            
            entry_costs, _, _ = calculate_spread_costs(buy_leg['close'], sell_leg['close'], qty)
            exit_costs, _, _ = calculate_spread_costs(sell_leg_exit['close'], buy_leg_exit['close'], qty)
            total_costs = entry_costs + exit_costs
            
            gross_pnl = (exit_premium_received - entry_premium_paid) * qty
            net_pnl = gross_pnl - total_costs
            
            trades.append({
                'date': df_spot['date'].iloc[0],
                'month': month,
                'regime': 'BULLISH' if is_bullish else 'BEARISH',
                'buy_strike': buy_leg['strike'],
                'sell_strike': sell_leg['strike'],
                'net_premium_paid': entry_premium_paid,
                'net_premium_received': exit_premium_received,
                'gross_pnl': gross_pnl,
                'costs': total_costs,
                'net_pnl': net_pnl
            })
        
    df_trades = pd.DataFrame(trades)
    
    if df_trades.empty:
        print("No valid trades found.")
        return
        
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['net_pnl'] > 0])
    gross_total = df_trades['gross_pnl'].sum()
    net_total = df_trades['net_pnl'].sum()
    cost_total = df_trades['costs'].sum()
    
    # Calculate Max Drawdown
    df_trades['cumulative_net'] = df_trades['net_pnl'].cumsum()
    df_trades['high_water_mark'] = df_trades['cumulative_net'].cummax()
    df_trades['drawdown'] = df_trades['cumulative_net'] - df_trades['high_water_mark']
    max_dd = df_trades['drawdown'].min()
    
    # Calculate Profit Factor
    gross_profits = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_losses = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Average Winner / Loser
    avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean()
    
    print("\n" + "="*80)
    print(f"V18.1 DIRECTIONAL DEBIT SPREADS - FULL YEAR {year_folder}")
    print("="*80)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {(wins/total_trades)*100:.2f}%")
    print(f"Gross P&L:    Rs {gross_total:.2f}")
    print(f"Total Costs:  Rs {cost_total:.2f}")
    print(f"Net P&L:      Rs {net_total:.2f}")
    print(f"Net Exp/Trd:  Rs {net_total/total_trades:.2f}")
    print(f"Profit Factor:{profit_factor:.2f}")
    print(f"Max Drawdown: Rs {max_dd:.2f}")
    print(f"Avg Winner:   Rs {avg_win:.2f}")
    print(f"Avg Loser:    Rs {avg_loss:.2f}")
    
    print("\n--- Monthly Breakdown ---")
    monthly_stats = df_trades.groupby('month').agg(
        trades=('net_pnl', 'count'),
        net_pnl=('net_pnl', 'sum'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100)
    )
    for index, row in monthly_stats.iterrows():
        print(f"Month {index:02d}: {row['trades']} trades | Net P&L: Rs {row['net_pnl']:8.2f} | Win Rate: {row['win_rate']:5.1f}%")
    print("="*80)

if __name__ == "__main__":
    run_debit_spread_backtest_for_year("2023")
