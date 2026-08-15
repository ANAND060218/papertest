"""
V18.2 Final Options Matrix Runner
=================================
Sweeps across spread widths, entry times, and days-to-expiry buckets
using a single data load per day to maximize speed.

Also runs a CONTROL TEST: Same signal applied to the underlying spot
to determine whether options add value vs simply trading the index.

If no configuration survives with positive net expectancy, V18 is REJECTED.
"""
import sys, os, re, glob
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from v18.core.v18_cost_model import calculate_spread_costs

# ─── Configuration Matrix ───────────────────────────────────────────────
SPREAD_WIDTHS = [50, 100, 200, 300]
ENTRY_TIMES = ['09:20:00', '09:30:00', '09:45:00', '10:15:00']
EXIT_TIME = '15:15:00'
LOT_SIZE = 15  # BankNifty lot size

# DTE buckets: "near" = 0-2 DTE (weekly-like), "far" = 3+ DTE (monthly-like)
DTE_BUCKETS = {'near': (0, 2), 'far': (3, 30)}

# ─── Helpers ────────────────────────────────────────────────────────────
def parse_option_symbol(symbol):
    match = re.match(r"(BANKNIFTY|NIFTY)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)", symbol)
    if not match:
        return None
    return {
        'underlying': match.group(1),
        'expiry': datetime.strptime(match.group(2), "%d%b%y"),
        'strike': int(match.group(3)),
        'type': match.group(4)
    }

def find_atm_strike(spot_price, interval=100):
    return round(spot_price / interval) * interval

def get_chain_at_time(df_opt, time_str, date_str):
    """Extract option chain at a specific minute, parsing symbols only for that slice."""
    target = pd.to_datetime(f"{date_str} {time_str}")
    chain = df_opt[df_opt['datetime'] == target].copy()
    if chain.empty:
        return chain
    parsed = chain['symbol'].apply(parse_option_symbol).apply(pd.Series)
    chain = pd.concat([chain, parsed], axis=1)
    return chain

# ─── Core Engine ────────────────────────────────────────────────────────
def run_matrix(year_folder):
    """
    Single-pass matrix sweep across all configurations.
    Loads each day's data ONCE, tests all (entry_time × spread_width × dte_bucket) combos.
    """
    # Initialize results storage
    configs = []
    for width, entry_time, dte_label in product(SPREAD_WIDTHS, ENTRY_TIMES, DTE_BUCKETS.keys()):
        configs.append({
            'width': width,
            'entry_time': entry_time,
            'dte_bucket': dte_label,
            'trades': [],
        })
    
    # Also track underlying control trades (one per entry_time)
    control_configs = {et: [] for et in ENTRY_TIMES}
    
    print(f"V18.2 MATRIX RUNNER — Full Year {year_folder}")
    print(f"Configurations: {len(configs)} option combos + {len(ENTRY_TIMES)} control tests")
    print(f"Sweeping: widths={SPREAD_WIDTHS}, entries={ENTRY_TIMES}, DTE={list(DTE_BUCKETS.keys())}")
    print("-" * 80)
    
    days_processed = 0
    
    for month in range(1, 13):
        opt_dir = f"E:\\v18_options_data\\banknifty_data\\banknifty_options\\{year_folder}\\{month}"
        spot_dir = f"E:\\v18_options_data\\banknifty_data\\banknifty_spot\\{year_folder}\\{month}"
        
        if not os.path.exists(opt_dir):
            continue
        
        opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.csv")))
        
        for opt_file in opt_files:
            basename = os.path.basename(opt_file)
            spot_basename = basename.replace("options_", "spot")
            spot_file = os.path.join(spot_dir, spot_basename)
            
            if not os.path.exists(spot_file):
                continue
            
            # === LOAD DATA ONCE PER DAY ===
            df_spot = pd.read_csv(spot_file)
            df_spot['datetime'] = pd.to_datetime(df_spot['date'] + ' ' + df_spot['time'])
            
            df_opt = pd.read_csv(opt_file)
            df_opt = df_opt[df_opt['symbol'].str.endswith('CE') | df_opt['symbol'].str.endswith('PE')].copy()
            df_opt['datetime'] = pd.to_datetime(df_opt['date'] + ' ' + df_opt['time'])
            
            date_str = df_spot['date'].iloc[0]
            spot_open = df_spot.iloc[0]['open']
            
            days_processed += 1
            
            # === SWEEP ALL ENTRY TIMES ===
            for entry_time_str in ENTRY_TIMES:
                entry_dt = pd.to_datetime(f"{date_str} {entry_time_str}")
                exit_dt = pd.to_datetime(f"{date_str} {EXIT_TIME}")
                
                # Get spot at entry
                spot_row = df_spot[df_spot['datetime'] == entry_dt]
                if spot_row.empty:
                    continue
                spot_entry = spot_row.iloc[0]['close']
                
                # Get spot at exit (for control test)
                spot_exit_row = df_spot[df_spot['datetime'] == exit_dt]
                if spot_exit_row.empty:
                    continue
                spot_exit = spot_exit_row.iloc[0]['close']
                
                # Regime: entry price vs open
                is_bullish = spot_entry > spot_open
                
                # === CONTROL TEST: Underlying P&L ===
                if is_bullish:
                    underlying_pnl = (spot_exit - spot_entry) * LOT_SIZE
                else:
                    underlying_pnl = (spot_entry - spot_exit) * LOT_SIZE
                
                control_configs[entry_time_str].append({
                    'date': date_str,
                    'regime': 'BULL' if is_bullish else 'BEAR',
                    'gross_pnl': underlying_pnl,
                    'net_pnl': underlying_pnl  # No option costs for underlying
                })
                
                # === OPTIONS CHAIN ===
                chain_entry = get_chain_at_time(df_opt, entry_time_str, date_str)
                if chain_entry.empty:
                    continue
                
                chain_exit = get_chain_at_time(df_opt, EXIT_TIME, date_str)
                if chain_exit.empty:
                    continue
                
                atm_strike = find_atm_strike(spot_entry, 100)
                
                # Get expiry and compute DTE
                expiries = sorted(chain_entry['expiry'].unique())
                if not expiries:
                    continue
                target_expiry = expiries[0]
                trade_date = datetime.strptime(date_str, "%Y-%m-%d")
                dte = (target_expiry - trade_date).days
                
                # Determine DTE bucket
                dte_label = None
                for label, (lo, hi) in DTE_BUCKETS.items():
                    if lo <= dte <= hi:
                        dte_label = label
                        break
                if dte_label is None:
                    dte_label = 'far'  # default if > 30 somehow
                
                chain_entry_exp = chain_entry[chain_entry['expiry'] == target_expiry]
                chain_exit_exp = chain_exit[chain_exit['expiry'] == target_expiry]
                
                # === SWEEP ALL SPREAD WIDTHS ===
                for width in SPREAD_WIDTHS:
                    try:
                        if is_bullish:
                            buy_leg = chain_entry_exp[(chain_entry_exp['strike'] == atm_strike) & (chain_entry_exp['type'] == 'CE')].iloc[0]
                            sell_leg = chain_entry_exp[(chain_entry_exp['strike'] == atm_strike + width) & (chain_entry_exp['type'] == 'CE')].iloc[0]
                        else:
                            buy_leg = chain_entry_exp[(chain_entry_exp['strike'] == atm_strike) & (chain_entry_exp['type'] == 'PE')].iloc[0]
                            sell_leg = chain_entry_exp[(chain_entry_exp['strike'] == atm_strike - width) & (chain_entry_exp['type'] == 'PE')].iloc[0]
                    except IndexError:
                        continue
                    
                    entry_debit = buy_leg['close'] - sell_leg['close']
                    
                    try:
                        buy_exit = chain_exit_exp[(chain_exit_exp['strike'] == buy_leg['strike']) & (chain_exit_exp['type'] == buy_leg['type'])].iloc[0]
                        sell_exit = chain_exit_exp[(chain_exit_exp['strike'] == sell_leg['strike']) & (chain_exit_exp['type'] == sell_leg['type'])].iloc[0]
                    except IndexError:
                        continue
                    
                    exit_credit = buy_exit['close'] - sell_exit['close']
                    
                    # Costs
                    entry_costs, _, _ = calculate_spread_costs(buy_leg['close'], sell_leg['close'], LOT_SIZE)
                    exit_costs, _, _ = calculate_spread_costs(sell_exit['close'], buy_exit['close'], LOT_SIZE)
                    total_costs = entry_costs + exit_costs
                    
                    gross_pnl = (exit_credit - entry_debit) * LOT_SIZE
                    net_pnl = gross_pnl - total_costs
                    
                    # Find matching config and append trade
                    for cfg in configs:
                        if cfg['width'] == width and cfg['entry_time'] == entry_time_str and cfg['dte_bucket'] == dte_label:
                            cfg['trades'].append({
                                'date': date_str,
                                'dte': dte,
                                'gross_pnl': gross_pnl,
                                'costs': total_costs,
                                'net_pnl': net_pnl,
                            })
                            break
    
    # ─── RESULTS ────────────────────────────────────────────────────────
    print(f"\nDays processed: {days_processed}")
    print("\n" + "=" * 120)
    print("V18.2 OPTIONS MATRIX RESULTS — FULL YEAR " + year_folder)
    print("=" * 120)
    print(f"{'Width':>6} | {'Entry':>8} | {'DTE':>5} | {'Trades':>6} | {'WinRate':>7} | {'Gross P&L':>10} | {'Costs':>10} | {'Net P&L':>10} | {'NetExp/Trd':>10} | {'PF':>5} | {'MaxDD':>10}")
    print("-" * 120)
    
    best_net = -float('inf')
    best_cfg = None
    
    for cfg in configs:
        trades = cfg['trades']
        if not trades:
            print(f"{cfg['width']:>6} | {cfg['entry_time']:>8} | {cfg['dte_bucket']:>5} | {'NO TRADES':>6}")
            continue
        
        df_t = pd.DataFrame(trades)
        n = len(df_t)
        wins = (df_t['net_pnl'] > 0).sum()
        gross = df_t['gross_pnl'].sum()
        costs = df_t['costs'].sum()
        net = df_t['net_pnl'].sum()
        net_exp = net / n
        
        gp = df_t[df_t['net_pnl'] > 0]['net_pnl'].sum()
        gl = abs(df_t[df_t['net_pnl'] < 0]['net_pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
        
        cum = df_t['net_pnl'].cumsum()
        maxdd = (cum - cum.cummax()).min()
        
        print(f"{cfg['width']:>6} | {cfg['entry_time']:>8} | {cfg['dte_bucket']:>5} | {n:>6} | {wins/n*100:>6.1f}% | {gross:>10.0f} | {costs:>10.0f} | {net:>10.0f} | {net_exp:>10.1f} | {pf:>5.2f} | {maxdd:>10.0f}")
        
        if net > best_net:
            best_net = net
            best_cfg = cfg.copy()
            best_cfg['pf'] = pf
            best_cfg['net_exp'] = net_exp
    
    # ─── CONTROL TEST RESULTS ───────────────────────────────────────────
    print("\n" + "=" * 120)
    print("CONTROL TEST: SAME SIGNAL ON UNDERLYING (NO OPTIONS)")
    print("=" * 120)
    print(f"{'Entry':>8} | {'Trades':>6} | {'WinRate':>7} | {'Gross P&L':>10} | {'NetExp/Trd':>10} | {'PF':>5} | {'MaxDD':>10}")
    print("-" * 80)
    
    for et, trades in control_configs.items():
        if not trades:
            continue
        df_c = pd.DataFrame(trades)
        n = len(df_c)
        wins = (df_c['gross_pnl'] > 0).sum()
        gross = df_c['gross_pnl'].sum()
        gp = df_c[df_c['gross_pnl'] > 0]['gross_pnl'].sum()
        gl = abs(df_c[df_c['gross_pnl'] < 0]['gross_pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
        cum = df_c['gross_pnl'].cumsum()
        maxdd = (cum - cum.cummax()).min()
        
        print(f"{et:>8} | {n:>6} | {wins/n*100:>6.1f}% | {gross:>10.0f} | {gross/n:>10.1f} | {pf:>5.2f} | {maxdd:>10.0f}")
    
    # ─── VERDICT ────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    if best_cfg and best_net > 0:
        print(f"⚠️  BEST CONFIG: Width={best_cfg['width']}, Entry={best_cfg['entry_time']}, DTE={best_cfg['dte_bucket']}")
        print(f"    Net P&L: Rs {best_net:.0f}, PF: {best_cfg['pf']:.2f}, Net Exp/Trd: Rs {best_cfg['net_exp']:.1f}")
        print(f"    → REQUIRES OOS VALIDATION ON 2020-2022 AND 2024 BEFORE ACCEPTANCE")
    else:
        print(f"❌  NO CONFIGURATION PRODUCED POSITIVE NET P&L IN {year_folder}")
        print(f"    Best was: Width={best_cfg['width'] if best_cfg else 'N/A'}, Net={best_net:.0f}")
        print(f"    → V18 DIRECTIONAL DEBIT SPREADS: REJECTED")
    print("=" * 120)


if __name__ == "__main__":
    run_matrix("2023")
