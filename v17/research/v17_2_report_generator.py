"""
V17.2 Multi-Year OOS Report Generator
Provides a deep breakdown of the V17 strategy performance:
- Overall Metrics
- Year-by-Year Breakdown
- Regime Breakdown
- Strategy Breakdown
- Stock Breakdown
- Time-of-Day Analysis
- Friction Sensitivity Test
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def intraday_costs_recalc(entry_p, exit_p, qty, direction, slip):
    eb = entry_p * (1 + slip) if direction == 'LONG' else entry_p * (1 - slip)
    es = exit_p * (1 - slip) if direction == 'LONG' else exit_p * (1 + slip)
    
    bv, sv = eb * qty, es * qty
    if direction == 'SHORT':
        bv, sv = es * qty, eb * qty
        
    tv = bv + sv
    brk = min(20, bv * 0.0003) + min(20, sv * 0.0003)
    stt = sv * 0.00025
    exc = tv * 0.0000345
    gst = (brk + exc) * 0.18
    stamp = bv * 0.00003
    sebi = tv * 0.000001
    stat = brk + stt + exc + gst + stamp + sebi
    
    gross = (exit_p - entry_p) * qty if direction == 'LONG' else (entry_p - exit_p) * qty
    net = (es - eb) * qty - stat if direction == 'LONG' else (eb - es) * qty - stat
    return gross, net, stat + abs((es - exit_p) * qty) + abs((eb - entry_p) * qty)

def generate_report():
    csv_path = os.path.join(BASE_DIR, "results", "v17_multi_year_trades.csv")
    if not os.path.exists(csv_path):
        print("Trade file not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No trades in file.")
        return
        
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    print("=" * 100)
    print("V17.2 MULTI-YEAR OOS VALIDATION REPORT")
    print("=" * 100)

    # 1. Overall
    n = len(df)
    gw = df[df['net'] > 0]['net'].sum()
    gl = abs(df[df['net'] <= 0]['net'].sum())
    pf = round(gw / gl, 3) if gl > 0 else 99.0
    win_r = (df['net'] > 0).sum() / n * 100
    avg_net = df['net'].mean()
    
    cum_ret = df['net'].cumsum()
    max_dd = (cum_ret.cummax() - cum_ret).max()
    
    print("\n[1. OVERALL METRICS]")
    print(f"  Total Trades:     {n}")
    print(f"  Win %:            {win_r:.1f}%")
    print(f"  Profit Factor:    {pf:.3f}")
    print(f"  Gross P&L:        Rs {df['gross'].sum():,.0f}")
    print(f"  Total Costs:      Rs {df['costs'].sum():,.0f}")
    print(f"  Net P&L:          Rs {df['net'].sum():,.0f}")
    print(f"  Expectancy/Trade: Rs {avg_net:.0f}")
    print(f"  Max Drawdown:     Rs {max_dd:,.0f}")

    # 2. Year-by-year
    print("\n[2. YEAR-BY-YEAR]")
    for y in sorted(df['year'].unique()):
        sub = df[df['year'] == y]
        p = round(sub[sub['net']>0]['net'].sum() / abs(sub[sub['net']<=0]['net'].sum()), 3) if sub[sub['net']<=0]['net'].sum() != 0 else 99.0
        print(f"  {y}: {len(sub)} trades | Win: {(sub['net']>0).mean()*100:.1f}% | PF: {p:.2f} | Net: Rs {sub['net'].sum():,.0f}")

    # 3. Market Regime Breakdown
    print("\n[3. REGIME BREAKDOWN]")
    for r in sorted(df['regime'].unique()):
        sub = df[df['regime'] == r]
        print(f"  {r:<8}: {len(sub):>4} trades | Win: {(sub['net']>0).mean()*100:>4.1f}% | Net: Rs {sub['net'].sum():>8,.0f}")

    # 4. Strategy Breakdown
    print("\n[4. STRATEGY BREAKDOWN]")
    for s in sorted(df['strat'].unique()):
        sub = df[df['strat'] == s]
        print(f"  {s:<22}: {len(sub):>4} trades | Win: {(sub['net']>0).mean()*100:>4.1f}% | Net: Rs {sub['net'].sum():>8,.0f}")

    # 5. Stock Breakdown
    print("\n[5. STOCK BREAKDOWN]")
    for s in sorted(df['sym'].unique()):
        sub = df[df['sym'] == s]
        print(f"  {s:<12}: {len(sub):>4} trades | Win: {(sub['net']>0).mean()*100:>4.1f}% | Net: Rs {sub['net'].sum():>8,.0f}")

    # 6. Time-of-Day Analysis
    print("\n[6. TIME-OF-DAY ANALYSIS]")
    bins = [0, 9*60+45, 10*60+30, 11*60+30, 13*60, 14*60+45]
    labels = ['< 09:45', '09:45-10:30', '10:30-11:30', '11:30-13:00', '13:00-14:45']
    df['time_min'] = df['entry_time'].dt.hour * 60 + df['entry_time'].dt.minute
    df['time_bracket'] = pd.cut(df['time_min'], bins=bins, labels=labels, right=False)
    
    for tb in labels:
        sub = df[df['time_bracket'] == tb]
        if len(sub) > 0:
            print(f"  {tb:<12}: {len(sub):>4} trades | Win: {(sub['net']>0).mean()*100:>4.1f}% | Exp/Trade: Rs {sub['net'].mean():>4.0f}")

    # 7. Friction Sensitivity
    print("\n[7. FRICTION SENSITIVITY TEST]")
    print(f"  {'Slippage':<10} | {'Win %':<6} | {'PF':<5} | {'Net P&L':<10} | {'Expectancy':<10}")
    print("  " + "-"*55)
    
    for slip in [0.0000, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015]:
        nets = []
        for _, row in df.iterrows():
            _, n, _ = intraday_costs_recalc(row['entry'], row['exit'], row['qty'], row['dir'], slip)
            nets.append(n)
        
        nets = np.array(nets)
        gw = nets[nets > 0].sum()
        gl = abs(nets[nets <= 0].sum())
        pf = round(gw / gl, 3) if gl > 0 else 99.0
        win_r = (nets > 0).mean() * 100
        
        print(f"  {slip*100:>5.2f}%    | {win_r:>5.1f}% | {pf:>5.2f} | Rs {nets.sum():>7,.0f} | Rs {nets.mean():>7.0f}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    generate_report()
