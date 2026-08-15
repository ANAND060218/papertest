"""
V17.1 Expectancy Report
Calculates exact Gross Expectancy, Friction, and Net Expectancy per trade.
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_report():
    csv_path = os.path.join(BASE_DIR, "results", "v17_1_5m_trades.csv")
    if not os.path.exists(csv_path):
        print("No trade file found at", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Trade file is empty.")
        return

    print("=" * 100)
    print("V17.1 5-MINUTE INTRADAY EXPECTANCY REPORT")
    print(f"Total Trades: {len(df)}")
    print("=" * 100)

    for strat in sorted(df['strat'].unique()):
        print(f"\n[{strat}]")
        sub = df[df['strat'] == strat]
        n = len(sub)
        
        gw = sub[sub['net'] > 0]['net'].sum()
        gl = abs(sub[sub['net'] <= 0]['net'].sum())
        pf = round(gw / gl, 3) if gl > 0 else 99.0
        win_r = (sub['net'] > 0).sum() / n * 100
        
        avg_gross = sub['gross'].mean()
        avg_cost = sub['costs'].mean()
        avg_net = sub['net'].mean()
        
        # Breakdown of Gross Expectancy
        gross_wins = sub[sub['gross'] > 0]
        gross_losses = sub[sub['gross'] <= 0]
        avg_gross_win = gross_wins['gross'].mean() if len(gross_wins) > 0 else 0
        avg_gross_loss = gross_losses['gross'].mean() if len(gross_losses) > 0 else 0
        gross_win_r = len(gross_wins) / n * 100

        print(f"  Gross Expectancy Profile:")
        print(f"    Avg Gross Win:    Rs {avg_gross_win:>8,.0f}")
        print(f"    Avg Gross Loss:   Rs {avg_gross_loss:>8,.0f}")
        print(f"    Gross Win Rate:   {gross_win_r:>8.1f}%")
        print(f"    ----------------------------------")
        print(f"    Gross Expectancy: Rs {avg_gross:>8,.0f} per trade")
        
        print(f"\n  Friction & Net Expectancy:")
        print(f"    Gross Expectancy: Rs {avg_gross:>8,.0f}")
        print(f"    Avg Friction:   - Rs {avg_cost:>8,.0f}")
        print(f"    ----------------------------------")
        print(f"    Net Expectancy:   Rs {avg_net:>8,.0f} per trade")
        
        print(f"\n  Net Performance:")
        print(f"    Net Win Rate:     {win_r:>8.1f}%")
        print(f"    Profit Factor:    {pf:>8.3f}")
        print(f"    Total Net P&L:    Rs {sub['net'].sum():>8,.0f}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    generate_report()
