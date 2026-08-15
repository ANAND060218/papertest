"""
V16.1 Optimized Causal Intraday Backtest Engine (Hourly Bars)
Pre-computes all daily aggregates and contexts up-front for speed.
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def intraday_costs(buy_p, sell_p, qty, slip=0.0003):
    eb = buy_p * (1 + slip)
    es = sell_p * (1 - slip)
    bv, sv = eb * qty, es * qty
    tv = bv + sv
    brk = min(20, bv * 0.0003) + min(20, sv * 0.0003)
    stt = sv * 0.00025
    exc = tv * 0.0000345
    gst = (brk + exc) * 0.18
    stamp = bv * 0.00003
    sebi = tv * 0.000001
    stat = brk + stt + exc + gst + stamp + sebi
    slip_cost = (eb - buy_p + sell_p - es) * qty
    gross = (sell_p - buy_p) * qty
    net = (es - eb) * qty - stat
    return gross, net, stat + slip_cost


def run_fast_backtest():
    print("V16.1 FAST INTRADAY VALIDATION", flush=True)

    db_path = os.path.join(BASE_DIR, "data", "historical_2year_hourly.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT Date, Open, High, Low, Close, Volume, symbol FROM universe_hourly_2year ORDER BY Date", conn
    )
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df['td'] = df['Date'].dt.date

    symbols = [s for s in df['symbol'].unique() if s != '^NSEI']
    nifty = df[df['symbol'] == '^NSEI'].copy()

    # Precompute NIFTY daily closes for regime
    nifty_daily = nifty.groupby('td').agg({'Close': 'last', 'Open': 'first'}).reset_index().sort_values('td')
    nifty_daily['sma5'] = nifty_daily['Close'].rolling(5).mean()
    nifty_daily['sma20'] = nifty_daily['Close'].rolling(20).mean()

    def get_regime(td):
        prior = nifty_daily[nifty_daily['td'] < td]
        if len(prior) < 20:
            return 'NEUTRAL'
        last = prior.iloc[-1]
        score = 50
        if last['Close'] > last['sma20']: score += 15
        else: score -= 15
        if last['sma5'] > last['sma20']: score += 10
        else: score -= 10
        if last['Close'] > last['Open']: score += 10
        else: score -= 10
        if score >= 60: return 'BULLISH'
        elif score <= 40: return 'BEARISH'
        return 'NEUTRAL'

    # Precompute stock daily aggregates
    stock_daily = {}
    for sym in symbols:
        sd = df[df['symbol'] == sym].groupby('td').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        ).reset_index().sort_values('td')
        sd['TR'] = np.maximum(sd['High'] - sd['Low'],
                   np.maximum(abs(sd['High'] - sd['Close'].shift(1)),
                              abs(sd['Low'] - sd['Close'].shift(1))))
        sd['ATR5'] = sd['TR'].rolling(5).mean()
        stock_daily[sym] = sd

    all_dates = sorted(df['td'].unique())
    all_trades = []

    strategies = {
        'ORB': [],
        'VWAP': [],
        'GAP_FADE': []
    }

    for td in all_dates:
        regime = get_regime(td)

        for sym in symbols:
            sd = stock_daily[sym]
            prior = sd[sd['td'] < td]
            if len(prior) < 5:
                continue
            prev_close = prior['Close'].iloc[-1]
            atr = prior['ATR5'].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                continue

            today_bars = df[(df['symbol'] == sym) & (df['td'] == td)].sort_values('Date').reset_index(drop=True)
            if len(today_bars) < 4:
                continue

            # ========== ORB ==========
            fb = today_bars.iloc[0]
            or_h, or_l = fb['High'], fb['Low']
            or_range = or_h - or_l
            if or_range > 0 and or_range / fb['Close'] * 100 >= 0.15:
                direction = 'LONG' if regime != 'BEARISH' else 'SHORT'
                triggered = False
                for i in range(1, min(4, len(today_bars))):
                    bar = today_bars.iloc[i]
                    if direction == 'LONG' and bar['High'] > or_h:
                        entry_p = or_h + 0.05
                        sl = or_l
                        risk = entry_p - sl
                        if risk > 0 and risk / entry_p * 100 <= 2.0:
                            tgt = entry_p + risk * 1.5
                            qty = max(1, int(50000 / entry_p))
                            exit_p, reason = entry_p, 'EOD'
                            for j in range(i + 1, len(today_bars)):
                                eb = today_bars.iloc[j]
                                if eb['Low'] <= sl: exit_p, reason = sl, 'SL'; break
                                if eb['High'] >= tgt: exit_p, reason = tgt, 'TGT'; break
                                exit_p = eb['Close']
                            g, n, c = intraday_costs(entry_p, exit_p, qty)
                            strategies['ORB'].append({
                                'date': str(td), 'sym': sym, 'dir': direction, 'regime': regime,
                                'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                                'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2)
                            })
                            triggered = True
                        break
                    elif direction == 'SHORT' and bar['Low'] < or_l:
                        entry_p = or_l - 0.05
                        sl = or_h
                        risk = sl - entry_p
                        if risk > 0 and risk / entry_p * 100 <= 2.0:
                            tgt = entry_p - risk * 1.5
                            qty = max(1, int(50000 / entry_p))
                            exit_p, reason = entry_p, 'EOD'
                            for j in range(i + 1, len(today_bars)):
                                eb = today_bars.iloc[j]
                                if eb['High'] >= sl: exit_p, reason = sl, 'SL'; break
                                if eb['Low'] <= tgt: exit_p, reason = tgt, 'TGT'; break
                                exit_p = eb['Close']
                            g = (entry_p - exit_p) * qty
                            n_cost = intraday_costs(exit_p, entry_p, qty) if exit_p < entry_p else intraday_costs(entry_p, exit_p, qty)
                            g, n, c = intraday_costs(min(entry_p, exit_p), max(entry_p, exit_p), qty)
                            if direction == 'SHORT':
                                g = (entry_p - exit_p) * qty
                                n = g - c
                            strategies['ORB'].append({
                                'date': str(td), 'sym': sym, 'dir': direction, 'regime': regime,
                                'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                                'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2)
                            })
                            triggered = True
                        break

            # ========== GAP FADE ==========
            gap_pct = (fb['Open'] - prev_close) / prev_close * 100
            if abs(gap_pct) >= 1.0:
                if gap_pct > 1.0:
                    entry_p = fb['Close']
                    tgt = prev_close + (fb['Open'] - prev_close) * 0.5
                    sl = entry_p + atr * 0.5
                    qty = max(1, int(50000 / entry_p))
                    exit_p, reason = entry_p, 'EOD'
                    for j in range(1, len(today_bars)):
                        eb = today_bars.iloc[j]
                        if eb['High'] >= sl: exit_p, reason = sl, 'SL'; break
                        if eb['Low'] <= tgt: exit_p, reason = tgt, 'TGT'; break
                        exit_p = eb['Close']
                    g = (entry_p - exit_p) * qty  # Short
                    _, _, c = intraday_costs(min(entry_p, exit_p), max(entry_p, exit_p), qty)
                    n = g - c
                    strategies['GAP_FADE'].append({
                        'date': str(td), 'sym': sym, 'dir': 'SHORT', 'regime': regime,
                        'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                        'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2),
                        'gap_pct': round(gap_pct, 2)
                    })
                elif gap_pct < -1.0:
                    entry_p = fb['Close']
                    tgt = prev_close - (prev_close - fb['Open']) * 0.5
                    sl = entry_p - atr * 0.5
                    qty = max(1, int(50000 / entry_p))
                    exit_p, reason = entry_p, 'EOD'
                    for j in range(1, len(today_bars)):
                        eb = today_bars.iloc[j]
                        if eb['Low'] <= sl: exit_p, reason = sl, 'SL'; break
                        if eb['High'] >= tgt: exit_p, reason = tgt, 'TGT'; break
                        exit_p = eb['Close']
                    g = (exit_p - entry_p) * qty  # Long
                    _, _, c = intraday_costs(entry_p, exit_p, qty)
                    n = g - c
                    strategies['GAP_FADE'].append({
                        'date': str(td), 'sym': sym, 'dir': 'LONG', 'regime': regime,
                        'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                        'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2),
                        'gap_pct': round(gap_pct, 2)
                    })

            # ========== VWAP PULLBACK ==========
            if regime != 'NEUTRAL':
                cum_tp_vol, cum_vol = 0.0, 0.0
                for bi in range(len(today_bars)):
                    b = today_bars.iloc[bi]
                    tp = (b['High'] + b['Low'] + b['Close']) / 3
                    vol = max(b['Volume'], 1)
                    cum_tp_vol += tp * vol
                    cum_vol += vol
                vwap = cum_tp_vol / cum_vol if cum_vol > 0 else fb['Close']

                direction = 'LONG' if regime == 'BULLISH' else 'SHORT'
                for i in range(1, min(4, len(today_bars))):
                    bar = today_bars.iloc[i]
                    if direction == 'LONG' and bar['Low'] <= vwap * 1.002 and bar['Close'] > vwap:
                        entry_p = bar['Close']
                        risk = atr * 0.5
                        sl = entry_p - risk
                        tgt = entry_p + risk * 2.0
                        qty = max(1, int(50000 / entry_p))
                        exit_p, reason = entry_p, 'EOD'
                        for j in range(i + 1, len(today_bars)):
                            eb = today_bars.iloc[j]
                            if eb['Low'] <= sl: exit_p, reason = sl, 'SL'; break
                            if eb['High'] >= tgt: exit_p, reason = tgt, 'TGT'; break
                            exit_p = eb['Close']
                        g, n, c = intraday_costs(entry_p, exit_p, qty)
                        strategies['VWAP'].append({
                            'date': str(td), 'sym': sym, 'dir': direction, 'regime': regime,
                            'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                            'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2)
                        })
                        break
                    elif direction == 'SHORT' and bar['High'] >= vwap * 0.998 and bar['Close'] < vwap:
                        entry_p = bar['Close']
                        risk = atr * 0.5
                        sl = entry_p + risk
                        tgt = entry_p - risk * 2.0
                        qty = max(1, int(50000 / entry_p))
                        exit_p, reason = entry_p, 'EOD'
                        for j in range(i + 1, len(today_bars)):
                            eb = today_bars.iloc[j]
                            if eb['High'] >= sl: exit_p, reason = sl, 'SL'; break
                            if eb['Low'] <= tgt: exit_p, reason = tgt, 'TGT'; break
                            exit_p = eb['Close']
                        g = (entry_p - exit_p) * qty
                        _, _, c = intraday_costs(min(entry_p, exit_p), max(entry_p, exit_p), qty)
                        n = g - c
                        strategies['VWAP'].append({
                            'date': str(td), 'sym': sym, 'dir': direction, 'regime': regime,
                            'entry': round(entry_p, 2), 'exit': round(exit_p, 2), 'reason': reason,
                            'qty': qty, 'gross': round(g, 2), 'net': round(n, 2), 'costs': round(c, 2)
                        })
                        break

    # ========== RESULTS ==========
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 120)
    print("V16.1 STRATEGY SHOOTOUT RESULTS")
    print("=" * 120)
    print(f"{'Strategy':<15} | {'Trades':<7} | {'Wins':<5} | {'Win %':<7} | {'PF':<7} | {'Gross P&L':<12} | {'Costs':<10} | {'Net P&L':<12} | {'Avg Net/Trade':<14}")
    print("-" * 120)

    report = {}
    for strat_name, trades in strategies.items():
        df_t = pd.DataFrame(trades)
        if df_t.empty:
            print(f"{strat_name:<15} | {'0':<7} | — No trades generated —")
            report[strat_name] = {'trades': 0}
            continue

        df_t.to_csv(os.path.join(results_dir, f"v16_{strat_name.lower()}_trades.csv"), index=False)

        n = len(df_t)
        wins = (df_t['net'] > 0).sum()
        win_r = wins / n * 100
        gw = df_t[df_t['net'] > 0]['net'].sum()
        gl = abs(df_t[df_t['net'] <= 0]['net'].sum())
        pf = round(gw / gl, 3) if gl > 0 else 99.0
        tot_g = df_t['gross'].sum()
        tot_c = df_t['costs'].sum()
        tot_n = df_t['net'].sum()
        avg_n = tot_n / n

        print(f"{strat_name:<15} | {n:<7} | {wins:<5} | {win_r:>5.1f}% | {pf:>7.3f} | Rs {tot_g:>9,.0f} | Rs {tot_c:>7,.0f} | Rs {tot_n:>9,.0f} | Rs {avg_n:>10,.0f}")

        # Exit reason breakdown
        for reason in ['TGT', 'SL', 'EOD']:
            sub = df_t[df_t['reason'] == reason]
            if not sub.empty:
                print(f"  -- {reason:<10}: {len(sub):>4} trades | Avg Net: Rs {sub['net'].mean():>8,.0f}")

        # Regime breakdown
        for reg in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            sub = df_t[df_t['regime'] == reg]
            if not sub.empty:
                rw = (sub['net'] > 0).sum()
                rpf = round(sub[sub['net'] > 0]['net'].sum() / max(1, abs(sub[sub['net'] <= 0]['net'].sum())), 2)
                print(f"  -- {reg:<10}: {len(sub):>4} trades | Win: {rw/len(sub)*100:>5.1f}% | PF: {rpf:>5.2f} | Net: Rs {sub['net'].sum():>9,.0f}")

        report[strat_name] = {
            'trades': n, 'wins': int(wins), 'win_rate': round(win_r, 1),
            'profit_factor': pf, 'gross_pnl': round(tot_g, 2),
            'costs': round(tot_c, 2), 'net_pnl': round(tot_n, 2)
        }

    with open(os.path.join(results_dir, "v16_1_validation_report.json"), "w") as f:
        json.dump({'generated_at': datetime.now().isoformat(), 'results': report}, f, indent=2)

    print(f"\n[REPORT SAVED] -> results/v16_1_validation_report.json")
    print("=" * 120)


if __name__ == "__main__":
    run_fast_backtest()
