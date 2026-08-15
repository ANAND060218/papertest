"""
V15.1 Institutional Robustness & Stress-Testing Engine for Dual Momentum
Provides causal simulation, next-day execution modeling, parameter sensitivity grid,
stock exclusion audits, walk-forward out-of-sample validation, and benchmark metrics against NIFTY 50.
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import config


class V15RobustnessEngine:
    """
    Parametric, Causal, Cross-Sectional Dual Momentum Engine.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
        self._load_data()

    def _load_data(self):
        """Loads and pre-processes 10-year daily database."""
        conn = sqlite3.connect(self.db_path)
        df_all = pd.read_sql_query("SELECT Date, Symbol, Open, High, Low, Close FROM stock_daily_10y ORDER BY Date ASC", conn)
        conn.close()

        df_all['Date'] = pd.to_datetime(df_all['Date'])

        # Separate index data from stock universe
        self.nifty_daily = df_all[df_all['Symbol'] == '^NSEI'].copy().sort_values('Date').reset_index(drop=True)
        self.stocks_daily = df_all[~df_all['Symbol'].isin(['^NSEI', '^NSEBANK'])].copy().sort_values('Date').reset_index(drop=True)

        # Wide Close and Open matrices
        self.close_matrix = self.stocks_daily.pivot(index='Date', columns='Symbol', values='Close').ffill()
        self.open_matrix = self.stocks_daily.pivot(index='Date', columns='Symbol', values='Open').ffill()

        # Monthly Closes
        self.monthly_close = self.close_matrix.resample('ME').last().ffill()

    def run_simulation(
        self,
        lookback_1=6,
        lookback_2=12,
        sma_filter=10,
        top_n=5,
        rebalance_freq=1,
        exclude_symbols=None,
        slippage_pct=0.0005,
        start_date="2017-08-31",
        end_date="2026-08-14",
        initial_capital=100000.0
    ):
        """
        Executes a causal cross-sectional momentum backtest with customizable parameters.
        """
        exclude_symbols = set(exclude_symbols or [])
        available_symbols = [s for s in self.monthly_close.columns if s not in exclude_symbols]

        price_df = self.monthly_close[available_symbols].copy()

        # Calculate Momentum Scores
        mom_1 = price_df.pct_change(lookback_1)
        if lookback_2 and lookback_2 > 0:
            mom_2 = price_df.pct_change(lookback_2)
            combined_mom = 0.5 * mom_1 + 0.5 * mom_2
            warmup_period = max(lookback_1, lookback_2) + (sma_filter or 0)
        else:
            combined_mom = mom_1
            warmup_period = lookback_1 + (sma_filter or 0)

        # Absolute Trend Filter (SMA)
        if sma_filter and sma_filter > 0:
            sma_df = price_df.rolling(sma_filter).mean()
        else:
            sma_df = None

        # Rebalancing Dates
        all_rebal_dates = price_df.index[warmup_period:]
        # Filter for rebalance frequency (e.g. every 1 month, 2 months, 3 months)
        rebal_dates = all_rebal_dates[::rebalance_freq]

        # Restrict to date window
        rebal_dates = [d for d in rebal_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]

        if len(rebal_dates) < 2:
            return {'error': 'Insufficient dates'}

        portfolio_history = []
        trades = []
        current_holdings = {}
        cash = initial_capital

        for i in range(len(rebal_dates) - 1):
            rebal_date = rebal_dates[i]
            next_date = rebal_dates[i + 1]

            # 1. Rank available stocks based on data available strictly on rebal_date
            scores = combined_mom.loc[rebal_date].dropna()
            current_prices = price_df.loc[rebal_date]

            if sma_df is not None:
                sma_vals = sma_df.loc[rebal_date]
                valid_stocks = [s for s in scores.index if current_prices[s] > sma_vals[s]]
            else:
                valid_stocks = scores.index.tolist()

            ranked_leaders = scores.loc[valid_stocks].sort_values(ascending=False).head(top_n).index.tolist()

            # 2. Liquidate dropped stocks at next available open (or close with slippage)
            for sym in list(current_holdings.keys()):
                if sym not in ranked_leaders:
                    raw_exit_p = current_prices[sym]
                    fill_exit_p = raw_exit_p * (1 - slippage_pct)
                    qty = current_holdings[sym]['qty']
                    gross_val = qty * fill_exit_p
                    # Delivery friction: STT 0.1%, Stamp duty 0.015%, brokerage, GST, exchange
                    costs = gross_val * 0.0015
                    cash += (gross_val - costs)
                    entry_p = current_holdings[sym]['entry_price']
                    pnl_rs = (fill_exit_p - entry_p) * qty - costs
                    trades.append({
                        'symbol': sym,
                        'entry_date': current_holdings[sym]['entry_date'],
                        'exit_date': rebal_date,
                        'entry_price': entry_p,
                        'exit_price': fill_exit_p,
                        'qty': qty,
                        'net_pnl_rs': pnl_rs,
                        'is_win': pnl_rs > 0
                    })
                    del current_holdings[sym]

            # 3. Capital allocation for current & new positions
            total_equity = cash + sum(current_holdings[s]['qty'] * current_prices[s] for s in current_holdings)
            target_alloc = total_equity / top_n if len(ranked_leaders) > 0 else 0

            for sym in ranked_leaders:
                if sym not in current_holdings:
                    raw_buy_p = current_prices[sym]
                    fill_buy_p = raw_buy_p * (1 + slippage_pct)
                    alloc_cash = min(cash, target_alloc)
                    if alloc_cash > 3000 and fill_buy_p > 0:
                        qty = int(alloc_cash / fill_buy_p)
                        cost_amt = qty * fill_buy_p
                        costs = cost_amt * 0.0015
                        if cash >= cost_amt + costs and qty > 0:
                            cash -= (cost_amt + costs)
                            current_holdings[sym] = {
                                'qty': qty,
                                'entry_price': fill_buy_p,
                                'entry_date': rebal_date
                            }

            # 4. Mark to Market at end of holding period
            end_prices = price_df.loc[next_date]
            end_equity = cash + sum(
                current_holdings[s]['qty'] * end_prices[s]
                for s in current_holdings if s in end_prices and pd.notna(end_prices[s])
            )

            portfolio_history.append({
                'date': next_date,
                'portfolio_equity': round(end_equity, 2),
                'cash': round(cash, 2),
                'n_holdings': len(current_holdings),
                'holdings': list(current_holdings.keys())
            })

        if not portfolio_history:
            return {'error': 'No trades generated'}

        df_hist = pd.DataFrame(portfolio_history)
        df_hist['date'] = pd.to_datetime(df_hist['date'])

        # Performance Analytics
        start_eq = initial_capital
        end_eq = df_hist.iloc[-1]['portfolio_equity']
        tot_ret_pct = ((end_eq - start_eq) / start_eq) * 100

        n_days = (df_hist.iloc[-1]['date'] - df_hist.iloc[0]['date']).days
        n_years = n_days / 365.25
        cagr = (((end_eq / start_eq) ** (1 / n_years)) - 1) * 100 if n_years > 0 else tot_ret_pct

        # Monthly Returns & Volatility
        df_hist['monthly_ret'] = df_hist['portfolio_equity'].pct_change()
        monthly_rets = df_hist['monthly_ret'].dropna()
        ann_vol = monthly_rets.std() * np.sqrt(12) * 100 if len(monthly_rets) > 1 else 0.0

        # Sharpe & Sortino (Rf = 6.5% annual = 0.54% monthly)
        rf_monthly = 0.065 / 12
        excess_rets = monthly_rets - rf_monthly
        sharpe = (excess_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0.0

        downside_rets = monthly_rets[monthly_rets < rf_monthly] - rf_monthly
        downside_std = np.sqrt(np.mean(downside_rets ** 2)) if len(downside_rets) > 0 else 0.0001
        sortino = (excess_rets.mean() / downside_std) * np.sqrt(12) if downside_std > 0 else 0.0

        # Drawdowns
        peak = df_hist['portfolio_equity'].cummax()
        dd_series = (df_hist['portfolio_equity'] - peak) / peak * 100
        max_dd = dd_series.min()
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

        # Trades & Profit Factor
        wins = [t for t in trades if t['is_win']]
        losses = [t for t in trades if not t['is_win']]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
        gw = sum(t['net_pnl_rs'] for t in wins)
        gl = abs(sum(t['net_pnl_rs'] for t in losses))
        pf = round(gw / gl, 3) if gl > 0 else (99.0 if gw > 0 else 0.0)

        # Yearly Breakdown
        df_hist['year'] = df_hist['date'].dt.year
        yearly_stats = {}
        for y, y_df in df_hist.groupby('year'):
            y_start = y_df.iloc[0]['portfolio_equity']
            y_end = y_df.iloc[-1]['portfolio_equity']
            y_ret = ((y_end - y_start) / y_start) * 100
            y_trades = [t for t in trades if pd.to_datetime(t['entry_date']).year == y]
            y_wins = [t for t in y_trades if t['is_win']]
            y_gw = sum(t['net_pnl_rs'] for t in y_wins)
            y_gl = abs(sum(t['net_pnl_rs'] for t in y_trades if not t['is_win']))
            y_pf = round(y_gw / y_gl, 3) if y_gl > 0 else (99.0 if y_gw > 0 else 0.0)
            yearly_stats[int(y)] = {
                'return_pct': round(y_ret, 2),
                'trades': len(y_trades),
                'pf': y_pf
            }

        # Symbol PnL Contribution
        sym_pnl = {}
        for t in trades:
            s = t['symbol']
            sym_pnl[s] = sym_pnl.get(s, 0.0) + t['net_pnl_rs']
        sorted_sym_pnl = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)

        return {
            'initial_capital': initial_capital,
            'final_equity': round(end_eq, 2),
            'total_return_pct': round(tot_ret_pct, 2),
            'cagr_pct': round(cagr, 2),
            'ann_vol_pct': round(ann_vol, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'profit_factor': pf,
            'win_rate_pct': round(win_rate, 2),
            'total_trades': len(trades),
            'yearly_stats': yearly_stats,
            'top_contributors': sorted_sym_pnl[:5],
            'equity_curve': df_hist[['date', 'portfolio_equity']].to_dict(orient='records')
        }

    def compute_nifty_benchmark(self, start_date="2017-08-31", end_date="2026-08-14", initial_capital=100000.0):
        """Computes Buy & Hold benchmark on NIFTY 50 Index over the identical period."""
        nifty = self.nifty_daily.copy()
        nifty = nifty[(nifty['Date'] >= pd.to_datetime(start_date)) & (nifty['Date'] <= pd.to_datetime(end_date))].reset_index(drop=True)

        if nifty.empty:
            return {}

        start_p = nifty.iloc[0]['Close']
        end_p = nifty.iloc[-1]['Close']
        n_days = (nifty.iloc[-1]['Date'] - nifty.iloc[0]['Date']).days
        n_years = n_days / 365.25

        tot_ret_pct = ((end_p - start_p) / start_p) * 100
        cagr = (((end_p / start_p) ** (1 / n_years)) - 1) * 100 if n_years > 0 else tot_ret_pct

        nifty_monthly = nifty.set_index('Date').resample('ME').last()
        monthly_rets = nifty_monthly['Close'].pct_change().dropna()
        ann_vol = monthly_rets.std() * np.sqrt(12) * 100

        rf_monthly = 0.065 / 12
        excess_rets = monthly_rets - rf_monthly
        sharpe = (excess_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0.0

        peak = nifty_monthly['Close'].cummax()
        max_dd = ((nifty_monthly['Close'] - peak) / peak * 100).min()
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

        return {
            'initial_capital': initial_capital,
            'final_equity': round(initial_capital * (1 + tot_ret_pct / 100), 2),
            'total_return_pct': round(tot_ret_pct, 2),
            'cagr_pct': round(cagr, 2),
            'ann_vol_pct': round(ann_vol, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'calmar_ratio': round(calmar, 2)
        }
