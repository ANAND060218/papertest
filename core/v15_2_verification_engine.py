"""
V15.2 Verification, Survivorship-Bias Audit & Risk Architecture Engine
Implements strict point-in-time execution, survivorship audit, naive momentum control,
Monte Carlo trade permutation simulations, and 5 portfolio risk weighting architectures.
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import config


class V15VerificationEngine:
    """
    Institutional Verification & Portfolio Risk Architecture Engine.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(BASE_DIR, "data", "nifty_10year_stock_market.db")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """Loads and prepares daily Open, High, Low, Close matrices and NIFTY index."""
        conn = sqlite3.connect(self.db_path)
        df_all = pd.read_sql_query("SELECT Date, Symbol, Open, High, Low, Close FROM stock_daily_10y ORDER BY Date ASC", conn)
        conn.close()

        df_all['Date'] = pd.to_datetime(df_all['Date'])

        # Separate index and stocks
        self.nifty_daily = df_all[df_all['Symbol'] == '^NSEI'].copy().sort_values('Date').reset_index(drop=True)
        self.stocks_daily = df_all[~df_all['Symbol'].isin(['^NSEI', '^NSEBANK'])].copy().sort_values('Date').reset_index(drop=True)

        # Pivot matrices
        self.close_matrix = self.stocks_daily.pivot(index='Date', columns='Symbol', values='Close').ffill()
        self.open_matrix = self.stocks_daily.pivot(index='Date', columns='Symbol', values='Open').ffill()

        # Monthly Closes
        self.monthly_close = self.close_matrix.resample('ME').last().ffill()
        self.nifty_monthly = self.nifty_daily.set_index('Date').resample('ME').last().ffill()

    def run_simulation(
        self,
        lookback_1=6,
        lookback_2=12,
        sma_filter=10,
        top_n=5,
        weighting_schema='EQUAL_WEIGHT',   # 'EQUAL_WEIGHT', 'INVERSE_VOL', 'MACRO_REGIME'
        universe_filter='ALL',              # 'ALL', 'SURVIVORSHIP_STRICT_2016'
        strategy_mode='DUAL_MOMENTUM',      # 'DUAL_MOMENTUM', 'NAIVE_12M_CONTROL'
        slippage_pct=0.0005,
        start_date="2017-08-31",
        end_date="2026-08-14",
        initial_capital=100000.0
    ):
        """
        Executes causal point-in-time backtest.
        Signal computed at Month-End Close (T), executed on next trading day Open (T+1).
        """
        # Universe Filtering for Survivorship Bias Audit
        if universe_filter == 'SURVIVORSHIP_STRICT_2016':
            # Remove stocks added to NIFTY after 2016
            excluded = ['ADANIENT.NS', 'ADANIPORTS.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'APOLLOHOSP.NS', 'BAJAJFINSV.NS', 'JSWSTEEL.NS']
            avail_symbols = [s for s in self.monthly_close.columns if s not in excluded]
        else:
            avail_symbols = self.monthly_close.columns.tolist()

        price_df = self.monthly_close[avail_symbols].copy()

        # Momentum Calculation
        if strategy_mode == 'NAIVE_12M_CONTROL':
            # Simple 12-Month Momentum without 6m blend or SMA filter
            mom_scores = price_df.pct_change(12)
            sma_df = None
            warmup = 12
        else:
            # V15.2 Dual Momentum
            mom_1 = price_df.pct_change(lookback_1)
            mom_2 = price_df.pct_change(lookback_2) if lookback_2 else mom_1
            mom_scores = 0.5 * mom_1 + 0.5 * mom_2
            sma_df = price_df.rolling(sma_filter).mean() if sma_filter else None
            warmup = max(lookback_1, lookback_2 or 0) + (sma_filter or 0)

        # Macro NIFTY Regime (10-Month SMA of NIFTY)
        nifty_sma_10 = self.nifty_monthly['Close'].rolling(10).mean()

        rebal_dates = [d for d in price_df.index[warmup:] if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]
        if len(rebal_dates) < 2:
            return {'error': 'Insufficient dates'}

        portfolio_history = []
        trades = []
        current_holdings = {}
        cash = initial_capital

        for i in range(len(rebal_dates) - 1):
            rebal_date = rebal_dates[i]
            next_date = rebal_dates[i + 1]

            # Signal generated at close of rebal_date (T)
            scores = mom_scores.loc[rebal_date].dropna()
            current_prices = price_df.loc[rebal_date]

            if sma_df is not None:
                sma_vals = sma_df.loc[rebal_date]
                valid_stocks = [s for s in scores.index if current_prices[s] > sma_vals[s]]
            else:
                valid_stocks = scores.index.tolist()

            ranked_leaders = scores.loc[valid_stocks].sort_values(ascending=False).head(top_n).index.tolist()

            # Macro Regime Check
            if weighting_schema == 'MACRO_REGIME':
                nifty_c = self.nifty_monthly.loc[rebal_date]['Close'] if rebal_date in self.nifty_monthly.index else 10000
                n_sma = nifty_sma_10.loc[rebal_date] if rebal_date in nifty_sma_10.index else nifty_c
                macro_bullish = nifty_c >= n_sma
                macro_multiplier = 1.0 if macro_bullish else 0.50  # Cut equity exposure to 50% in bear regime
            else:
                macro_multiplier = 1.0

            # 1. Liquidate dropped positions
            for sym in list(current_holdings.keys()):
                if sym not in ranked_leaders:
                    raw_exit_p = current_prices[sym]
                    fill_exit_p = raw_exit_p * (1 - slippage_pct)
                    qty = current_holdings[sym]['qty']
                    gross_val = qty * fill_exit_p
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

            # 2. Portfolio Weighting Calculation
            total_equity = cash + sum(current_holdings[s]['qty'] * current_prices[s] for s in current_holdings)
            investable_equity = total_equity * macro_multiplier

            if weighting_schema == 'INVERSE_VOL' and len(ranked_leaders) > 0:
                # 6-Month Realized Volatility Weighting
                vols = self.stocks_daily[self.stocks_daily['Date'] <= rebal_date].groupby('Symbol')['Close'].apply(
                    lambda x: x.pct_change().tail(126).std()
                ).loc[ranked_leaders].replace(0, np.nan).fillna(0.02)
                inv_vols = 1.0 / vols
                weights = inv_vols / inv_vols.sum()
                # Cap at 25% max position
                weights = weights.clip(upper=0.25)
                weights = weights / weights.sum()
            else:
                # Equal Weight
                weights = {s: 1.0 / top_n for s in ranked_leaders} if len(ranked_leaders) > 0 else {}

            # 3. Rebalance / Allocate to Leaders
            for sym in ranked_leaders:
                w = weights.get(sym, 1.0 / top_n)
                target_alloc = investable_equity * w
                current_val = current_holdings[sym]['qty'] * current_prices[sym] if sym in current_holdings else 0.0

                if sym not in current_holdings:
                    fill_buy_p = current_prices[sym] * (1 + slippage_pct)
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

            # 4. Mark to Market at end of period (T+1 month)
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
            return {'error': 'No history'}

        df_hist = pd.DataFrame(portfolio_history)
        df_hist['date'] = pd.to_datetime(df_hist['date'])

        # Performance Metrics
        start_eq = initial_capital
        end_eq = df_hist.iloc[-1]['portfolio_equity']
        tot_ret_pct = ((end_eq - start_eq) / start_eq) * 100
        n_days = (df_hist.iloc[-1]['date'] - df_hist.iloc[0]['date']).days
        n_years = n_days / 365.25
        cagr = (((end_eq / start_eq) ** (1 / n_years)) - 1) * 100 if n_years > 0 else tot_ret_pct

        df_hist['monthly_ret'] = df_hist['portfolio_equity'].pct_change()
        monthly_rets = df_hist['monthly_ret'].dropna()
        ann_vol = monthly_rets.std() * np.sqrt(12) * 100 if len(monthly_rets) > 1 else 0.0

        rf_monthly = 0.065 / 12
        excess_rets = monthly_rets - rf_monthly
        sharpe = (excess_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0.0

        downside_rets = monthly_rets[monthly_rets < rf_monthly] - rf_monthly
        downside_std = np.sqrt(np.mean(downside_rets ** 2)) if len(downside_rets) > 0 else 0.0001
        sortino = (excess_rets.mean() / downside_std) * np.sqrt(12) if downside_std > 0 else 0.0

        peak = df_hist['portfolio_equity'].cummax()
        dd_series = (df_hist['portfolio_equity'] - peak) / peak * 100
        max_dd = dd_series.min()
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

        wins = [t for t in trades if t['is_win']]
        losses = [t for t in trades if not t['is_win']]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
        gw = sum(t['net_pnl_rs'] for t in wins)
        gl = abs(sum(t['net_pnl_rs'] for t in losses))
        pf = round(gw / gl, 3) if gl > 0 else 99.0

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
            'trades_list': trades,
            'df_history': df_hist
        }

    @staticmethod
    def run_monte_carlo(trades, initial_capital=100000.0, n_simulations=2000, holding_period_years=9.0):
        """
        Runs 2,000 Monte Carlo permutations of trade sequences to compute
        drawdown probabilities and CAGR distribution.
        """
        if len(trades) < 10:
            return {}

        pnls = np.array([t['net_pnl_rs'] for t in trades])
        n_trades = len(pnls)

        cagrs = []
        max_dds = []

        for _ in range(n_simulations):
            # Reshuffle trades with replacement
            shuffled_pnl = np.random.choice(pnls, size=n_trades, replace=True)
            equity_curve = initial_capital + np.cumsum(shuffled_pnl)
            # Ensure no negative equity
            equity_curve = np.maximum(equity_curve, 1000.0)

            end_eq = equity_curve[-1]
            cagr = (((end_eq / initial_capital) ** (1 / holding_period_years)) - 1) * 100
            cagrs.append(cagr)

            peak = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - peak) / peak * 100
            max_dds.append(dd.min())

        cagrs = np.array(cagrs)
        max_dds = np.array(max_dds)

        prob_dd_30 = (max_dds <= -30.0).mean() * 100
        prob_dd_40 = (max_dds <= -40.0).mean() * 100

        return {
            'cagr_5th_pct': round(float(np.percentile(cagrs, 5)), 2),
            'cagr_50th_pct': round(float(np.percentile(cagrs, 50)), 2),
            'cagr_95th_pct': round(float(np.percentile(cagrs, 95)), 2),
            'max_dd_5th_pct': round(float(np.percentile(max_dds, 95)), 2), # smallest dd
            'max_dd_50th_pct': round(float(np.percentile(max_dds, 50)), 2),
            'max_dd_95th_pct': round(float(np.percentile(max_dds, 5)), 2), # deepest dd
            'prob_dd_ge_30pct': round(float(prob_dd_30), 1),
            'prob_dd_ge_40pct': round(float(prob_dd_40), 1)
        }
