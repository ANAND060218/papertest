"""
V3 -- Backtester: Brutally Honest
Calculates real P&L for labeled trades including ALL Indian intraday costs.

No ML. No AI. No RL. No LLM.
Just: setup + target + stop + real costs = does this make money?

Cost Model (Indian Equity Intraday):
  - Brokerage:           0.03% per side
  - STT:                 0.025% on sell side
  - Stamp Duty:          0.003% on buy side
  - Exchange Txn:        0.00345% per side
  - SEBI Turnover:       0.0001% per side
  - GST:                 18% on (brokerage + exchange txn + SEBI fee)
  - Slippage:            0.05% adverse on entry and exit
  - Spread:              0.02% (included in slippage for simplicity)
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CostModel:
    """
    Calculates exact round-trip trading costs for Indian equity intraday.
    """

    def __init__(self):
        self.brokerage_pct = config.BROKERAGE_PCT
        self.stt_sell_pct = config.STT_SELL_PCT
        self.stamp_duty_buy_pct = config.STAMP_DUTY_BUY_PCT
        self.exchange_txn_pct = config.EXCHANGE_TXN_PCT
        self.sebi_pct = config.SEBI_TURNOVER_PCT
        self.gst_pct = config.GST_ON_BROKERAGE_PCT
        self.slippage_pct = config.SLIPPAGE_PCT
        self.spread_pct = config.SPREAD_PCT

    def calculate_round_trip_cost(self, entry_price, exit_price, quantity):
        """
        Calculate total round-trip cost for a trade.

        Returns dict:
        {
            'buy_value': float,
            'sell_value': float,
            'brokerage': float,
            'stt': float,
            'stamp_duty': float,
            'exchange_txn': float,
            'sebi_fee': float,
            'gst': float,
            'slippage_cost': float,
            'total_cost': float,
            'cost_pct': float,  # total cost as % of entry value
        }
        """
        buy_value = entry_price * quantity
        sell_value = exit_price * quantity

        # Brokerage (both sides)
        brokerage = (buy_value + sell_value) * self.brokerage_pct

        # STT (sell side only for intraday)
        stt = sell_value * self.stt_sell_pct

        # Stamp duty (buy side only)
        stamp_duty = buy_value * self.stamp_duty_buy_pct

        # Exchange transaction charges (both sides)
        exchange_txn = (buy_value + sell_value) * self.exchange_txn_pct

        # SEBI turnover fee (both sides)
        sebi_fee = (buy_value + sell_value) * self.sebi_pct

        # GST = 18% on (brokerage + exchange + SEBI)
        gst = (brokerage + exchange_txn + sebi_fee) * self.gst_pct

        # Slippage: adverse price movement on BOTH entry and exit
        slippage_cost = (buy_value + sell_value) * self.slippage_pct

        total_cost = brokerage + stt + stamp_duty + exchange_txn + sebi_fee + gst + slippage_cost

        return {
            'buy_value': round(buy_value, 2),
            'sell_value': round(sell_value, 2),
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'stamp_duty': round(stamp_duty, 2),
            'exchange_txn': round(exchange_txn, 2),
            'sebi_fee': round(sebi_fee, 2),
            'gst': round(gst, 2),
            'slippage_cost': round(slippage_cost, 2),
            'total_cost': round(total_cost, 2),
            'cost_pct': round(total_cost / buy_value * 100, 4) if buy_value > 0 else 0,
        }


class Backtester:
    """
    V3 Backtester: Applies real costs to labeled trades and produces
    an honest performance report.
    """

    def __init__(self, initial_capital=None):
        self.capital = initial_capital or config.INITIAL_CAPITAL
        self.cost_model = CostModel()

    def run(self, labeled_trades):
        """
        Run backtest on labeled trades with full cost model.

        Args:
            labeled_trades: List of dicts from TradeLabeler

        Returns:
            results_df: DataFrame with per-trade P&L after costs
            performance: dict of performance metrics
        """
        if not labeled_trades:
            print("[V3] No trades to backtest.")
            return None, None

        results = []
        equity = self.capital
        peak_equity = equity
        max_drawdown = 0
        equity_curve = [equity]

        for trade in labeled_trades:
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # Position sizing: risk 1% of current equity, or max position % from config
            max_position_value = equity * config.MAX_POSITION_PCT
            quantity = int(max_position_value / entry_price)
            if quantity <= 0:
                quantity = 1  # Minimum 1 share

            # Calculate costs
            costs = self.cost_model.calculate_round_trip_cost(entry_price, exit_price, quantity)

            # Gross P&L
            gross_pnl = (exit_price - entry_price) * quantity

            # Net P&L
            net_pnl = gross_pnl - costs['total_cost']

            # Adjust entry/exit for slippage (for accurate reporting)
            slipped_entry = entry_price * (1 + config.SLIPPAGE_PCT)
            slipped_exit = exit_price * (1 - config.SLIPPAGE_PCT) if net_pnl >= 0 else exit_price * (1 + config.SLIPPAGE_PCT)

            # Update equity
            equity += net_pnl
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)

            equity_curve.append(equity)

            results.append({
                'timestamp': trade['timestamp'],
                'trade_date': trade['trade_date'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'result': trade['result'],
                'bars_held': trade['bars_held'],
                'quantity': quantity,
                'gross_pnl': round(gross_pnl, 2),
                'total_cost': costs['total_cost'],
                'cost_pct': costs['cost_pct'],
                'net_pnl': round(net_pnl, 2),
                'equity': round(equity, 2),
                'drawdown_pct': round(drawdown, 2),
                'pnl_pct_gross': trade['pnl_pct'],
                'pnl_pct_net': round(net_pnl / (entry_price * quantity) * 100, 4),
            })

        results_df = pd.DataFrame(results)
        performance = self._calculate_performance(results_df, equity_curve, max_drawdown)

        return results_df, performance

    def _calculate_performance(self, df, equity_curve, max_drawdown):
        """Calculate comprehensive performance metrics."""
        total_trades = len(df)
        if total_trades == 0:
            return {}

        wins = df[df['net_pnl'] > 0]
        losses = df[df['net_pnl'] <= 0]

        # Win rate
        win_rate = len(wins) / total_trades * 100

        # Average win/loss
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy (avg P&L per trade)
        expectancy = df['net_pnl'].mean()

        # Expectancy ratio (expectancy / avg loss)
        expectancy_ratio = expectancy / avg_loss if avg_loss > 0 else float('inf')

        # Total P&L
        total_pnl = df['net_pnl'].sum()
        total_costs = df['total_cost'].sum()

        # Return %
        total_return_pct = (total_pnl / self.capital) * 100

        # Sharpe (simplified: using trade returns, annualized assuming 250 trading days)
        trade_returns = df['pnl_pct_net'].values
        if len(trade_returns) > 1 and trade_returns.std() > 0:
            # Rough annualization: assume avg ~2 trades/day, 250 days/year
            trades_per_year = min(total_trades / max(len(df['trade_date'].unique()), 1), 5) * 250
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        # Outcome breakdown
        targets = (df['result'] == 'TARGET').sum()
        stops = (df['result'] == 'STOP').sum()
        timeouts = (df['result'] == 'TIMEOUT').sum()

        return {
            'total_trades': total_trades,
            'targets': targets,
            'stops': stops,
            'timeouts': timeouts,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_loss_ratio': round(avg_win / avg_loss, 2) if avg_loss > 0 else float('inf'),
            'profit_factor': round(profit_factor, 4),
            'expectancy': round(expectancy, 2),
            'expectancy_ratio': round(expectancy_ratio, 4),
            'total_pnl_gross': round(df['gross_pnl'].sum(), 2),
            'total_costs': round(total_costs, 2),
            'total_pnl_net': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'initial_capital': self.capital,
            'final_equity': round(self.capital + total_pnl, 2),
        }

    def print_report(self, results_df, performance):
        """Print the V3 brutally honest backtest report."""
        if results_df is None or performance is None:
            print("[V3] No results to report.")
            return

        p = performance

        print(f"\n{'=' * 90}")
        print("V3 BACKTEST REPORT -- BRUTALLY HONEST (WITH ALL COSTS)")
        print(f"{'=' * 90}")

        print(f"\n--- SETUP ---")
        print(f"  Strategy:         Previous-Day High Breakout")
        print(f"  Symbol:           {config.PRIMARY_SYMBOL}")
        print(f"  Initial Capital:  Rs. {p['initial_capital']:,.2f}")

        print(f"\n--- TRADE STATISTICS ---")
        print(f"  Total Trades:     {p['total_trades']}")
        print(f"    TARGET (win):   {p['targets']}")
        print(f"    STOP (loss):    {p['stops']}")
        print(f"    TIMEOUT:        {p['timeouts']}")

        print(f"\n--- PROFITABILITY ---")
        print(f"  Win Rate:           {p['win_rate']:.1f}%")
        print(f"  Avg Win:            Rs. {p['avg_win']:.2f}")
        print(f"  Avg Loss:           Rs. {p['avg_loss']:.2f}")
        print(f"  Win/Loss Ratio:     {p['avg_win_loss_ratio']:.2f}")
        print(f"  Profit Factor:      {p['profit_factor']:.4f}")
        print(f"  Expectancy/Trade:   Rs. {p['expectancy']:.2f}")

        print(f"\n--- P&L ---")
        print(f"  Gross P&L:          Rs. {p['total_pnl_gross']:.2f}")
        print(f"  Total Costs:        Rs. {p['total_costs']:.2f}")
        print(f"  NET P&L:            Rs. {p['total_pnl_net']:.2f}")
        print(f"  Return:             {p['total_return_pct']:.2f}%")
        print(f"  Final Equity:       Rs. {p['final_equity']:.2f}")

        print(f"\n--- RISK ---")
        print(f"  Max Drawdown:       {p['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:       {p['sharpe_ratio']:.2f}")

        # V3 DECISION GATE
        print(f"\n{'=' * 90}")
        print("V3 DECISION GATE")
        print(f"{'=' * 90}")

        if p['profit_factor'] >= 1.15 and p['expectancy'] > 0:
            print("  >>> RESULT: POSITIVE EDGE DETECTED <<<")
            print(f"  Profit Factor {p['profit_factor']:.2f} >= 1.15 AND Expectancy Rs.{p['expectancy']:.2f} > 0")
            print("  --> Proceed to V4 (XGBoost ML filter)")
        elif p['profit_factor'] >= 1.0 and p['expectancy'] > 0:
            print("  >>> RESULT: MARGINAL EDGE <<<")
            print(f"  Profit Factor {p['profit_factor']:.2f} >= 1.0 but < 1.15")
            print("  --> Edge exists but weak. Consider V4 to see if ML improves it.")
            print("  --> Also consider testing alternative setups.")
        else:
            print("  >>> RESULT: NO EDGE -- DO NOT BUILD ML <<<")
            print(f"  Profit Factor {p['profit_factor']:.2f} < 1.0 OR Expectancy Rs.{p['expectancy']:.2f} <= 0")
            print("  --> This setup loses money after costs.")
            print("  --> Change the setup BEFORE adding XGBoost.")
            print("  --> Test: ORB, VWAP breakout, Opening momentum, Support bounce")

        # Per-trade detail
        print(f"\n{'=' * 90}")
        print("PER-TRADE DETAIL")
        print(f"{'=' * 90}")
        print(f"{'Timestamp':>22s} {'Entry':>8s} {'Exit':>8s} {'Result':>8s} "
              f"{'Qty':>5s} {'Gross':>9s} {'Costs':>8s} {'Net':>9s} {'Equity':>10s} {'DD%':>6s}")
        for _, r in results_df.iterrows():
            print(
                f"{str(r['timestamp']):>22s} "
                f"{r['entry_price']:>8.2f} "
                f"{r['exit_price']:>8.2f} "
                f"{r['result']:>8s} "
                f"{r['quantity']:>5d} "
                f"{r['gross_pnl']:>+9.2f} "
                f"{r['total_cost']:>8.2f} "
                f"{r['net_pnl']:>+9.2f} "
                f"{r['equity']:>10.2f} "
                f"{r['drawdown_pct']:>6.2f}"
            )


if __name__ == "__main__":
    # Full V2 -> V3 pipeline
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
    from data_manager import DataManager
    from setup_detector import SetupDetector
    from labeler import TradeLabeler

    # V1: Load data
    dm = DataManager()
    intraday_df = dm.load_intraday(config.PRIMARY_SYMBOL)
    if intraday_df is None:
        print("[ERROR] No intraday data. Run core/data_manager.py first.")
        sys.exit(1)

    daily_ctx = dm.build_daily_context(intraday_df)

    # V2: Detect setups
    detector = SetupDetector()
    setups = detector.detect_setups(intraday_df, daily_ctx)
    detector.summarize_setups(setups)

    # V2: Label outcomes
    labeler = TradeLabeler()
    labeled = labeler.label_setups(setups, intraday_df)
    labeler.summarize_labels(labeled)

    # V3: Backtest with costs
    bt = Backtester()
    results_df, performance = bt.run(labeled)
    bt.print_report(results_df, performance)

    # Save results
    if results_df is not None:
        results_path = os.path.join(config.RESULTS_DIR, "v3_backtest_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n[V3] Results saved to {results_path}")
