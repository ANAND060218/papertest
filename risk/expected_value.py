"""
V5 -- Expected Value Calculator
Don't trade on probability alone. Calculate Expected Value:

  EV = P(win) x avg_win - P(loss) x avg_loss - costs

Trade only when EV > threshold.
Better than probability alone because it accounts for asymmetric payoffs.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backtest.backtester import CostModel


class ExpectedValueCalculator:
    """
    Calculates Expected Value for each potential trade.
    Decides whether a trade is economically attractive.
    """

    def __init__(self, avg_win_pct=None, avg_loss_pct=None, cost_pct=None):
        """
        Args:
            avg_win_pct: Historical average win % (from V3 backtest)
            avg_loss_pct: Historical average loss % (from V3 backtest)
            cost_pct: Round-trip cost as % of trade value
        """
        self.avg_win_pct = avg_win_pct or 1.0   # Default to target %
        self.avg_loss_pct = avg_loss_pct or 0.5  # Default to stop %
        self.cost_model = CostModel()

        # Estimate cost % from the cost model
        # Using a representative price and quantity
        if cost_pct is None:
            sample_cost = self.cost_model.calculate_round_trip_cost(
                entry_price=1300, exit_price=1300, quantity=7
            )
            self.cost_pct = sample_cost['cost_pct']
        else:
            self.cost_pct = cost_pct

    def calculate_ev(self, p_win, avg_win_pct=None, avg_loss_pct=None):
        """
        Calculate Expected Value for a single trade.

        Args:
            p_win: Probability of winning (from XGBoost)
            avg_win_pct: Average win % (override default)
            avg_loss_pct: Average loss % (override default)

        Returns:
            dict with EV details
        """
        p_loss = 1 - p_win
        win_pct = avg_win_pct or self.avg_win_pct
        loss_pct = avg_loss_pct or self.avg_loss_pct

        ev_pct = (p_win * win_pct) - (p_loss * loss_pct) - self.cost_pct
        ev_rs = ev_pct / 100 * 1300 * 7  # Approximate Rs value for RELIANCE

        return {
            'p_win': round(p_win, 4),
            'p_loss': round(p_loss, 4),
            'avg_win_pct': win_pct,
            'avg_loss_pct': loss_pct,
            'cost_pct': round(self.cost_pct, 4),
            'ev_pct': round(ev_pct, 4),
            'ev_rupees_approx': round(ev_rs, 2),
            'decision': 'TRADE' if ev_pct > 0 else 'SKIP',
        }

    def filter_by_ev(self, trades_df, probabilities, min_ev_pct=0.0):
        """
        Filter trades by Expected Value.

        Args:
            trades_df: DataFrame of trades
            probabilities: P(win) array from XGBoost
            min_ev_pct: Minimum EV % to take trade

        Returns:
            filtered_df: Only trades with EV > min_ev_pct
            ev_details: List of EV calculations
        """
        ev_details = []
        mask = []

        for i, p_win in enumerate(probabilities):
            ev = self.calculate_ev(p_win)
            ev_details.append(ev)
            mask.append(ev['ev_pct'] > min_ev_pct)

        filtered_df = trades_df[mask].copy()

        return filtered_df, ev_details

    def print_ev_analysis(self, probabilities, labels=None, pnl_pcts=None):
        """
        Print comprehensive EV analysis for a set of trades.
        """
        print(f"\n{'=' * 90}")
        print("V5 -- EXPECTED VALUE ANALYSIS")
        print(f"{'=' * 90}")
        print(f"  Cost per trade: {self.cost_pct:.4f}%")
        print(f"  Avg win target: {self.avg_win_pct:.2f}%")
        print(f"  Avg loss target: {self.avg_loss_pct:.2f}%")

        # Test different P(win) scenarios
        print(f"\n  EV at different P(win) levels:")
        print(f"  {'P(win)':>8s} {'P(loss)':>8s} {'EV%':>10s} {'Decision':>10s}")
        for p in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            ev = self.calculate_ev(p)
            print(f"  {p:>8.2f} {1-p:>8.2f} {ev['ev_pct']:>+10.4f} {ev['decision']:>10s}")

        # Breakeven probability
        # EV = 0 when P * win - (1-P) * loss - cost = 0
        # P * win - loss + P * loss = cost
        # P * (win + loss) = cost + loss
        breakeven_p = (self.avg_loss_pct + self.cost_pct) / (self.avg_win_pct + self.avg_loss_pct)
        print(f"\n  Breakeven P(win): {breakeven_p:.4f} ({breakeven_p*100:.1f}%)")
        print(f"  -> Only trade when XGBoost P(win) > {breakeven_p:.2f}")

        # If actual data provided, compare EV vs probability filtering
        if probabilities is not None and pnl_pcts is not None:
            print(f"\n  COMPARISON: Probability Filter vs EV Filter")
            print(f"  {'Filter':>15s} {'Trades':>8s} {'Wins':>6s} {'TotalPnL%':>12s} {'AvgPnL%':>10s}")

            # Baseline
            total = len(probabilities)
            wins = labels.sum() if labels is not None else 0
            print(f"  {'BASELINE':>15s} {total:>8d} {int(wins):>6d} "
                  f"{pnl_pcts.sum():>+12.4f} {pnl_pcts.mean():>+10.4f}")

            # P(win) > 0.55
            mask_p = probabilities >= 0.55
            print(f"  {'P>0.55':>15s} {mask_p.sum():>8d} "
                  f"{int(labels[mask_p].sum()) if labels is not None else 'N/A':>6} "
                  f"{pnl_pcts[mask_p].sum():>+12.4f} "
                  f"{pnl_pcts[mask_p].mean() if mask_p.sum() > 0 else 0:>+10.4f}")

            # EV > 0
            ev_mask = np.array([self.calculate_ev(p)['ev_pct'] > 0 for p in probabilities])
            print(f"  {'EV>0':>15s} {ev_mask.sum():>8d} "
                  f"{int(labels[ev_mask].sum()) if labels is not None else 'N/A':>6} "
                  f"{pnl_pcts[ev_mask].sum():>+12.4f} "
                  f"{pnl_pcts[ev_mask].mean() if ev_mask.sum() > 0 else 0:>+10.4f}")

        return breakeven_p
