"""
V15.2 Block Bootstrap Robustness Engine
Performs Circular Block Bootstrap (Block Length = 6 to 12 Months) on monthly portfolio returns
to preserve volatility clustering, momentum regimes, and autocorrelated drawdown paths.
"""
import pandas as pd
import numpy as np


class BlockBootstrapEngine:
    """
    Circular Block Bootstrap Simulator for Time-Series Returns.
    """

    @staticmethod
    def run_block_bootstrap(monthly_returns, initial_capital=100000.0, block_size=6, n_simulations=2000, holding_period_years=9.0):
        """
        Resamples contiguous multi-month blocks to simulate 2,000 realistic return paths.
        """
        returns = np.array(monthly_returns.dropna())
        n_obs = len(returns)
        if n_obs < 12:
            return {}

        cagrs = []
        max_dds = []

        n_blocks_needed = int(np.ceil(n_obs / block_size))

        for _ in range(n_simulations):
            # Select random block start indices with circular wrap-around
            start_indices = np.random.randint(0, n_obs, size=n_blocks_needed)
            sampled_returns = []

            for start in start_indices:
                for offset in range(block_size):
                    idx = (start + offset) % n_obs
                    sampled_returns.append(returns[idx])
                    if len(sampled_returns) >= n_obs:
                        break
                if len(sampled_returns) >= n_obs:
                    break

            sampled_returns = np.array(sampled_returns[:n_obs])

            # Compound returns
            growth_curve = initial_capital * np.cumprod(1 + sampled_returns)
            # Ensure no negative equity
            growth_curve = np.maximum(growth_curve, 1000.0)

            end_eq = growth_curve[-1]
            cagr = (((end_eq / initial_capital) ** (1 / holding_period_years)) - 1) * 100
            cagrs.append(cagr)

            peak = np.maximum.accumulate(growth_curve)
            dd = (growth_curve - peak) / peak * 100
            max_dds.append(dd.min())

        cagrs = np.array(cagrs)
        max_dds = np.array(max_dds)

        prob_dd_30 = (max_dds <= -30.0).mean() * 100
        prob_dd_40 = (max_dds <= -40.0).mean() * 100
        prob_dd_50 = (max_dds <= -50.0).mean() * 100

        return {
            'cagr_5th_pct': round(float(np.percentile(cagrs, 5)), 2),
            'cagr_50th_pct (Median)': round(float(np.percentile(cagrs, 50)), 2),
            'cagr_95th_pct': round(float(np.percentile(cagrs, 95)), 2),
            'max_dd_5th_pct': round(float(np.percentile(max_dds, 95)), 2), # Best case dd
            'max_dd_50th_pct (Median)': round(float(np.percentile(max_dds, 50)), 2),
            'max_dd_95th_pct': round(float(np.percentile(max_dds, 5)), 2), # Worst case dd
            'prob_dd_ge_30pct': round(float(prob_dd_30), 1),
            'prob_dd_ge_40pct': round(float(prob_dd_40), 1),
            'prob_dd_ge_50pct': round(float(prob_dd_50), 1)
        }
