"""
V12 -- Dynamic Exit & Profit Capture Engine
Simulates bar-by-bar path-dependent exit policies across 8 distinct architectures:
  - V12_A: Fixed Stop -0.50%, Fixed Target +1.00%
  - V12_B: Fixed Stop -0.50%, Fixed Target +0.50%
  - V12_C: Partial Exit 50% @ +0.40% + Breakeven Stop + Runner @ +1.00%
  - V12_D: MFE-Based Trailing Stop (Activates @ +0.30%, Trails 0.20%)
  - V12_E: Time-Based Exit (Max 4 bars / 20 min if profit < +0.30%)
  - V12_F: Partial Exit 50% @ +0.40% + Trailing Stop on Remaining 50%
  - V12_G: MFE-Adaptive Multi-Stage Exit Policy
  - V12_H: ATR-Calibrated Volatility Dynamic Exit
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class V12ExitEngine:
    """
    Path-dependent Bar-by-Bar Trade Lifecycle & Exit Simulator.
    """

    @staticmethod
    def simulate_trade_exit(trade, future_df, exit_policy='V12_A'):
        """
        Simulates the execution of a trade bar-by-bar according to the chosen exit policy.

        Args:
            trade: Dict with entry_price, direction, bar_index, atr_val, etc.
            future_df: DataFrame of subsequent 5m bars for that day.
            exit_policy: 'V12_A' through 'V12_H'

        Returns:
            Dict containing:
              - net_pnl_pct: Total weighted return percentage across all lots.
              - exit_reason: 'TARGET', 'STOP', 'TRAILING_STOP', 'TIME_LIMIT', 'EOD', 'PARTIAL_RUNNER'
              - bars_held: Number of 5m bars held.
              - r_multiple: Realized return divided by initial risk.
        """
        entry_p = trade['entry_price']
        direction = trade.get('direction', 'LONG')
        atr_val = trade.get('atr_val', entry_p * 0.005)
        initial_risk_pct = 0.005  # 0.50% baseline initial risk

        if future_df.empty:
            return {
                'net_pnl_pct': 0.0,
                'exit_reason': 'EOD',
                'bars_held': 0,
                'r_multiple': 0.0
            }

        # Policy Configurations
        if exit_policy == 'V12_A': # Baseline TP 1.0% / SL -0.5%
            return V12ExitEngine._sim_fixed(future_df, entry_p, direction, sl_pct=0.005, tp_pct=0.010)

        elif exit_policy == 'V12_B': # Calibrated TP 0.5% / SL -0.5%
            return V12ExitEngine._sim_fixed(future_df, entry_p, direction, sl_pct=0.005, tp_pct=0.005)

        elif exit_policy == 'V12_C': # Partial 50% @ +0.40% + BE + Runner @ +1.0%
            return V12ExitEngine._sim_partial_be(future_df, entry_p, direction, partial_tp=0.004, final_tp=0.010, initial_sl=0.005)

        elif exit_policy == 'V12_D': # Trailing Stop (Trigger @ +0.30%, Trail 0.20%)
            return V12ExitEngine._sim_trailing(future_df, entry_p, direction, initial_sl=0.005, trigger_pct=0.003, trail_dist=0.0020)

        elif exit_policy == 'V12_E': # Time-Based Momentum Exit (Max 4 bars / 20 mins if profit < +0.30%)
            return V12ExitEngine._sim_time_exit(future_df, entry_p, direction, sl_pct=0.005, tp_pct=0.010, max_stagnant_bars=4, stagnant_threshold=0.003)

        elif exit_policy == 'V12_F': # Partial 50% @ +0.40% + Trailing Runner (Trail 0.25%)
            return V12ExitEngine._sim_partial_trailing(future_df, entry_p, direction, partial_tp=0.004, initial_sl=0.005, trail_dist=0.0025)

        elif exit_policy == 'V12_G': # MFE-Adaptive Multi-Stage
            return V12ExitEngine._sim_mfe_adaptive(future_df, entry_p, direction)

        elif exit_policy == 'V12_H': # ATR Dynamic (SL 0.75 ATR, TP 1.5 ATR)
            sl_pct = (0.75 * atr_val) / entry_p
            tp_pct = (1.50 * atr_val) / entry_p
            return V12ExitEngine._sim_fixed(future_df, entry_p, direction, sl_pct=sl_pct, tp_pct=tp_pct)

        return V12ExitEngine._sim_fixed(future_df, entry_p, direction, sl_pct=0.005, tp_pct=0.010)

    # -----------------------------------------------------------------
    # Specialized Path-Dependent Simulation Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _sim_fixed(future_df, entry_p, direction, sl_pct, tp_pct):
        target_p = entry_p * (1 + tp_pct) if direction == 'LONG' else entry_p * (1 - tp_pct)
        stop_p = entry_p * (1 - sl_pct) if direction == 'LONG' else entry_p * (1 + sl_pct)

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            if direction == 'LONG':
                if l <= stop_p:
                    return {'net_pnl_pct': -sl_pct, 'exit_reason': 'STOP', 'bars_held': bar_idx, 'r_multiple': -1.0}
                if h >= target_p:
                    return {'net_pnl_pct': tp_pct, 'exit_reason': 'TARGET', 'bars_held': bar_idx, 'r_multiple': tp_pct / sl_pct}
            else:
                if h >= stop_p:
                    return {'net_pnl_pct': -sl_pct, 'exit_reason': 'STOP', 'bars_held': bar_idx, 'r_multiple': -1.0}
                if l <= target_p:
                    return {'net_pnl_pct': tp_pct, 'exit_reason': 'TARGET', 'bars_held': bar_idx, 'r_multiple': tp_pct / sl_pct}

        # End of day exit at last close
        last_c = future_df.iloc[-1]['Close']
        ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        return {'net_pnl_pct': ret, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': ret / sl_pct}

    @staticmethod
    def _sim_partial_be(future_df, entry_p, direction, partial_tp, final_tp, initial_sl):
        partial_hit = False
        partial_pnl = 0.0
        active_stop = entry_p * (1 - initial_sl) if direction == 'LONG' else entry_p * (1 + initial_sl)
        target_partial = entry_p * (1 + partial_tp) if direction == 'LONG' else entry_p * (1 - partial_tp)
        target_final = entry_p * (1 + final_tp) if direction == 'LONG' else entry_p * (1 - final_tp)

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            # Check partial target
            if not partial_hit:
                if (direction == 'LONG' and h >= target_partial) or (direction == 'SHORT' and l <= target_partial):
                    partial_hit = True
                    partial_pnl = partial_tp
                    # Move stop to breakeven
                    active_stop = entry_p

            # Check stop loss
            if (direction == 'LONG' and l <= active_stop) or (direction == 'SHORT' and h >= active_stop):
                if partial_hit:
                    # 50% made partial_tp, 50% exited at breakeven (0%)
                    tot_pnl = 0.5 * partial_pnl + 0.5 * 0.0
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'PARTIAL_BE', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}
                else:
                    return {'net_pnl_pct': -initial_sl, 'exit_reason': 'STOP', 'bars_held': bar_idx, 'r_multiple': -1.0}

            # Check final target for remaining 50%
            if partial_hit:
                if (direction == 'LONG' and h >= target_final) or (direction == 'SHORT' and l <= target_final):
                    tot_pnl = 0.5 * partial_pnl + 0.5 * final_tp
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'TARGET_FULL', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}

        last_c = future_df.iloc[-1]['Close']
        rem_ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
        return {'net_pnl_pct': tot_pnl, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': tot_pnl / initial_sl}

    @staticmethod
    def _sim_trailing(future_df, entry_p, direction, initial_sl, trigger_pct, trail_dist):
        trailing_active = False
        peak_excursion = 0.0
        active_stop = entry_p * (1 - initial_sl) if direction == 'LONG' else entry_p * (1 + initial_sl)

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            if direction == 'LONG':
                cur_exc = (h - entry_p) / entry_p
                if cur_exc > peak_excursion:
                    peak_excursion = cur_exc

                if peak_excursion >= trigger_pct:
                    trailing_active = True
                    # Trail by trail_dist from peak
                    new_stop = entry_p * (1 + peak_excursion - trail_dist)
                    if new_stop > active_stop:
                        active_stop = new_stop

                if l <= active_stop:
                    ret = (active_stop - entry_p) / entry_p
                    reason = 'TRAILING_STOP' if trailing_active else 'STOP'
                    return {'net_pnl_pct': ret, 'exit_reason': reason, 'bars_held': bar_idx, 'r_multiple': ret / initial_sl}

            else: # SHORT
                cur_exc = (entry_p - l) / entry_p
                if cur_exc > peak_excursion:
                    peak_excursion = cur_exc

                if peak_excursion >= trigger_pct:
                    trailing_active = True
                    new_stop = entry_p * (1 - peak_excursion + trail_dist)
                    if new_stop < active_stop:
                        active_stop = new_stop

                if h >= active_stop:
                    ret = (entry_p - active_stop) / entry_p
                    reason = 'TRAILING_STOP' if trailing_active else 'STOP'
                    return {'net_pnl_pct': ret, 'exit_reason': reason, 'bars_held': bar_idx, 'r_multiple': ret / initial_sl}

        last_c = future_df.iloc[-1]['Close']
        ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        return {'net_pnl_pct': ret, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': ret / initial_sl}

    @staticmethod
    def _sim_time_exit(future_df, entry_p, direction, sl_pct, tp_pct, max_stagnant_bars, stagnant_threshold):
        stop_p = entry_p * (1 - sl_pct) if direction == 'LONG' else entry_p * (1 + sl_pct)
        target_p = entry_p * (1 + tp_pct) if direction == 'LONG' else entry_p * (1 - tp_pct)

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            if direction == 'LONG':
                if l <= stop_p:
                    return {'net_pnl_pct': -sl_pct, 'exit_reason': 'STOP', 'bars_held': bar_idx, 'r_multiple': -1.0}
                if h >= target_p:
                    return {'net_pnl_pct': tp_pct, 'exit_reason': 'TARGET', 'bars_held': bar_idx, 'r_multiple': tp_pct / sl_pct}
                cur_ret = (c - entry_p) / entry_p
            else:
                if h >= stop_p:
                    return {'net_pnl_pct': -sl_pct, 'exit_reason': 'STOP', 'bars_held': bar_idx, 'r_multiple': -1.0}
                if l <= target_p:
                    return {'net_pnl_pct': tp_pct, 'exit_reason': 'TARGET', 'bars_held': bar_idx, 'r_multiple': tp_pct / sl_pct}
                cur_ret = (entry_p - c) / entry_p

            # Time limit check
            if bar_idx >= max_stagnant_bars and cur_ret < stagnant_threshold:
                return {'net_pnl_pct': cur_ret, 'exit_reason': 'TIME_LIMIT', 'bars_held': bar_idx, 'r_multiple': cur_ret / sl_pct}

        last_c = future_df.iloc[-1]['Close']
        ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        return {'net_pnl_pct': ret, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': ret / sl_pct}

    @staticmethod
    def _sim_partial_trailing(future_df, entry_p, direction, partial_tp, initial_sl, trail_dist):
        partial_hit = False
        partial_pnl = 0.0
        peak_excursion = 0.0
        active_stop = entry_p * (1 - initial_sl) if direction == 'LONG' else entry_p * (1 + initial_sl)
        target_partial = entry_p * (1 + partial_tp) if direction == 'LONG' else entry_p * (1 - partial_tp)

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            if direction == 'LONG':
                cur_exc = (h - entry_p) / entry_p
                if cur_exc > peak_excursion:
                    peak_excursion = cur_exc

                if not partial_hit and h >= target_partial:
                    partial_hit = True
                    partial_pnl = partial_tp
                    active_stop = entry_p  # BE stop for remaining 50%

                if partial_hit:
                    new_stop = entry_p * (1 + peak_excursion - trail_dist)
                    if new_stop > active_stop:
                        active_stop = new_stop

                if l <= active_stop:
                    rem_ret = (active_stop - entry_p) / entry_p
                    tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'PARTIAL_TRAIL', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}

            else: # SHORT
                cur_exc = (entry_p - l) / entry_p
                if cur_exc > peak_excursion:
                    peak_excursion = cur_exc

                if not partial_hit and l <= target_partial:
                    partial_hit = True
                    partial_pnl = partial_tp
                    active_stop = entry_p

                if partial_hit:
                    new_stop = entry_p * (1 - peak_excursion + trail_dist)
                    if new_stop < active_stop:
                        active_stop = new_stop

                if h >= active_stop:
                    rem_ret = (entry_p - active_stop) / entry_p
                    tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'PARTIAL_TRAIL', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}

        last_c = future_df.iloc[-1]['Close']
        rem_ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
        return {'net_pnl_pct': tot_pnl, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': tot_pnl / initial_sl}

    @staticmethod
    def _sim_mfe_adaptive(future_df, entry_p, direction):
        initial_sl = 0.005
        active_stop = entry_p * (1 - initial_sl) if direction == 'LONG' else entry_p * (1 + initial_sl)
        partial_hit = False
        partial_pnl = 0.0
        peak_exc = 0.0

        for bar_idx, (_, row) in enumerate(future_df.iterrows(), 1):
            h, l, c = row['High'], row['Low'], row['Close']

            if direction == 'LONG':
                cur_exc = (h - entry_p) / entry_p
                if cur_exc > peak_exc:
                    peak_exc = cur_exc

                # Adaptive stages
                if peak_exc >= 0.0080: # Lock 0.60%
                    return {'net_pnl_pct': 0.0070, 'exit_reason': 'LOCK_PROFIT', 'bars_held': bar_idx, 'r_multiple': 1.4}
                elif peak_exc >= 0.0060: # Trail by 0.20%
                    new_stop = entry_p * (1 + peak_exc - 0.0020)
                    if new_stop > active_stop:
                        active_stop = new_stop
                elif peak_exc >= 0.0040 and not partial_hit: # Take 50% @ 0.40%, BE
                    partial_hit = True
                    partial_pnl = 0.0040
                    active_stop = entry_p
                elif peak_exc >= 0.0025: # Tighten SL to -0.25%
                    new_stop = entry_p * (1 - 0.0025)
                    if new_stop > active_stop:
                        active_stop = new_stop

                if l <= active_stop:
                    rem_ret = (active_stop - entry_p) / entry_p
                    tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'MFE_ADAPTIVE_STOP', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}

            else: # SHORT
                cur_exc = (entry_p - l) / entry_p
                if cur_exc > peak_exc:
                    peak_exc = cur_exc

                if peak_exc >= 0.0080:
                    return {'net_pnl_pct': 0.0070, 'exit_reason': 'LOCK_PROFIT', 'bars_held': bar_idx, 'r_multiple': 1.4}
                elif peak_exc >= 0.0060:
                    new_stop = entry_p * (1 - peak_exc + 0.0020)
                    if new_stop < active_stop:
                        active_stop = new_stop
                elif peak_exc >= 0.0040 and not partial_hit:
                    partial_hit = True
                    partial_pnl = 0.0040
                    active_stop = entry_p
                elif peak_exc >= 0.0025:
                    new_stop = entry_p * (1 + 0.0025)
                    if new_stop < active_stop:
                        active_stop = new_stop

                if h >= active_stop:
                    rem_ret = (entry_p - active_stop) / entry_p
                    tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
                    return {'net_pnl_pct': tot_pnl, 'exit_reason': 'MFE_ADAPTIVE_STOP', 'bars_held': bar_idx, 'r_multiple': tot_pnl / initial_sl}

        last_c = future_df.iloc[-1]['Close']
        rem_ret = (last_c - entry_p) / entry_p if direction == 'LONG' else (entry_p - last_c) / entry_p
        tot_pnl = 0.5 * partial_pnl + 0.5 * rem_ret if partial_hit else rem_ret
        return {'net_pnl_pct': tot_pnl, 'exit_reason': 'EOD', 'bars_held': len(future_df), 'r_multiple': tot_pnl / initial_sl}
