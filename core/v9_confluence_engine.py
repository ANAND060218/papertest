"""
V9 Phase 4 -- Confluence Setup Engine
Generates trading candidate setups across all ablation strategy modes:
  - V9-A : Patterns Only
  - V9-B : Market Structure Only
  - V9-C : Pattern + Structure Confluence
  - V9-D : Pattern + Structure + Indicators (Volume, VWAP, RSI, ATR R:R)
  - V9-E : Above + XGBoost Probability Gating
  - V9-F : Above + Market Regime Filter
  - V9-G : Pattern + Structure + Indicators + Regime WITHOUT XGBoost
"""
import pandas as pd
import numpy as np
from datetime import time as dtime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.pattern_detector import CandlestickPatternDetector
from core.market_structure import MarketStructureEngine
from risk.dynamic_risk import DynamicRiskEngine


class V9ConfluenceEngine:
    """
    Modular setup generation engine supporting all V9 ablation variants.
    """

    def __init__(self):
        self.min_time = dtime(config.BREAKOUT_MIN_TIME_HOUR, config.BREAKOUT_MIN_TIME_MINUTE)
        self.max_time = dtime(config.BREAKOUT_MAX_TIME_HOUR, config.BREAKOUT_MAX_TIME_MINUTE)

    def prepare_dataset(self, intraday_df, daily_df=None):
        """
        Enriches intraday dataframe with patterns, structure, and indicators.
        """
        df = intraday_df.copy()

        # 1. Patterns
        df = CandlestickPatternDetector.detect_patterns(df)

        # 2. Market Structure
        df = MarketStructureEngine.compute_market_structure(df, daily_df=daily_df)

        # 3. Dynamic ATR & Technical Indicators
        df['atr_14'] = DynamicRiskEngine.compute_atr(df, period=14)
        df['vol_sma_20'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma_20'].replace(0, np.nan)

        # RSI 14
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=10).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # EMA 20 and EMA 50
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        return df

    def generate_setups(self, df, variant='V9_C', symbol='RELIANCE.NS', regime_info=None):
        """
        Generates candidate setups according to the specified ablation variant.

        Variants:
          'V9_A' : Patterns Only
          'V9_B' : Structure Only
          'V9_C' : Structure + Patterns
          'V9_D' : Structure + Patterns + Indicators + Dynamic ATR R:R
          'V9_E' : V9_D + XGBoost
          'V9_F' : V9_D + XGBoost + Regime
          'V9_G' : V9_D + Regime (No XGBoost)
        """
        setups = []
        n = len(df)

        for i in range(10, n):
            row = df.iloc[i]
            bt = row['bar_time']
            td = row['trade_date']

            # Session timing window: 09:30 - 14:30
            if bt < self.min_time or bt > self.max_time:
                continue

            entry_price = row['Close']
            atr_val = row['atr_14'] if pd.notna(row['atr_14']) and row['atr_14'] > 0 else 0.005 * entry_price
            vr = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
            rsi = row['rsi_14'] if pd.notna(row['rsi_14']) else 50.0

            # Level proximities (within 0.25% of level)
            pdl, pdh = row['pdl'], row['pdh']
            last_sl, last_sh = row['last_swing_low'], row['last_swing_high']
            vwap = row['vwap']

            near_support = (
                (pd.notna(pdl) and abs(entry_price - pdl) / pdl <= 0.0035) or
                (pd.notna(last_sl) and abs(entry_price - last_sl) / last_sl <= 0.0035) or
                (pd.notna(vwap) and entry_price >= vwap * 0.997 and entry_price <= vwap * 1.003)
            )

            near_resistance = (
                (pd.notna(pdh) and abs(entry_price - pdh) / pdh <= 0.0035) or
                (pd.notna(last_sh) and abs(entry_price - last_sh) / last_sh <= 0.0035) or
                (pd.notna(vwap) and abs(entry_price - vwap) / vwap <= 0.0035)
            )

            is_bull_pattern = (row['is_hammer'] or row['is_bullish_engulfing'])
            is_bear_pattern = (row['is_shooting_star'] or row['is_bearish_engulfing'])

            # ========================================================
            # Strategy Variant Rules
            # ========================================================
            trigger_long = False
            trigger_short = False
            setup_name = ""

            if variant == 'V9_A': # Patterns Only
                if is_bull_pattern:
                    trigger_long = True
                    setup_name = f"V9A_PATTERN_{row['pattern_name']}"
                elif is_bear_pattern:
                    trigger_short = True
                    setup_name = f"V9A_PATTERN_{row['pattern_name']}"

            elif variant == 'V9_B': # Structure Only
                # Long: Bounce off Support or Break above PDH/SH
                if near_support and row['is_green']:
                    trigger_long = True
                    setup_name = "V9B_STRUCT_SUPPORT_BOUNCE"
                elif near_resistance and row['is_red']:
                    trigger_short = True
                    setup_name = "V9B_STRUCT_RESISTANCE_REJECT"

            elif variant == 'V9_C': # Structure + Pattern Confluence
                if near_support and is_bull_pattern:
                    trigger_long = True
                    setup_name = f"V9C_CONFLUENCE_{row['pattern_name']}_SUPPORT"
                elif near_resistance and is_bear_pattern:
                    trigger_short = True
                    setup_name = f"V9C_CONFLUENCE_{row['pattern_name']}_RESIST"

            elif variant in ['V9_D', 'V9_E', 'V9_F', 'V9_G']: # Structure + Pattern + Volume + RSI + VWAP + ATR R:R
                # Long: Pattern at Support + Volume >= 1.2x + RSI recovering (< 55) + Close >= VWAP
                if (near_support and is_bull_pattern and vr >= 1.20 and rsi <= 58 and entry_price >= vwap * 0.998):
                    trigger_long = True
                    setup_name = f"V9D_MULTI_CONFLUENCE_LONG_{row['pattern_name']}"

                # Short: Pattern at Resistance + Volume >= 1.2x + RSI elevated (> 42) + Close <= VWAP
                elif (near_resistance and is_bear_pattern and vr >= 1.20 and rsi >= 42 and entry_price <= vwap * 1.002):
                    trigger_short = True
                    setup_name = f"V9D_MULTI_CONFLUENCE_SHORT_{row['pattern_name']}"

            # If setup triggered, compute dynamic boundaries
            if trigger_long:
                pattern_low = row['Low']
                risk_bounds = DynamicRiskEngine.calculate_trade_boundaries(
                    direction='LONG',
                    entry_price=entry_price,
                    current_atr=atr_val,
                    structure_support=min(pdl, last_sl) if pd.notna(pdl) and pd.notna(last_sl) else pdl,
                    structure_resistance=max(pdh, last_sh) if pd.notna(pdh) and pd.notna(last_sh) else pdh,
                    pattern_extreme=pattern_low,
                    min_rr=1.40
                )

                # For V9-D/E/F/G, enforce min RR
                if variant not in ['V9_A', 'V9_B', 'V9_C'] and not risk_bounds['is_valid_rr']:
                    continue

                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'trade_date': td,
                    'symbol': symbol,
                    'direction': 'LONG',
                    'setup_name': setup_name,
                    'pattern': row['pattern_name'],
                    'entry_price': entry_price,
                    'stop_price': risk_bounds['stop_price'],
                    'target_price': risk_bounds['target_price'],
                    'risk_distance': risk_bounds['risk_distance'],
                    'target_distance': risk_bounds['target_distance'],
                    'rr_ratio': risk_bounds['rr_ratio'],
                    'max_hold_bars': 30,
                    'volume_ratio': vr,
                    'rsi_14': rsi,
                    'market_trend': row['market_trend'],
                    'atr_14': atr_val,
                    'features': {
                        'vol_ratio': vr,
                        'rsi_14': rsi,
                        'dist_to_vwap': row['dist_to_vwap_pct'],
                        'dist_to_pdh': row['dist_to_pdh_pct'],
                        'dist_to_pdl': row['dist_to_pdl_pct'],
                        'atr_pct': (atr_val / entry_price) * 100,
                        'is_hammer': int(row['is_hammer']),
                        'is_bull_engulf': int(row['is_bullish_engulfing']),
                        'trend_bull': int(row['market_trend'] == 'BULLISH'),
                        'trend_bear': int(row['market_trend'] == 'BEARISH'),
                        'hour': row['Date'].hour,
                        'minute': row['Date'].minute
                    }
                })

            elif trigger_short:
                pattern_high = row['High']
                risk_bounds = DynamicRiskEngine.calculate_trade_boundaries(
                    direction='SHORT',
                    entry_price=entry_price,
                    current_atr=atr_val,
                    structure_support=pdl,
                    structure_resistance=pdh,
                    pattern_extreme=pattern_high,
                    min_rr=1.40
                )

                if variant not in ['V9_A', 'V9_B', 'V9_C'] and not risk_bounds['is_valid_rr']:
                    continue

                setups.append({
                    'bar_index': i,
                    'timestamp': row['Date'],
                    'trade_date': td,
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'setup_name': setup_name,
                    'pattern': row['pattern_name'],
                    'entry_price': entry_price,
                    'stop_price': risk_bounds['stop_price'],
                    'target_price': risk_bounds['target_price'],
                    'risk_distance': risk_bounds['risk_distance'],
                    'target_distance': risk_bounds['target_distance'],
                    'rr_ratio': risk_bounds['rr_ratio'],
                    'max_hold_bars': 30,
                    'volume_ratio': vr,
                    'rsi_14': rsi,
                    'market_trend': row['market_trend'],
                    'atr_14': atr_val,
                    'features': {
                        'vol_ratio': vr,
                        'rsi_14': rsi,
                        'dist_to_vwap': row['dist_to_vwap_pct'],
                        'dist_to_pdh': row['dist_to_pdh_pct'],
                        'dist_to_pdl': row['dist_to_pdl_pct'],
                        'atr_pct': (atr_val / entry_price) * 100,
                        'is_shooting_star': int(row['is_shooting_star']),
                        'is_bear_engulf': int(row['is_bearish_engulfing']),
                        'trend_bull': int(row['market_trend'] == 'BULLISH'),
                        'trend_bear': int(row['market_trend'] == 'BEARISH'),
                        'hour': row['Date'].hour,
                        'minute': row['Date'].minute
                    }
                })

        return setups
