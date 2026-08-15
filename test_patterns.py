"""
Unit Test Suite for CandlestickPatternDetector on Synthetic Candles.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pattern_detector import CandlestickPatternDetector


def test_synthetic_patterns():
    # 1. Synthetic Hammer: Open 100, High 101, Low 90, Close 100.8
    # Body = 0.8, Range = 11.0, Upper Wick = 0.2, Lower Wick = 10.0 (>= 2 * body)
    data = [
        {"Date": "2026-06-01 09:15", "Open": 100.0, "High": 101.0, "Low": 90.0, "Close": 100.8, "Volume": 1000},
        # 2. Synthetic Shooting Star: Open 100.0, High 110.0, Low 99.5, Close 99.8
        # Body = 0.2, Range = 10.5, Upper Wick = 10.0 (>= 2 * body), Lower Wick = 0.3
        {"Date": "2026-06-01 09:20", "Open": 100.0, "High": 110.0, "Low": 99.5, "Close": 99.8, "Volume": 1000},
        # 3. Setup for Bullish Engulfing: Prior Red Candle
        {"Date": "2026-06-01 09:25", "Open": 102.0, "High": 102.5, "Low": 98.0, "Close": 98.5, "Volume": 1000},
        # 4. Bullish Engulfing Green Candle: Open 98.0, High 103.5, Low 97.5, Close 103.0
        {"Date": "2026-06-01 09:30", "Open": 98.0, "High": 103.5, "Low": 97.5, "Close": 103.0, "Volume": 2000},
        # 5. Doji: Open 100.0, High 105.0, Low 95.0, Close 100.2 (Body 0.2 <= 1.0 = 10% of 10)
        {"Date": "2026-06-01 09:35", "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 100.2, "Volume": 1000},
        # 6. Inside Bar Parent: Open 100, High 110, Low 90, Close 105
        {"Date": "2026-06-01 09:40", "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": 1000},
        # 7. Inside Bar Child: Open 95, High 102, Low 92, Close 98 (Contained completely within 90 - 110)
        {"Date": "2026-06-01 09:45", "Open": 95.0, "High": 102.0, "Low": 92.0, "Close": 98.0, "Volume": 1000},
    ]

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])

    result = CandlestickPatternDetector.detect_patterns(df)

    print("Pattern Detection Results:")
    for i, row in result.iterrows():
        print(f"Row {i} ({row['Date'].strftime('%H:%M')}): {row['pattern_name']} | Hammer={row['is_hammer']} | ShootingStar={row['is_shooting_star']} | BullEngulf={row['is_bullish_engulfing']} | InsideBar={row['is_inside_bar']} | Doji={row['is_doji']}")

    # Assertions
    assert result.iloc[0]['is_hammer'] == True, "Failed to detect Hammer"
    assert result.iloc[1]['is_shooting_star'] == True, "Failed to detect Shooting Star"
    assert result.iloc[3]['is_bullish_engulfing'] == True, "Failed to detect Bullish Engulfing"
    assert result.iloc[4]['is_doji'] == True, "Failed to detect Doji"
    assert result.iloc[6]['is_inside_bar'] == True, "Failed to detect Inside Bar"
    print("\nALL SYNTHETIC PATTERN TESTS PASSED SUCCESSFULLY (100% ACCURACY)!")


if __name__ == "__main__":
    test_synthetic_patterns()
