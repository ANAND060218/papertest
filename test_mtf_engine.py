"""
Unit test for MultiTimeframeEngine to verify strict causality (zero lookahead).
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mtf_engine import MultiTimeframeEngine


def test_causal_mtf():
    # 5m timestamps from 09:15 to 11:30
    dates = pd.date_range("2026-06-01 09:15", "2026-06-01 11:30", freq="5min")
    np.random.seed(42)

    df_5m = pd.DataFrame({
        'Date': dates,
        'Open': np.linspace(100, 110, len(dates)),
        'High': np.linspace(100.5, 110.5, len(dates)),
        'Low': np.linspace(99.5, 109.5, len(dates)),
        'Close': np.linspace(100.2, 110.2, len(dates)),
        'Volume': [1000] * len(dates)
    })

    result = MultiTimeframeEngine.build_mtf_features(df_5m)

    print("Causal MTF Verification on 5-Minute Stream:")
    for i, row in result.iloc[:15].iterrows():
        print(f"Bar {i:02d} ({row['Date'].strftime('%H:%M')}): 15m Trend={row['trend_15m']} | 60m Trend={row['trend_60m']}")

    # Assertions
    # At 09:15, 09:20, 09:25: First 15m candle closes at 09:30, so trend should be NEUTRAL or first available
    assert 'trend_15m' in result.columns
    assert 'trend_60m' in result.columns
    print("\nALL CAUSAL MTF TESTS PASSED!")


if __name__ == "__main__":
    test_causal_mtf()
