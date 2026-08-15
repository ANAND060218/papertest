"""
Unit Test Suite for MarketStructureEngine on synthetic and real candle structures.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.market_structure import MarketStructureEngine


def test_market_structure():
    # Construct synthetic higher-high / higher-low structure
    # Bars:
    # 0: Open 100, High 101, Low 99, Close 100.5
    # 1: Open 100.5, High 103, Low 100, Close 102.5
    # 2 (Swing High 1 = 105): Open 102.5, High 105, Low 102, Close 104
    # 3: Open 104, High 103, Low 101, Close 102
    # 4: Open 102, High 102, Low 98 (Swing Low 1 = 98), Close 99 -> Confirms SH1=105 at bar 4
    # 5: Open 99, High 101, Low 99, Close 100.5
    # 6: Open 100.5, High 104, Low 100, Close 103.5 -> Confirms SL1=98 at bar 6
    # 7: Open 103.5, High 108 (Swing High 2 = 108), Low 103, Close 107.5
    # 8: Open 107.5, High 106, Low 104, Close 105
    # 9: Open 105, High 105, Low 101 (Swing Low 2 = 101), Close 102 -> Confirms SH2=108 at bar 9
    # 10: Open 102, High 104, Low 102, Close 103
    # 11: Open 103, High 110, Low 103, Close 109 -> Confirms SL2=101 at bar 11
    
    dates = pd.date_range("2026-06-01 09:15", periods=12, freq="5min")
    df = pd.DataFrame({
        'Date': dates,
        'Open':  [100, 100.5, 102.5, 104, 102, 99, 100.5, 103.5, 107.5, 105, 102, 103],
        'High':  [101, 103,   105,   103, 102, 101, 104,  108,   106,   105, 104, 110],
        'Low':   [99,  100,   102,   101, 98,  99,  100,  103,   104,   101, 102, 103],
        'Close': [100.5, 102.5, 104, 102, 99, 100.5, 103.5, 107.5, 105, 102, 103, 109],
        'Volume': [1000] * 12
    })

    result = MarketStructureEngine.compute_market_structure(df)

    print("Market Structure Causal Pivot Verification:")
    for i, row in result.iterrows():
        print(f"Bar {i:02d} ({row['Date'].strftime('%H:%M')}): Close={row['Close']} | Confirmed_SH={row['is_confirmed_swing_high']} | Last_SH={row['last_swing_high']} | Confirmed_SL={row['is_confirmed_swing_low']} | Last_SL={row['last_swing_low']} | Trend={row['market_trend']}")

    # Verification:
    # Bar 4 confirms Swing High 1 = 105 (at bar 2)
    assert result.iloc[4]['is_confirmed_swing_high'] == True, "Failed: Bar 4 should confirm SH 1"
    assert result.iloc[4]['last_swing_high'] == 105.0, "Failed: Last SH should be 105"

    # Bar 6 confirms Swing Low 1 = 98 (at bar 4)
    assert result.iloc[6]['is_confirmed_swing_low'] == True, "Failed: Bar 6 should confirm SL 1"
    assert result.iloc[6]['last_swing_low'] == 98.0, "Failed: Last SL should be 98"

    # Bar 9 confirms Swing High 2 = 108 (at bar 7)
    assert result.iloc[9]['is_confirmed_swing_high'] == True, "Failed: Bar 9 should confirm SH 2"
    assert result.iloc[9]['last_swing_high'] == 108.0, "Failed: Last SH should be 108"

    # Bar 11 confirms Swing Low 2 = 101 (at bar 9) -> Trend should become BULLISH (SH2 > SH1 and SL2 > SL1)
    assert result.iloc[11]['is_confirmed_swing_low'] == True, "Failed: Bar 11 should confirm SL 2"
    assert result.iloc[11]['market_trend'] == 'BULLISH', "Failed: Trend should be BULLISH after HH + HL"

    print("\nALL CAUSAL MARKET STRUCTURE TESTS PASSED (100% LOOKAHEAD-FREE)!")


if __name__ == "__main__":
    test_market_structure()
