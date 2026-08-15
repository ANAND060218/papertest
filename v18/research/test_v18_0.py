import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from v18.core.v18_cost_model import calculate_option_trade_costs, calculate_spread_costs
from v18.core.v18_volatility_engine import BSMGreeksCalculator

def test_cost_model():
    print("Testing 2026 Cost Model...")
    # Sell NIFTY Option at Rs 100, Lot size 25, 1 lot
    premium = 100.0
    qty = 25
    
    # 1. Test Sell Side
    sell_costs = calculate_option_trade_costs(premium, qty, is_buy=False, brokerage_per_order=20.0)
    
    turnover = 100.0 * 25 # 2500
    expected_stt = turnover * 0.0015 # 0.15% = 3.75
    expected_exchange = turnover * 0.00035 # 0.035% = 0.875
    expected_gst = (20.0 + expected_exchange) * 0.18 # 3.7575
    expected_sebi = turnover * 0.000001 # 0.0025
    expected_stamp = 0.0 # Sale side no stamp
    
    total_expected = 20.0 + expected_exchange + expected_gst + expected_stt + expected_sebi + expected_stamp
    
    assert math.isclose(sell_costs['stt'], expected_stt, rel_tol=1e-5), f"STT mismatch: {sell_costs['stt']} != {expected_stt}"
    assert math.isclose(sell_costs['total_cost'], total_expected, rel_tol=1e-5), f"Total mismatch"
    print("  [OK] Sell side costs correctly implement 0.15% STT.")
    
    # 2. Test Buy Side
    buy_costs = calculate_option_trade_costs(premium, qty, is_buy=True, brokerage_per_order=20.0)
    assert buy_costs['stt'] == 0.0, "STT should be 0 on buy side"
    assert buy_costs['stamp_duty'] > 0.0, "Stamp duty should apply on buy side"
    print("  [OK] Buy side costs correctly omit STT.")

def test_volatility_engine():
    print("Testing BSM Greeks Calculator...")
    
    # Common test case: S=100, K=100, T=1.0, r=0.05, sigma=0.2
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    call_p = BSMGreeksCalculator.call_price(S, K, T, r, sigma)
    put_p = BSMGreeksCalculator.put_price(S, K, T, r, sigma)
    
    # Put-Call Parity: C - P = S - K*e^(-rT)
    pc_left = call_p - put_p
    pc_right = S - K * math.exp(-r * T)
    assert math.isclose(pc_left, pc_right, rel_tol=1e-5), "Put-Call parity failed"
    print(f"  [OK] Call Price: {call_p:.2f}, Put Price: {put_p:.2f}. Put-Call Parity holds.")
    
    greeks_c = BSMGreeksCalculator.calculate_greeks(S, K, T, r, sigma, 'CE')
    greeks_p = BSMGreeksCalculator.calculate_greeks(S, K, T, r, sigma, 'PE')
    
    assert greeks_c['delta'] > 0 and greeks_c['delta'] < 1
    assert greeks_p['delta'] < 0 and greeks_p['delta'] > -1
    assert math.isclose(greeks_c['delta'] - greeks_p['delta'], 1.0, rel_tol=1e-5), "Delta parity failed"
    print("  [OK] Deltas calculate correctly.")
    
    # Test Implied Volatility calculation
    iv_c = BSMGreeksCalculator.implied_volatility(call_p, S, K, T, r, 'CE')
    assert math.isclose(iv_c, sigma, rel_tol=1e-4), f"IV calculation failed: {iv_c} != {sigma}"
    print("  [OK] Implied Volatility (Newton/Brent) recovers input sigma exactly.")

if __name__ == "__main__":
    test_cost_model()
    test_volatility_engine()
    print("\nALL V18.0 TESTS PASSED. Ready for V18.1 Directional Debit Spreads.")
