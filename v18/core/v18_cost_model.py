"""
V18.0 Options Cost Model
Implements 2026 Indian Option transaction costs.
Effective STT: 0.15% on sale side
"""

def calculate_option_trade_costs(premium, qty, is_buy, brokerage_per_order=20.0):
    """
    Calculates total transaction costs for a single option leg trade.
    
    Args:
        premium (float): Option price
        qty (int): Number of units (lot size * number of lots)
        is_buy (bool): True if buying the option, False if selling
        brokerage_per_order (float): Flat brokerage fee
        
    Returns:
        dict: Breakdown of all costs
    """
    turnover = premium * qty
    
    # Brokerage
    brokerage = brokerage_per_order
    
    # Exchange Transaction Charges (0.035% of premium turnover)
    exchange_charges = turnover * 0.00035
    
    # GST (18% on Brokerage + Exchange Charges)
    gst = (brokerage + exchange_charges) * 0.18
    
    # STT (Securities Transaction Tax) - 0.15% on sell side premium
    stt = (turnover * 0.0015) if not is_buy else 0.0
    
    # Stamp Duty - 0.003% on buy side premium
    stamp_duty = (turnover * 0.00003) if is_buy else 0.0
    
    # SEBI Turnover Fee - 0.0001% on premium
    sebi_fee = turnover * 0.000001
    
    total_taxes_and_charges = brokerage + exchange_charges + gst + stt + stamp_duty + sebi_fee
    
    return {
        'turnover': turnover,
        'brokerage': brokerage,
        'exchange_charges': exchange_charges,
        'gst': gst,
        'stt': stt,
        'stamp_duty': stamp_duty,
        'sebi_fee': sebi_fee,
        'total_cost': total_taxes_and_charges
    }

def calculate_spread_costs(buy_leg_premium, sell_leg_premium, qty):
    """
    Calculates combined cost for a Debit/Credit Spread executed simultaneously.
    Assuming 2 orders (one buy, one sell).
    """
    buy_costs = calculate_option_trade_costs(buy_leg_premium, qty, is_buy=True)
    sell_costs = calculate_option_trade_costs(sell_leg_premium, qty, is_buy=False)
    
    total_cost = buy_costs['total_cost'] + sell_costs['total_cost']
    return total_cost, buy_costs, sell_costs
