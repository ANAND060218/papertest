import pandas as pd
import re
from datetime import datetime

def parse_option_symbol(symbol):
    """
    Parses a symbol like 'BANKNIFTY02JAN2027500PE' into its components.
    """
    # Regex: (BANKNIFTY|NIFTY)([0-9]{2}[A-Z]{3}[0-9]{2})([0-9]+)(CE|PE)
    match = re.match(r"(BANKNIFTY|NIFTY)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)", symbol)
    if not match:
        return None
    
    underlying = match.group(1)
    expiry_str = match.group(2)
    strike = int(match.group(3))
    opt_type = match.group(4)
    
    expiry_date = datetime.strptime(expiry_str, "%d%b%y")
    
    return {
        'underlying': underlying,
        'expiry': expiry_date,
        'strike': strike,
        'type': opt_type
    }

def load_options_data_for_day(filepath):
    """
    Loads raw Kaggle CSV minute data and sets up datetime.
    Does NOT parse all symbols to save massive compute time.
    """
    df = pd.read_csv(filepath)
    # Filter only CE and PE
    df = df[df['symbol'].str.endswith('CE') | df['symbol'].str.endswith('PE')].copy()
    
    # Convert dates
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    return df

def get_point_in_time_chain(df, current_time, expiry_date=None):
    """
    Returns the option chain available exactly at 'current_time'.
    Avoids look-ahead bias by strictly filtering on datetime.
    Parses symbols ONLY for this specific minute to save time.
    """
    chain = df[df['datetime'] == current_time].copy()
    if chain.empty:
        return chain
        
    # Parse symbols only for this minute slice
    parsed = chain['symbol'].apply(parse_option_symbol).apply(pd.Series)
    chain = pd.concat([chain, parsed], axis=1)
    
    if expiry_date:
        chain = chain[chain['expiry'] == expiry_date]
    return chain

def find_atm_strike(spot_price, strike_interval=100):
    """
    Finds the ATM strike for BankNifty (typically 100 intervals).
    """
    return round(spot_price / strike_interval) * strike_interval
