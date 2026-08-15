import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BSMGreeksCalculator:
    """
    Internal Black-Scholes-Merton Greek and Implied Volatility Calculator.
    Avoids reliance on vendor black-box Greeks and look-ahead bias.
    """
    
    @staticmethod
    def _d1_d2(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
        
    @classmethod
    def call_price(cls, S, K, T, r, sigma):
        if T <= 0:
            return max(0.0, S - K)
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put_price(cls, S, K, T, r, sigma):
        if T <= 0:
            return max(0.0, K - S)
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    @classmethod
    def calculate_greeks(cls, S, K, T, r, sigma, option_type='CE'):
        """
        Calculates all standard Greeks.
        option_type: 'CE' for Call, 'PE' for Put
        """
        if T <= 0.00001:
            T = 0.00001
            
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        
        # Gamma and Vega are same for both Calls and Puts
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01 # per 1% change
        
        if option_type == 'CE':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        elif option_type == 'PE':
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
        else:
            raise ValueError("option_type must be 'CE' or 'PE'")
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    @classmethod
    def implied_volatility(cls, price, S, K, T, r, option_type='CE'):
        """
        Calculates implied volatility using Brent's method.
        """
        if T <= 0:
            return 0.0
            
        def objective(sigma):
            if option_type == 'CE':
                return cls.call_price(S, K, T, r, sigma) - price
            else:
                return cls.put_price(S, K, T, r, sigma) - price
                
        try:
            # Look for a root between 1% and 300% volatility
            iv = brentq(objective, 1e-4, 3.0)
            return iv
        except (ValueError, RuntimeError):
            return np.nan
