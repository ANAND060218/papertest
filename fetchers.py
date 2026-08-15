import os
import json
import time
import requests
import pandas as pd

# Optional / Standard imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class KYCFreeDataFetchers:
    """
    Suite of Data Fetchers for Indian Stock Market (NSE / BSE)
    using 100% FREE, NO-KYC, NO-DEMAT Data APIs and Endpoints.
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

    # ==========================================
    # 1. YAHOO FINANCE (yfinance)
    # ==========================================
    def fetch_yfinance_history(self, symbol="RELIANCE.NS", period="1y", interval="1d"):
        """
        Fetch historical data from Yahoo Finance.
        Supports multi-year daily data or intraday intervals (1m, 5m, 15m, 60m).
        """
        if not YFINANCE_AVAILABLE:
            return None, "yfinance library not installed"

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                # Fallback to download
                df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            return df, "Success"
        except Exception as e:
            return None, f"yfinance Error: {str(e)}"

    def fetch_yfinance_latest_quote(self, symbol="RELIANCE.NS"):
        """
        Fetch latest fast quote & info metadata from Yahoo Finance.
        """
        if not YFINANCE_AVAILABLE:
            return None, "yfinance library not installed"

        try:
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            info_dict = {
                "symbol": symbol,
                "last_price": fast_info.get("lastPrice"),
                "previous_close": fast_info.get("previousClose"),
                "open": fast_info.get("open"),
                "day_high": fast_info.get("dayHigh"),
                "day_low": fast_info.get("dayLow"),
                "fifty_two_week_high": fast_info.get("yearHigh"),
                "fifty_two_week_low": fast_info.get("yearLow"),
                "volume": fast_info.get("lastVolume"),
                "market_cap": fast_info.get("marketCap"),
                "currency": fast_info.get("currency"),
                "exchange": fast_info.get("exchange")
            }
            return info_dict, "Success"
        except Exception as e:
            return None, f"yfinance quote error: {str(e)}"

    # ==========================================
    # 2. OFFICIAL NSE PUBLIC API (nseindia.com)
    # ==========================================
    def fetch_nse_official_live_quote(self, symbol="RELIANCE"):
        """
        Fetch 100% official live equity quote directly from NSE India public API.
        Uses session cookie initialization to pass security check without login/KYC.
        """
        session = requests.Session()
        session.headers.update(self.headers)
        try:
            # Step 1: Visit main homepage to establish cookies
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)

            # Step 2: Fetch quote API
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            headers_api = self.headers.copy()
            headers_api['Referer'] = f'https://www.nseindia.com/get-quotes/equity?symbol={symbol}'
            
            res = session.get(url, headers=headers_api, timeout=10)
            if res.status_code == 200:
                data = res.json()
                price_info = data.get("priceInfo", {})
                security_info = data.get("securityInfo", {})
                
                quote_summary = {
                    "symbol": symbol,
                    "company_name": data.get("info", {}).get("companyName"),
                    "last_price": price_info.get("lastPrice"),
                    "change": price_info.get("change"),
                    "pChange": price_info.get("pChange"),
                    "previous_close": price_info.get("previousClose"),
                    "open": price_info.get("open"),
                    "close": price_info.get("close"),
                    "vwap": price_info.get("vwap"),
                    "day_high": price_info.get("intraDayHighLow", {}).get("max"),
                    "day_low": price_info.get("intraDayHighLow", {}).get("min"),
                    "week_52_high": price_info.get("weekHighLow", {}).get("max"),
                    "week_52_low": price_info.get("weekHighLow", {}).get("min"),
                    "total_traded_volume": price_info.get("totalTradedVolume"),
                    "last_update_time": data.get("metadata", {}).get("lastUpdateTime"),
                    "raw_response": data
                }
                return quote_summary, "Success"
            else:
                return None, f"NSE API HTTP {res.status_code}"
        except Exception as e:
            return None, f"NSE Direct Fetch Error: {str(e)}"

    def fetch_nse_official_index(self, index_symbol="NIFTY 50"):
        """
        Fetch official live index quote (e.g. NIFTY 50, NIFTY BANK) from NSE.
        """
        session = requests.Session()
        session.headers.update(self.headers)
        try:
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)

            url = "https://www.nseindia.com/api/allIndices"
            res = session.get(url, timeout=10)
            if res.status_code == 200:
                indices = res.json().get("data", [])
                for idx in indices:
                    if idx.get("indexSymbol") == index_symbol or idx.get("index") == index_symbol:
                        return idx, "Success"
                return indices[0] if indices else None, "Index not explicitly matched, returned first item"
            else:
                return None, f"NSE Indices HTTP {res.status_code}"
        except Exception as e:
            return None, f"NSE Index Error: {str(e)}"

    # ==========================================
    # 3. ALPHA VANTAGE (Free Demo / API key)
    # ==========================================
    def fetch_alpha_vantage_daily(self, symbol="RELIANCE.BSE", api_key="demo"):
        """
        Fetch Daily Historical Data & OHLCV from Alpha Vantage.
        Note: Use free key or 'demo' key for testing.
        """
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "Time Series (Daily)" in data:
                    ts = data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(ts, orient="index")
                    df.columns = [c.split(" ")[1].capitalize() for c in df.columns]
                    df.index.name = "Date"
                    df.reset_index(inplace=True)
                    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
                    return df, "Success"
                elif "Information" in data or "Note" in data:
                    return None, data.get("Information") or data.get("Note")
                else:
                    return None, f"AlphaVantage response issue: {list(data.keys())}"
            return None, f"AlphaVantage HTTP {res.status_code}"
        except Exception as e:
            return None, f"AlphaVantage Error: {str(e)}"

    # ==========================================
    # 4. STOOQ FINANCIAL DATA (Free CSV Direct)
    # ==========================================
    def fetch_stooq_history(self, symbol="RELIANCE.IN"):
        """
        Fetch historical EOD daily data from Stooq (No key, no account, 100% open).
        Symbol format for India: ticker.IN (e.g., RELIANCE.IN, TCS.IN, NIFTY.IN)
        """
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        try:
            df = pd.read_csv(url)
            if not df.empty and "Date" in df.columns and "Close" in df.columns:
                return df, "Success"
            else:
                return None, "Stooq returned empty or invalid CSV structure"
        except Exception as e:
            return None, f"Stooq Fetch Error: {str(e)}"

    # ==========================================
    # 5. GOOGLE FINANCE WEB PARSER (Latest Quote)
    # ==========================================
    def fetch_google_finance_quote(self, symbol="RELIANCE", exchange="NSE"):
        """
        Scrape real-time/latest price snapshot from Google Finance web interface.
        URL pattern: https://www.google.com/finance/quote/RELIANCE:NSE
        """
        url = f"https://www.google.com/finance/quote/{symbol}:{exchange}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                html = res.text
                price = None
                
                # Check known DOM tags
                import re
                if 'data-last-price="' in html:
                    price_str = html.split('data-last-price="')[1].split('"')[0]
                    price = float(price_str)
                else:
                    # Search for price patterns near currency symbol
                    match = re.search(r'class="YMlKec fxKbKc"[^>]*>₹?([\d,]+\.\d{2})', html)
                    if not match:
                        match = re.search(r'data-currency-code="INR"[^>]*>₹?([\d,]+\.\d{2})', html)
                    if not match:
                        match = re.search(r'₹([\d,]+\.\d{2})', html)
                    
                    if match:
                        price = float(match.group(1).replace(',', ''))

                result = {
                    "symbol": f"{symbol}:{exchange}",
                    "last_price": price,
                    "source": "Google Finance Web Parser"
                }
                return result, "Success" if price else "Page loaded but price pattern not matched"
            return None, f"Google Finance HTTP {res.status_code}"
        except Exception as e:
            return None, f"Google Finance Error: {str(e)}"
