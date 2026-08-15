import os
import json
import yfinance as yf
import pandas as pd

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def fetch_and_compare_intraday_stocks():
    """
    Fetches rich company details, latest today prices, intraday 5m candles,
    and calculates Intraday VWAP for top Indian equities without KYC.
    """
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

    # Top Indian Bluechip Companies
    symbols_map = {
        "RELIANCE.NS": "Reliance Industries Ltd",
        "TCS.NS": "Tata Consultancy Services Ltd",
        "INFY.NS": "Infosys Ltd",
        "HDFCBANK.NS": "HDFC Bank Ltd",
        "ICICIBANK.NS": "ICICI Bank Ltd",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel Ltd",
        "^NSEI": "NIFTY 50 Index"
    }

    print("==================================================================================")
    print(" TODAY'S INTRADAY MARKET DATA & MULTI-COMPANY COMPARISON (NSE / INDIA)")
    print("==================================================================================\n")

    company_summary_list = []
    all_intraday_dfs = []

    for sym, default_name in symbols_map.items():
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            comp_name = info.get("longName") or info.get("shortName") or default_name
            last_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            day_high = info.get("dayHigh")
            day_low = info.get("dayLow")
            prev_close = info.get("previousClose")
            volume = info.get("regularMarketVolume")
            week52_high = info.get("fiftyTwoWeekHigh")
            week52_low = info.get("fiftyTwoWeekLow")

            # Fetch Intraday 5-Minute Candles Today
            intra_df = ticker.history(period="1d", interval="5m")

            vwap_val = None
            price_vs_vwap = "N/A"

            if not intra_df.empty:
                # Intraday VWAP Calculation: Cumulative (Volume * TypicalPrice) / Cumulative Volume
                typical_price = (intra_df['High'] + intra_df['Low'] + intra_df['Close']) / 3
                cum_vol = intra_df['Volume'].cumsum()
                cum_vol_tp = (intra_df['Volume'] * typical_price).cumsum()
                
                # Replace 0 volume cumulative to avoid division by zero
                intra_df['VWAP'] = (cum_vol_tp / cum_vol.replace(0, 1)).round(2)
                
                vwap_val = float(intra_df['VWAP'].iloc[-1])
                if last_price is None and not intra_df.empty:
                    last_price = float(intra_df['Close'].iloc[-1])
                
                if last_price and vwap_val:
                    if last_price > vwap_val:
                        price_vs_vwap = "ABOVE VWAP (Bullish)"
                    elif last_price < vwap_val:
                        price_vs_vwap = "BELOW VWAP (Bearish)"
                    else:
                        price_vs_vwap = "AT VWAP"

                # Store stock tag
                intra_df['Symbol'] = sym
                intra_df['CompanyName'] = comp_name
                intra_df.reset_index(inplace=True)
                all_intraday_dfs.append(intra_df)

            # Calculate Price Change %
            p_change = None
            if last_price and prev_close:
                p_change = round(((last_price - prev_close) / prev_close) * 100, 2)

            record = {
                "Company Name": comp_name,
                "Symbol": sym,
                "Current Price": f"Rs.{last_price:,.2f}" if last_price else "N/A",
                "Prev Close": f"Rs.{prev_close:,.2f}" if prev_close else "N/A",
                "Change %": f"{p_change}%" if p_change is not None else "N/A",
                "Day High": f"Rs.{day_high:,.2f}" if day_high else "N/A",
                "Day Low": f"Rs.{day_low:,.2f}" if day_low else "N/A",
                "Intraday VWAP": f"Rs.{vwap_val:,.2f}" if vwap_val else "N/A",
                "VWAP Status": price_vs_vwap,
                "Traded Volume": f"{volume:,}" if volume else "N/A",
                "52W High": f"Rs.{week52_high:,.2f}" if week52_high else "N/A",
                "52W Low": f"Rs.{week52_low:,.2f}" if week52_low else "N/A",
            }
            company_summary_list.append(record)

            print(f"[SUCCESS] Fetched: {comp_name:<30} ({sym:<12}) -> Price: {record['Current Price']} | VWAP: {record['Intraday VWAP']}")

        except Exception as e:
            print(f"[ERROR] Error fetching {sym}: {e}")

    # Build Comparison Dataframe
    comparison_df = pd.DataFrame(company_summary_list)
    print("\n==================================================================================")
    print(" INTRADAY MULTI-COMPANY COMPARISON TABLE (TODAY'S DETAILS)")
    print("==================================================================================")
    print(comparison_df.to_string(index=False))

    # Save outputs to intra/samples/
    summary_path = os.path.join(SAMPLES_DIR, "today_intraday_multi_company_comparison.csv")
    comparison_df.to_csv(summary_path, index=False)

    if all_intraday_dfs:
        combined_intra_df = pd.concat(all_intraday_dfs, ignore_index=True)
        intra_candles_path = os.path.join(SAMPLES_DIR, "today_intraday_5m_candles_all_companies.csv")
        combined_intra_df.to_csv(intra_candles_path, index=False)
        print(f"\n[SUCCESS] Combined 5-minute Intraday Candles saved to: {intra_candles_path}")

    # Write Rich JSON Sample
    json_path = os.path.join(SAMPLES_DIR, "today_intraday_rich_quotes.json")
    with open(json_path, "w") as f:
        json.dump(company_summary_list, f, indent=2)

    print(f"[SUCCESS] Summary comparison saved to: {summary_path}")


if __name__ == "__main__":
    fetch_and_compare_intraday_stocks()
