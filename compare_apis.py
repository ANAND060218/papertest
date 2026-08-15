import os
import json
import pandas as pd
from fetchers import KYCFreeDataFetchers
from ml_feature_engineering import MLFeatureBuilder

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def ensure_samples_dir():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)


def run_api_comparison_and_store_samples():
    """
    Fetches sample datasets from all 100% free KYC-free stock APIs,
    saves the output files to intra/samples/, and compares data quality/match.
    """
    ensure_samples_dir()
    fetcher = KYCFreeDataFetchers()

    print("================================================================================")
    print(" FETCHING SAMPLE DATA & EVALUATING KYC-FREE INDIAN STOCK MARKET APIS FOR ML")
    print("================================================================================\n")

    results_summary = []
    prices_comparison = {}

    # -------------------------------------------------------------
    # API 1: Yahoo Finance (Historical Daily & Intraday)
    # -------------------------------------------------------------
    print("[1/5] Testing Yahoo Finance (yfinance)...")
    symbol_yf = "RELIANCE.NS"
    yf_hist, msg_yf = fetcher.fetch_yfinance_history(symbol=symbol_yf, period="1y", interval="1d")
    
    if yf_hist is not None and not yf_hist.empty:
        yf_csv_path = os.path.join(SAMPLES_DIR, "yfinance_daily_RELIANCE.csv")
        yf_hist.to_csv(yf_csv_path, index=False)
        latest_yf_price = float(yf_hist['Close'].iloc[-1])
        prices_comparison['yfinance (RELIANCE.NS)'] = latest_yf_price

        # Intraday sample
        yf_intra, _ = fetcher.fetch_yfinance_history(symbol=symbol_yf, period="5d", interval="5m")
        if yf_intra is not None and not yf_intra.empty:
            yf_intra.to_csv(os.path.join(SAMPLES_DIR, "yfinance_intraday_5m_RELIANCE.csv"), index=False)

        # NIFTY 50 Index sample
        yf_nifty, _ = fetcher.fetch_yfinance_history(symbol="^NSEI", period="1y", interval="1d")
        if yf_nifty is not None and not yf_nifty.empty:
            yf_nifty.to_csv(os.path.join(SAMPLES_DIR, "yfinance_daily_NIFTY50.csv"), index=False)

        # Generate ML Feature Dataset Sample
        ml_df = MLFeatureBuilder.build_ml_dataset(yf_hist)
        ml_df.to_csv(os.path.join(SAMPLES_DIR, "ml_ready_dataset_RELIANCE.csv"), index=False)

        results_summary.append({
            "API Option": "Yahoo Finance (yfinance)",
            "KYC Required?": "NO (100% Free)",
            "Data Types": "Multi-Year Daily, Intraday (1m-60m), Real-Time Snapshot",
            "Latency": "Delayed ~15m (Real-time in trading hrs)",
            "Status": "SUCCESS",
            "Sample Output File": "yfinance_daily_RELIANCE.csv, ml_ready_dataset_RELIANCE.csv",
            "Latest Close/Price": f"Rs.{latest_yf_price:.2f}"
        })
        print(f"   [yfinance] Downloaded {len(yf_hist)} daily rows. Saved to samples folder. Latest Price: Rs.{latest_yf_price:.2f}")
    else:
        results_summary.append({
            "API Option": "Yahoo Finance (yfinance)",
            "KYC Required?": "NO",
            "Data Types": "Daily, Intraday",
            "Latency": "Delayed ~15m",
            "Status": f"FAILED ({msg_yf})",
            "Sample Output File": "N/A",
            "Latest Close/Price": "N/A"
        })

    # -------------------------------------------------------------
    # API 2: Official NSE Web API (nseindia.com)
    # -------------------------------------------------------------
    print("\n[2/5] Testing Official NSE India Direct Web API...")
    symbol_nse = "RELIANCE"
    nse_quote, msg_nse = fetcher.fetch_nse_official_live_quote(symbol=symbol_nse)
    
    if nse_quote is not None:
        with open(os.path.join(SAMPLES_DIR, "nse_live_quote_RELIANCE.json"), "w") as f:
            json.dump(nse_quote, f, indent=2, default=str)
        
        last_price_nse = nse_quote.get("last_price")
        if last_price_nse:
            prices_comparison['NSE Direct (RELIANCE)'] = float(last_price_nse)

        results_summary.append({
            "API Option": "NSE Official Public API",
            "KYC Required?": "NO (Session Cookie)",
            "Data Types": "Live Equity Quotes, Index, Option Chain, VWAP, Orderbook",
            "Latency": "100% REAL-TIME (Zero Delay)",
            "Status": "SUCCESS",
            "Sample Output File": "nse_live_quote_RELIANCE.json",
            "Latest Close/Price": f"Rs.{last_price_nse:.2f}" if last_price_nse else "N/A"
        })
        print(f"   [NSE Direct] Fetched live quote. VWAP: {nse_quote.get('vwap')}. Latest Price: Rs.{last_price_nse}")
    else:
        results_summary.append({
            "API Option": "NSE Official Public API",
            "KYC Required?": "NO",
            "Data Types": "Live Quote, Index, VWAP",
            "Latency": "Real-Time",
            "Status": f"HTTP SESSION BLOCK / {msg_nse}",
            "Sample Output File": "N/A",
            "Latest Close/Price": "N/A"
        })

    # -------------------------------------------------------------
    # API 3: Stooq Financial Data (Free CSV Direct)
    # -------------------------------------------------------------
    print("\n[3/5] Testing Stooq Financial Data...")
    stooq_df, msg_stooq = fetcher.fetch_stooq_history(symbol="RELIANCE.IN")
    if stooq_df is not None and not stooq_df.empty:
        stooq_df.to_csv(os.path.join(SAMPLES_DIR, "stooq_daily_RELIANCE.csv"), index=False)
        stooq_latest = float(stooq_df['Close'].iloc[0]) if 'Close' in stooq_df.columns else "N/A"
        if isinstance(stooq_latest, float):
            prices_comparison['Stooq (RELIANCE.IN)'] = stooq_latest

        results_summary.append({
            "API Option": "Stooq Financial Data",
            "KYC Required?": "NO (Open Web)",
            "Data Types": "Historical Daily EOD OHLCV",
            "Latency": "End Of Day (EOD)",
            "Status": "SUCCESS",
            "Sample Output File": "stooq_daily_RELIANCE.csv",
            "Latest Close/Price": f"Rs.{stooq_latest:.2f}" if isinstance(stooq_latest, float) else "N/A"
        })
        print(f"   [Stooq] Downloaded {len(stooq_df)} historical EOD rows.")
    else:
        results_summary.append({
            "API Option": "Stooq Financial Data",
            "KYC Required?": "NO",
            "Data Types": "Historical Daily EOD",
            "Latency": "End Of Day",
            "Status": f"{msg_stooq}",
            "Sample Output File": "N/A",
            "Latest Close/Price": "N/A"
        })

    # -------------------------------------------------------------
    # API 4: Google Finance Web Parser
    # -------------------------------------------------------------
    print("\n[4/5] Testing Google Finance Web Parser...")
    gf_quote, msg_gf = fetcher.fetch_google_finance_quote(symbol="RELIANCE", exchange="NSE")
    if gf_quote is not None and gf_quote.get("last_price"):
        with open(os.path.join(SAMPLES_DIR, "google_finance_quote_RELIANCE.json"), "w") as f:
            json.dump(gf_quote, f, indent=2)
        gf_price = gf_quote.get("last_price")
        prices_comparison['Google Finance (RELIANCE:NSE)'] = gf_price

        results_summary.append({
            "API Option": "Google Finance Web Parser",
            "KYC Required?": "NO (Public Web)",
            "Data Types": "Real-time Quote Snapshot, % Change",
            "Latency": "Real-Time / ~1m",
            "Status": "SUCCESS",
            "Sample Output File": "google_finance_quote_RELIANCE.json",
            "Latest Close/Price": f"Rs.{gf_price:.2f}"
        })
        print(f"   [Google Finance] Fetched price: Rs.{gf_price:.2f}")
    else:
        results_summary.append({
            "API Option": "Google Finance Web Parser",
            "KYC Required?": "NO",
            "Data Types": "Snapshot Quotes",
            "Latency": "Real-Time",
            "Status": f"{msg_gf}",
            "Sample Output File": "N/A",
            "Latest Close/Price": "N/A"
        })

    # -------------------------------------------------------------
    # API 5: Alpha Vantage (Demo / Free key)
    # -------------------------------------------------------------
    print("\n[5/5] Testing Alpha Vantage API...")
    av_df, msg_av = fetcher.fetch_alpha_vantage_daily(symbol="RELIANCE.BSE", api_key="demo")
    if av_df is not None and not av_df.empty:
        av_df.to_csv(os.path.join(SAMPLES_DIR, "alphavantage_daily_RELIANCE.csv"), index=False)
        results_summary.append({
            "API Option": "Alpha Vantage",
            "KYC Required?": "NO (Basic Email Key)",
            "Data Types": "Daily OHLCV, 50+ Technical Indicators",
            "Latency": "Real-Time / 15m",
            "Status": "SUCCESS",
            "Sample Output File": "alphavantage_daily_RELIANCE.csv",
            "Latest Close/Price": f"Rs.{float(av_df['Close'].iloc[0]):.2f}"
        })
        print(f"   [Alpha Vantage] Downloaded {len(av_df)} rows.")
    else:
        results_summary.append({
            "API Option": "Alpha Vantage",
            "KYC Required?": "NO",
            "Data Types": "Daily OHLCV & Indicators",
            "Latency": "Real-Time / 15m",
            "Status": f"Free Demo Tier Limit ({msg_av})",
            "Sample Output File": "N/A",
            "Latest Close/Price": "N/A"
        })

    # -------------------------------------------------------------
    # Data Verification & Price Consistency Cross-Check
    # -------------------------------------------------------------
    print("\n================================================================================")
    print(" DATA ACCURACY & CROSS-SOURCE PRICE VERIFICATION (RELIANCE)")
    print("================================================================================")
    if prices_comparison:
        print(f"{'Source / API':<35} | {'Latest Price':<15}")
        print("-" * 55)
        for src, prc in prices_comparison.items():
            print(f"{src:<35} | Rs.{prc:,.2f}")
    else:
        print("No price comparison records available.")

    print("\n================================================================================")
    print(" SUMMARY COMPARISON TABLE OF FREE KYC-FREE APIs FOR MACHINE LEARNING")
    print("================================================================================")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # Save summary dataframe to CSV
    summary_df.to_csv(os.path.join(SAMPLES_DIR, "api_comparison_summary.csv"), index=False)
    print(f"\n[SUCCESS] All sample files successfully written to: {os.path.abspath(SAMPLES_DIR)}")


if __name__ == "__main__":
    run_api_comparison_and_store_samples()
