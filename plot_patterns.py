import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Try importing Plotly for interactive web charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "nifty_10year_stock_market.db")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def detect_support_resistance(df, distance=10):
    """
    Detects local peaks (Resistance) and valleys (Support) using Scipy peak finding.
    """
    prices = df['Close'].values
    
    # Resistance peaks
    peaks, _ = find_peaks(prices, distance=distance)
    # Support valleys (inverted peaks)
    valleys, _ = find_peaks(-prices, distance=distance)

    return peaks, valleys


def detect_candlestick_patterns(df):
    """
    Rule-based detection for key Candlestick Patterns:
    1. Bullish Engulfing
    2. Hammer / Pin Bar
    """
    df = df.copy()
    df['Pattern'] = None

    open_p = df['Open'].values
    high_p = df['High'].values
    low_p = df['Low'].values
    close_p = df['Close'].values

    for i in range(1, len(df)):
        body_curr = abs(close_p[i] - open_p[i])
        body_prev = abs(close_p[i-1] - open_p[i-1])

        # 1. Bullish Engulfing: Prev candle is Red, Current is Green and engulfs prev body
        if (close_p[i-1] < open_p[i-1]) and (close_p[i] > open_p[i]):
            if (close_p[i] >= open_p[i-1]) and (open_p[i] <= close_p[i-1]):
                df.iloc[i, df.columns.get_loc('Pattern')] = 'Bullish Engulfing'

        # 2. Hammer: Small body at top, lower shadow at least 2x body length
        lower_shadow = min(open_p[i], close_p[i]) - low_p[i]
        if lower_shadow > (2 * body_curr) and (high_p[i] - max(open_p[i], close_p[i])) < body_curr:
            df.iloc[i, df.columns.get_loc('Pattern')] = 'Hammer'

    return df


def generate_interactive_chart(symbol="RELIANCE.NS"):
    """
    Generates a full interactive Candlestick Chart with Volume, Moving Averages,
    Support/Resistance lines, and Pattern Annotations.
    Saves an interactive HTML chart file to intra/samples/.
    """
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run batch_fetch_10y.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM stock_daily_10y WHERE Symbol='{symbol}'", conn)
    conn.close()

    if df.empty:
        print(f"No data found for symbol {symbol}")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df = df.tail(200).reset_index(drop=True)  # Focus on recent 200 trading days

    # 1. Calculate Moving Averages & RSI
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    # 2. Detect Support & Resistance
    peaks, valleys = detect_support_resistance(df)

    # 3. Detect Candlestick Patterns
    df_pattern = detect_candlestick_patterns(df)

    print("==================================================================================")
    print(f" PATTERN RECOGNITION & GRAPH ANALYSIS FOR {symbol}")
    print("==================================================================================")
    print(f"Resistance Levels Detected : {len(peaks)}")
    print(f"Support Levels Detected    : {len(valleys)}")
    pattern_counts = df_pattern['Pattern'].value_counts()
    print(f"Patterns Detected          :\n{pattern_counts.to_string()}")

    if PLOTLY_AVAILABLE:
        # Create Subplots: Subplot 1 (Candlestick + MAs), Subplot 2 (Volume)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="OHLC Price"
        ), row=1, col=1)

        # Moving Averages
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='blue', width=1.5), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], line=dict(color='orange', width=1.5), name="SMA 50"), row=1, col=1)

        # Support & Resistance Peak Markers
        fig.add_trace(go.Scatter(
            x=df['Date'].iloc[peaks], y=df['Close'].iloc[peaks],
            mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Resistance Peak'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['Date'].iloc[valleys], y=df['Close'].iloc[valleys],
            mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Support Valley'
        ), row=1, col=1)

        # Volume Bar
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for idx, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            title=f"Technical Pattern & Graph Recognition - {symbol}",
            yaxis_title="Stock Price (INR)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700
        )

        html_path = os.path.join(SAMPLES_DIR, f"{symbol.replace('.', '_')}_pattern_graph.html")
        fig.write_html(html_path)
        print(f"\n✅ Interactive Chart HTML saved to: {html_path}")

    return df_pattern


if __name__ == "__main__":
    generate_interactive_chart("RELIANCE.NS")
