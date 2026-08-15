import pandas as pd
import numpy as np


class MLFeatureBuilder:
    """
    Engineers technical indicators, rolling features, and target variables
    specifically designed for training Machine Learning models (Regression & Classification).
    """

    @staticmethod
    def build_ml_dataset(df_raw, close_col="Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
        """
        Takes raw OHLCV DataFrame and adds technical indicators + ML targets.
        """
        if df_raw is None or df_raw.empty:
            raise ValueError("Input DataFrame is empty or None")

        df = df_raw.copy()

        # Ensure correct numeric types
        for col in [close_col, open_col, high_col, low_col, volume_col]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df[close_col]

        # ----------------------------------------------------
        # 1. Moving Average Features
        # ----------------------------------------------------
        df['MA_10'] = close.rolling(window=10).mean()
        df['MA_20'] = close.rolling(window=20).mean()
        df['MA_50'] = close.rolling(window=50).mean()
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()

        # ----------------------------------------------------
        # 2. Momentum & Oscillators (RSI, MACD)
        # ----------------------------------------------------
        # RSI (14 periods)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # ----------------------------------------------------
        # 3. Volatility Features (Bollinger Bands & ATR)
        # ----------------------------------------------------
        std_20 = close.rolling(window=20).std()
        df['BB_Upper'] = df['MA_20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA_20'] - (std_20 * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA_20']

        if high_col in df.columns and low_col in df.columns:
            tr1 = df[high_col] - df[low_col]
            tr2 = (df[high_col] - close.shift(1)).abs()
            tr3 = (df[low_col] - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(window=14).mean()

        # ----------------------------------------------------
        # 4. Lag & Return Features
        # ----------------------------------------------------
        df['Return_1D'] = close.pct_change()
        df['Return_5D'] = close.pct_change(periods=5)
        df['Close_Lag_1'] = close.shift(1)
        df['Close_Lag_2'] = close.shift(2)

        # ----------------------------------------------------
        # 5. Machine Learning Targets (Ground Truth Labels)
        # ----------------------------------------------------
        # Target 1: Regression Target -> Next Bar/Day Close Price
        df['Target_Next_Close'] = close.shift(-1)

        # Target 2: Regression Target -> Next Bar % Return
        df['Target_Next_Return'] = df['Target_Next_Close'].pct_change(fill_method=None)

        # Target 3: Binary Classification Target -> 1 if Next Close > Current Close, else 0
        df['Target_Direction'] = (df['Target_Next_Close'] > close).astype(int)

        # Clean missing values created by rolling windows & forward shifts
        clean_df = df.dropna().reset_index(drop=True)
        return clean_df


if __name__ == "__main__":
    from fetchers import KYCFreeDataFetchers
    
    fetcher = KYCFreeDataFetchers()
    df_raw, msg = fetcher.fetch_yfinance_history("RELIANCE.NS", period="1y", interval="1d")
    if df_raw is not None:
        ml_df = MLFeatureBuilder.build_ml_dataset(df_raw)
        print("ML Dataset Successfully Generated!")
        print(f"Original Rows: {len(df_raw)} -> ML Features Rows: {len(ml_df)}")
        print("\nFeatures preview:")
        print(ml_df[['Date', 'Close', 'MA_10', 'RSI_14', 'MACD', 'Target_Next_Close', 'Target_Direction']].tail())
    else:
        print("Error fetching data:", msg)
