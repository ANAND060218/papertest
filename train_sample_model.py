import os
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from ml_feature_engineering import MLFeatureBuilder

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "nifty_10year_stock_market.db")


def train_ml_model_on_10year_data():
    """
    Demonstrates how to load 10 years of historical data, engineer features,
    split chronologically, and train a Machine Learning model (RandomForest/XGBoost).
    """
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run batch_fetch_10y.py first.")
        return

    print("==================================================================================")
    print(" TRAINING MACHINE LEARNING MODEL ON 10+ YEARS HISTORICAL STOCK DATA")
    print("==================================================================================\n")

    # 1. Load Data from SQLite DB
    conn = sqlite3.connect(DB_PATH)
    raw_df = pd.read_sql_query("SELECT * FROM stock_daily_10y WHERE Symbol='RELIANCE.NS'", conn)
    conn.close()

    print(f"[1/4] Loaded {len(raw_df)} historical daily records for RELIANCE.NS")

    # 2. Engineer ML Features & Target Labels
    ml_df = MLFeatureBuilder.build_ml_dataset(raw_df)
    ml_df['Date'] = pd.to_datetime(ml_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    ml_df.dropna(subset=['Date'], inplace=True)
    ml_df.sort_values('Date', inplace=True)
    ml_df.reset_index(drop=True, inplace=True)

    print(f"[2/4] Engineered Technical Features (MA, RSI, MACD, BB, ATR). Clean Rows: {len(ml_df)}")

    # Define Feature Columns & Target Variable
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'MA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Width', 'Return_1D', 'Return_5D']
    target_col = 'Target_Direction'

    # 3. Chronological Time-Series Train/Test Split (Prevent Data Leakage)
    # Train: 2016 to 2024 | Test: 2025 to 2026
    train_mask = ml_df['Date'] < '2025-01-01'
    test_mask = ml_df['Date'] >= '2025-01-01'

    X_train, y_train = ml_df.loc[train_mask, feature_cols], ml_df.loc[train_mask, target_col]
    X_test, y_test = ml_df.loc[test_mask, feature_cols], ml_df.loc[test_mask, target_col]

    print(f"[3/4] Chronological Split -> Training Set: {len(X_train)} days | Out-of-Sample Test Set: {len(X_test)} days")

    # 4. Train Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Predict & Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==================================================================================")
    print(" 📊 MODEL PERFORMANCE EVALUATION (OUT-OF-SAMPLE TEST 2025-2026)")
    print("==================================================================================")
    print(f"Overall Classification Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))

    # Feature Importance Ranking
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[RANKING] Top Feature Importances:")
    print(importances.head(8).to_string())
    print("==================================================================================")


if __name__ == "__main__":
    train_ml_model_on_10year_data()
