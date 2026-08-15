"""
V1.1 -- Database Inspection Script
Answers: What data do we actually have?
"""
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "nifty_10year_stock_market.db")

def inspect_database():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. List all tables
    print("=" * 70)
    print("1. TABLES IN DATABASE")
    print("=" * 70)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    for t in tables:
        print(f"  - {t}")

    for table in tables:
        print(f"\n{'=' * 70}")
        print(f"2. TABLE: {table}")
        print(f"{'=' * 70}")

        # 2. Columns and types
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns_info = cursor.fetchall()
        print("\n  COLUMNS:")
        for col in columns_info:
            print(f"    {col[1]:20s} type={col[2]}")

        # 3. Total row count
        cursor.execute(f"SELECT COUNT(*) FROM '{table}';")
        total_rows = cursor.fetchone()[0]
        print(f"\n  TOTAL ROWS: {total_rows:,}")

        # Load into pandas for analysis
        df = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)

        if df.empty:
            print("  [EMPTY TABLE]")
            continue

        # 4. Find the date/timestamp column
        date_col = None
        for candidate in ['Date', 'date', 'Datetime', 'datetime', 'Timestamp', 'timestamp']:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is None:
            for col in df.columns:
                sample = str(df[col].iloc[0])
                if '-' in sample and len(sample) >= 10:
                    date_col = col
                    break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"\n  DATE COLUMN: {date_col}")
            print(f"  EARLIEST: {df[date_col].min()}")
            print(f"  LATEST:   {df[date_col].max()}")
        else:
            print("\n  [WARNING] No date/timestamp column found!")
            print(f"  Columns: {list(df.columns)}")

        # 5. Symbols
        symbol_col = None
        for candidate in ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock']:
            if candidate in df.columns:
                symbol_col = candidate
                break

        if symbol_col:
            symbols = sorted(df[symbol_col].unique())
            print(f"\n  SYMBOL COLUMN: {symbol_col}")
            print(f"  UNIQUE SYMBOLS: {len(symbols)}")
            print(f"  SYMBOLS: {symbols}")

            # 6. Per-symbol analysis
            print(f"\n  PER-SYMBOL BREAKDOWN:")
            print(f"  {'Symbol':25s} {'Rows':>8s} {'First Date':>12s} {'Last Date':>12s} {'Has Volume':>12s}")
            print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

            vol_col = None
            for candidate in ['Volume', 'volume', 'Vol']:
                if candidate in df.columns:
                    vol_col = candidate
                    break

            for sym in symbols:
                sym_df = df[df[symbol_col] == sym]
                rows = len(sym_df)
                first = str(sym_df[date_col].min())[:10] if date_col else "N/A"
                last = str(sym_df[date_col].max())[:10] if date_col else "N/A"
                has_vol = "Yes" if (vol_col and sym_df[vol_col].notna().sum() > 0 and sym_df[vol_col].sum() > 0) else "No"
                print(f"  {sym:25s} {rows:>8,} {first:>12s} {last:>12s} {has_vol:>12s}")
        else:
            print("\n  [WARNING] No symbol column found!")

        # 7. Timestamp frequency analysis (THE KEY QUESTION)
        if date_col and symbol_col:
            print(f"\n  TIMESTAMP FREQUENCY ANALYSIS (Is this daily or intraday?):")
            test_sym = symbols[0] if symbols else None
            if test_sym:
                sym_df = df[df[symbol_col] == test_sym].sort_values(date_col).reset_index(drop=True)
                if len(sym_df) >= 10:
                    print(f"\n  First 15 timestamps for {test_sym}:")
                    for i, row in sym_df.head(15).iterrows():
                        ts = row[date_col]
                        print(f"    [{i:3d}] {ts}")

                    deltas = sym_df[date_col].diff().dropna()
                    print(f"\n  Time delta statistics:")
                    print(f"    Min delta:    {deltas.min()}")
                    print(f"    Max delta:    {deltas.max()}")
                    print(f"    Median delta: {deltas.median()}")
                    print(f"    Mode delta:   {deltas.mode().iloc[0] if len(deltas.mode()) > 0 else 'N/A'}")

                    median_seconds = deltas.median().total_seconds()
                    if median_seconds < 120:
                        freq = "1-MINUTE"
                    elif median_seconds < 600:
                        freq = "5-MINUTE"
                    elif median_seconds < 1800:
                        freq = "15-MINUTE"
                    elif median_seconds < 7200:
                        freq = "1-HOUR"
                    elif median_seconds < 172800:
                        freq = "DAILY"
                    else:
                        freq = "WEEKLY or LONGER"

                    print(f"\n  >>> DETECTED FREQUENCY: {freq} <<<")

                    if freq == "DAILY":
                        print("\n  [IMPORTANT] This is DAILY data, NOT intraday.")
                        print("  For V2 (intraday setups), we need to fetch 5-minute data separately.")
                        print("  yfinance provides max ~60 days of 5m data.")

        # 8. Gap analysis
        if date_col and symbol_col:
            print(f"\n  GAP ANALYSIS:")
            test_sym = symbols[0] if symbols else None
            if test_sym:
                sym_df = df[df[symbol_col] == test_sym].sort_values(date_col).reset_index(drop=True)
                deltas = sym_df[date_col].diff().dropna()
                median_delta = deltas.median()

                if median_delta.total_seconds() > 7200:
                    gaps = deltas[deltas > pd.Timedelta(days=4)]
                else:
                    gaps = deltas[deltas > median_delta * 3]

                print(f"    Symbol: {test_sym}")
                print(f"    Total data points: {len(sym_df)}")
                print(f"    Suspicious gaps (>3x median or >4 days): {len(gaps)}")
                if len(gaps) > 0:
                    show_count = min(len(gaps), 10)
                    if len(gaps) > 10:
                        print(f"      (showing first {show_count} of {len(gaps)})")
                    for idx in list(gaps.index)[:show_count]:
                        prev_ts = sym_df.loc[idx-1, date_col] if idx > 0 else "N/A"
                        curr_ts = sym_df.loc[idx, date_col]
                        print(f"      {prev_ts} -> {curr_ts} (gap: {gaps.loc[idx]})")

        # 9. Sample rows
        print(f"\n  SAMPLE DATA (first 5 rows):")
        print(df.head().to_string(index=False))

        print(f"\n  SAMPLE DATA (last 5 rows):")
        print(df.tail().to_string(index=False))

    conn.close()
    print(f"\n{'=' * 70}")
    print("V1.1 INSPECTION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    inspect_database()
