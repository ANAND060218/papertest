import sqlite3
import pandas as pd

conn = sqlite3.connect("data/nifty_10year_stock_market.db")
df = pd.read_sql_query("SELECT * FROM stock_daily_10y WHERE Symbol='RELIANCE.NS'", conn)
print("DF shape:", df.shape)
print("DF columns:", df.columns.tolist())
print(df.head(2))
conn.close()
