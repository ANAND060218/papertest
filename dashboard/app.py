from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'paper_trading.db')

app = FastAPI(title="Antigravity Unified Trading Dashboard")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/portfolio")
def get_portfolio_overview():
    conn = get_db()
    
    # Get active strategies
    strats = pd.read_sql_query("SELECT * FROM strategies", conn).to_dict(orient="records")
    
    # Get equity curves
    eq_df = pd.read_sql_query("SELECT * FROM equity_curve ORDER BY date", conn)
    
    # Get open positions
    pos_df = pd.read_sql_query("SELECT * FROM positions WHERE quantity != 0", conn)
    
    # V19 Pairs
    v19_pairs = pd.read_sql_query("SELECT * FROM v19_pairs", conn).to_dict(orient="records")
    
    conn.close()
    
    # Group equity curve by date to get combined portfolio
    if not eq_df.empty:
        combined_eq = eq_df.groupby('date').agg({
            'cash': 'sum',
            'holdings_value': 'sum',
            'total_equity': 'sum'
        }).reset_index()
        combined_curve = combined_eq.to_dict(orient="records")
        strategy_curves = eq_df.groupby('strategy_id').apply(lambda x: x.to_dict(orient="records")).to_dict()
    else:
        combined_curve = []
        strategy_curves = {}
        
    return {
        "strategies": strats,
        "combined_equity": combined_curve,
        "strategy_curves": strategy_curves,
        "open_positions": pos_df.to_dict(orient="records"),
        "v19_monitor": v19_pairs
    }

@app.get("/api/trades")
def get_recent_trades():
    conn = get_db()
    trades = pd.read_sql_query("SELECT * FROM orders ORDER BY timestamp DESC LIMIT 100", conn).to_dict(orient="records")
    conn.close()
    return trades

# Mount static files (React frontend)
STATIC_DIR = os.path.join(BASE_DIR, 'dashboard', 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
