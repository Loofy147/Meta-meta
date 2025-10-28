import pandas as pd
import numpy as np
import psycopg2
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def run_backtest(symbol, start_date, end_date, initial_cash=100000):
    """
    Runs a vectorized backtest with realistic cost simulation.
    """
    with open('config/main.json', 'r') as f:
        config = json.load(f)['backtester']

    commission_pct = config['commission_pct']
    slippage_pct = config['slippage_pct']

    conn = get_db_connection()
    query = """
        SELECT f.time, f.rsi, f.macd, f.macds, c.close
        FROM features_1m f JOIN candles_1m c ON f.time = c.time AND f.symbol = c.symbol
        WHERE f.symbol = %s AND f.time BETWEEN %s AND %s ORDER BY f.time;
    """
    df = pd.read_sql(query, conn, params=(symbol, start_date, end_date), index_col='time')
    conn.close()

    if df.empty:
        return {"error": "No data found for the given symbol and date range."}

    # Generate signals
    df['signal'] = 0
    df.loc[df['macd'] > df['macds'], 'signal'] = 1
    df.loc[df['macd'] < df['macds'], 'signal'] = -1

    # Simulate portfolio
    cash = initial_cash
    position = 0
    portfolio_values = []

    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]

        # Apply slippage
        buy_price = current_close * (1 + slippage_pct)
        sell_price = current_close * (1 - slippage_pct)

        # Buy signal
        if df['signal'].iloc[i] == 1 and cash > 0:
            position = (cash / buy_price) * (1 - commission_pct)
            cash = 0
        # Sell signal
        elif df['signal'].iloc[i] == -1 and position > 0:
            cash = (position * sell_price) * (1 - commission_pct)
            position = 0

        portfolio_values.append(cash + position * current_close)

    if not portfolio_values:
        return {"error": "No trades were executed."}

    # Calculate KPIs
    returns = pd.Series(portfolio_values, index=df.index[1:]).pct_change().dropna()
    sharpe_ratio = np.sqrt(252 * 24 * 60) * returns.mean() / returns.std() if returns.std() != 0 else 0
    total_return = (portfolio_values[-1] / initial_cash - 1) * 100

    return {
        "symbol": symbol,
        "total_return_pct": round(total_return, 2),
        "annualized_sharpe_ratio": round(sharpe_ratio, 2),
        "final_portfolio_value": round(portfolio_values[-1], 2),
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct
    }

if __name__ == '__main__':
    print("Running backtest with cost simulation...")

    # Use a more sensible default date range (e.g., the last 24 hours)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    try:
        results = run_backtest('BTC/USDT', start_date.isoformat(), end_date.isoformat())
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Could not run backtest: {e}")
