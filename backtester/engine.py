import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv

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
    Runs a vectorized backtest and returns a dictionary of KPIs.
    """
    conn = get_db_connection()

    # Fetch historical data using parameterized queries
    query = """
        SELECT f.time, f.rsi, f.macd, f.macds, c.close
        FROM features_1m f
        JOIN candles_1m c ON f.time = c.time AND f.symbol = c.symbol
        WHERE f.symbol = %s AND f.time BETWEEN %s AND %s
        ORDER BY f.time;
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
        # Buy signal
        if df['signal'].iloc[i] == 1 and cash > 0:
            position = cash / current_close
            cash = 0
        # Sell signal
        elif df['signal'].iloc[i] == -1 and position > 0:
            cash = position * current_close
            position = 0

        portfolio_values.append(cash + position * current_close)

    if not portfolio_values:
        return {"error": "No trades were executed."}

    # Calculate KPIs
    returns = pd.Series(portfolio_values, index=df.index[1:]).pct_change().dropna()
    sharpe_ratio = np.sqrt(252 * 24 * 60) * returns.mean() / returns.std() if returns.std() != 0 else 0 # Annualized for 1m data
    total_return = (portfolio_values[-1] / initial_cash - 1) * 100

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "total_return_pct": round(total_return, 2),
        "annualized_sharpe_ratio": round(sharpe_ratio, 2),
        "final_portfolio_value": round(portfolio_values[-1], 2)
    }

if __name__ == '__main__':
    print("Running backtest...")
    try:
        results = run_backtest('BTC/USDT', '2025-10-27 00:00:00', '2025-10-27 23:59:59')
        import json
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Could not run backtest: {e}")
