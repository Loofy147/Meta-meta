"""Provides a fast, vectorized backtesting engine for strategy analysis.

This module is designed for quick, high-level analysis of trading strategies
that can be expressed using Pandas and NumPy array operations. Vectorized
backtesters are highly efficient for iterating on strategy ideas and tuning
parameters because they avoid explicit, slow, loop-based event simulation.

While less realistic than a full event-driven backtester (as it cannot easily
model order queue dynamics or complex path-dependent logic), it provides a
powerful tool for initial research. This implementation includes realistic cost
simulations for trade commissions and price slippage.
"""

import pandas as pd
import numpy as np
import psycopg2
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_db_connection() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    Uses credentials from environment variables (DB_HOST, DB_NAME, etc.).

    Returns:
        A psycopg2 database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def run_backtest(symbol: str, start_date: str, end_date: str, initial_cash: float = 100000.0, commission_pct: float = 0.001, slippage_pct: float = 0.0005) -> Dict[str, Any]:
    """Runs a vectorized backtest for a simple MACD crossover strategy.

    This function performs the entire backtest workflow:
    1.  Fetches historical feature and price data from the database.
    2.  Generates buy/sell signals in a vectorized manner using Pandas.
    3.  Iterates through the timeline, simulating portfolio changes based on
        the signals and applying transaction costs.
    4.  Calculates key performance indicators (KPIs) like total return and
        the Sharpe ratio.

    Args:
        symbol: The trading symbol to backtest (e.g., 'BTC/USDT').
        start_date: The start date for the backtest in 'YYYY-MM-DD' format.
        end_date: The end date for the backtest in 'YYYY-MM-DD' format.
        initial_cash: The starting cash balance for the simulation.
        commission_pct: The percentage commission charged per trade.
        slippage_pct: The estimated percentage of adverse price movement
            (slippage) for each trade.

    Returns:
        A dictionary containing the key performance indicators (KPIs) of the
        backtest, such as total return, Sharpe ratio, and final portfolio
        value.
    """
    conn = get_db_connection()
    try:
        # Fetch all required historical data in a single query.
        query = """
            SELECT f.time, f.rsi, f.macd, f.macds, c.close
            FROM features_1m f JOIN candles_1m c ON f.time = c.time AND f.symbol = c.symbol
            WHERE f.symbol = %s AND f.time BETWEEN %s AND %s ORDER BY f.time;
        """
        df = pd.read_sql(query, conn, params=(symbol, start_date, end_date), index_col='time')
    finally:
        conn.close()

    if df.empty:
        return {"error": "No data found for the given symbol and date range."}

    # --- Vectorized Strategy Logic: MACD Crossover ---
    # Create a 'signal' column based on the relationship between MACD and its signal line.
    df['signal'] = 0
    df.loc[df['macd'] > df['macds'], 'signal'] = 1   # Bullish signal
    df.loc[df['macd'] < df['macds'], 'signal'] = -1  # Bearish signal
    # ---------------------------------------------

    # --- Iterative Portfolio Simulation ---
    # Although the signals are generated vectorially, the portfolio simulation
    # must be iterative to handle path-dependent state (cash and position).
    cash = initial_cash
    position = 0.0
    portfolio_values = [initial_cash]

    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]

        # Simulate realistic execution prices by applying slippage.
        buy_price = current_close * (1 + slippage_pct)
        sell_price = current_close * (1 - slippage_pct)

        # --- Trade Execution Simulation ---
        # On a buy signal, if we are currently in cash, go all-in.
        if df['signal'].iloc[i] == 1 and cash > 0:
            position = (cash / buy_price) * (1 - commission_pct)
            cash = 0
        # On a sell signal, if we have a position, sell it all.
        elif df['signal'].iloc[i] == -1 and position > 0:
            cash = (position * sell_price) * (1 - commission_pct)
            position = 0

        portfolio_values.append(cash + position * current_close)

    if len(portfolio_values) <= 1:
        return {"error": "No trades were executed during the backtest period."}
    # ----------------------------------

    # --- KPI Calculation ---
    returns = pd.Series(portfolio_values, index=df.index).pct_change().dropna()
    # Annualize the Sharpe Ratio (assuming 1-minute data).
    # (252 trading days * 24 hours * 60 minutes)
    annualization_factor = np.sqrt(252 * 24 * 60)
    sharpe_ratio = (annualization_factor * returns.mean() / returns.std()) if returns.std() != 0 else 0
    total_return = (portfolio_values[-1] / initial_cash - 1) * 100
    # ---------------------

    return {
        "strategy": "Vectorized MACD Crossover",
        "symbol": symbol,
        "total_return_pct": round(total_return, 2),
        "annualized_sharpe_ratio": round(sharpe_ratio, 2),
        "final_portfolio_value": round(portfolio_values[-1], 2),
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct
    }

if __name__ == '__main__':
    print("Running vectorized backtest with cost simulation...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    try:
        results = run_backtest('BTC/USDT', start_date.isoformat(), end_date.isoformat())
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Could not run backtest: {e}")
