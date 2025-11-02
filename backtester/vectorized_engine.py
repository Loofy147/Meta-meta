"""
Vectorized Backtester

This module provides a fast, vectorized backtesting engine. It is designed for
quick, high-level analysis of trading strategies that can be expressed as
Pandas/NumPy array operations. While less realistic than an event-driven
backtester, it is highly efficient for iterating on strategy ideas and tuning
parameters.

The engine includes realistic cost simulations for commissions and slippage.
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

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_db_connection() -> connection:
    """
    Establishes and returns a connection to the PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: A database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def run_backtest(symbol: str, start_date: str, end_date: str, initial_cash: float = 100000.0, commission_pct: float = 0.001, slippage_pct: float = 0.0005) -> Dict[str, Any]:
    """
    Runs a vectorized backtest for a simple MACD crossover strategy.

    This function simulates the strategy's performance, including the impact of
    transaction costs (commission and slippage).

    Args:
        symbol (str): The trading symbol to backtest (e.g., 'BTC/USDT').
        start_date (str): The start date for the backtest ('YYYY-MM-DD').
        end_date (str): The end date for the backtest ('YYYY-MM-DD').
        initial_cash (float): The starting cash balance for the portfolio.
        commission_pct (float): The percentage commission per trade.
        slippage_pct (float): The percentage slippage per trade.

    Returns:
        Dict[str, Any]: A dictionary containing key performance indicators (KPIs)
                        of the backtest, such as total return and Sharpe ratio.
    """
    conn = get_db_connection()
    try:
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

    # --- Strategy Logic: MACD Crossover ---
    df['signal'] = 0
    df.loc[df['macd'] > df['macds'], 'signal'] = 1  # Buy signal
    df.loc[df['macd'] < df['macds'], 'signal'] = -1 # Sell signal
    # ------------------------------------

    # --- Portfolio Simulation ---
    cash = initial_cash
    position = 0.0
    portfolio_values = []

    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]

        # Apply slippage to execution prices
        buy_price = current_close * (1 + slippage_pct)
        sell_price = current_close * (1 - slippage_pct)

        # --- Trade Execution Logic ---
        # Buy signal and we have cash
        if df['signal'].iloc[i] == 1 and cash > 0:
            position = (cash / buy_price) * (1 - commission_pct)
            cash = 0
        # Sell signal and we have a position
        elif df['signal'].iloc[i] == -1 and position > 0:
            cash = (position * sell_price) * (1 - commission_pct)
            position = 0

        portfolio_values.append(cash + position * current_close)

    if not portfolio_values:
        return {"error": "No trades were executed."}
    # ---------------------------

    # --- KPI Calculation ---
    returns = pd.Series(portfolio_values, index=df.index[1:]).pct_change().dropna()
    # Annualize Sharpe Ratio (assuming 1-minute data)
    annualization_factor = np.sqrt(252 * 24 * 60)
    sharpe_ratio = annualization_factor * returns.mean() / returns.std() if returns.std() != 0 else 0
    total_return = (portfolio_values[-1] / initial_cash - 1) * 100
    # ---------------------

    return {
        "strategy": "MACD Crossover",
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
