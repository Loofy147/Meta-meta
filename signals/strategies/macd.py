"""
MACD Crossover Trading Strategy

This module implements a classic trading strategy based on the Moving Average
Convergence Divergence (MACD) indicator. It generates a 'buy' signal on a bullish
crossover (when the MACD line crosses above the signal line) and a 'sell' signal
on a bearish crossover (when the MACD line crosses below the signal line).
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple
from psycopg2.extensions import connection

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

def generate_signal(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """
    Generates a trading signal based on the MACD crossover strategy.

    This function fetches the two most recent MACD and MACD signal line values
    from the feature store. It then checks if a crossover event has occurred
    between the last two time periods.

    Args:
        symbol (str): The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        Tuple[str, float]: A tuple containing the signal direction ('buy', 'sell',
                           or 'hold') and a confidence score (0.0 to 1.0).
                           A crossover event returns a moderate confidence of 0.6.
    """
    conn = get_db_connection()
    try:
        # Fetch the two most recent MACD feature sets for the given symbol.
        # We need two points to detect a crossover.
        query = "SELECT macd, macds FROM features_1m WHERE symbol = %s ORDER BY time DESC LIMIT 2;"
        df = pd.read_sql(query, conn, params=(symbol,))

        if len(df) < 2:
            # Not enough data to determine a crossover
            return "hold", 0.0

        latest = df.iloc[0]
        previous = df.iloc[1]

        # --- Crossover Logic ---
        is_bullish_crossover = latest['macd'] > latest['macds'] and previous['macd'] <= previous['macds']
        is_bearish_crossover = latest['macd'] < latest['macds'] and previous['macd'] >= previous['macds']
        # ---------------------

        if is_bullish_crossover:
            return "buy", 0.6  # Moderate confidence for a crossover event
        elif is_bearish_crossover:
            return "sell", 0.6 # Moderate confidence for a crossover event
        else:
            # No crossover occurred in the last interval
            return "hold", 0.0

    except Exception as e:
        print(f"An error occurred in the MACD strategy for {symbol}: {e}")
        return "hold", 0.0
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example Usage:
    # Demonstrates how to generate a signal using this strategy.
    # The result depends on the latest data in the 'features_1m' table.

    print("--- MACD Crossover Strategy Example ---")
    direction, confidence = generate_signal('BTC/USDT')
    print(f"Generated Signal for BTC/USDT: {direction.upper()} (Confidence: {confidence:.2f})")
