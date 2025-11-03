"""Implements a MACD crossover trading strategy.

This module provides a signal generation function based on the classic Moving
Average Convergence Divergence (MACD) indicator. The strategy identifies
trading opportunities by detecting crossovers between the MACD line and its
signal line.

- A **bullish crossover** (MACD line crosses above the signal line) generates a 'buy' signal.
- A **bearish crossover** (MACD line crosses below the signal line) generates a 'sell' signal.

The function queries the database for the two most recent feature sets to
compare their MACD and signal line values.
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple
from psycopg2.extensions import connection

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

def generate_signal(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """Generates a trading signal based on a MACD crossover event.

    This function fetches the two most recent MACD and MACD signal line values
    (`macd` and `macds` columns) from the `features_1m` table for the specified
    symbol. It then compares the state of the latest values against the previous
    values to detect if a crossover has just occurred.

    - If the MACD line was below the signal line previously and is now above it,
      a 'buy' signal is generated.
    - If the MACD line was above the signal line previously and is now below it,
      a 'sell' signal is generated.

    Args:
        symbol: The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        A tuple containing the signal direction ('buy', 'sell', or 'hold')
        and a confidence score. A crossover event returns a moderate
        confidence of 0.6, while no event returns 'hold' with 0.0 confidence.
    """
    conn = get_db_connection()
    try:
        # Fetch the two most recent feature sets to detect a crossover event.
        query = """
            SELECT macd, macds FROM features_1m
            WHERE symbol = %s AND macd IS NOT NULL AND macds IS NOT NULL
            ORDER BY time DESC LIMIT 2;
        """
        df = pd.read_sql(query, conn, params=(symbol,))

        if len(df) < 2:
            # Not enough data to determine a crossover.
            return "hold", 0.0

        latest = df.iloc[0]
        previous = df.iloc[1]

        # --- Crossover Detection Logic ---
        # Bullish: Was below or equal, now is above.
        is_bullish_crossover = previous['macd'] <= previous['macds'] and latest['macd'] > latest['macds']
        # Bearish: Was above or equal, now is below.
        is_bearish_crossover = previous['macd'] >= previous['macds'] and latest['macd'] < latest['macds']
        # ---------------------------------

        if is_bullish_crossover:
            return "buy", 0.6  # Moderate confidence for a crossover.
        elif is_bearish_crossover:
            return "sell", 0.6 # Moderate confidence for a crossover.
        else:
            # No crossover occurred in the latest interval.
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
