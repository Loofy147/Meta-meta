"""Implements an RSI overbought/oversold trading strategy.

This module provides a signal generation function based on the classic
mean-reversion interpretation of the Relative Strength Index (RSI). The
strategy operates on the following principles:

- When the RSI crosses below a lower threshold (e.g., 30), the asset is
  considered **oversold**, generating a 'buy' signal.
- When the RSI crosses above an upper threshold (e.g., 70), the asset is
  considered **overbought**, generating a 'sell' signal.

The thresholds for this strategy are dynamically loaded from the central
database configuration, allowing for on-the-fly tuning without code changes.
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.manager import get_config

load_dotenv()

# --- Default Strategy Parameters ---
# These are used as fallbacks if the configuration is not found in the database.
DEFAULT_OVERSOLD_THRESHOLD = 30
DEFAULT_OVERBOUGHT_THRESHOLD = 70
# ---------------------------------

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
    """Generates a trading signal based on the latest RSI value.

    This function fetches the most recent RSI value from the `features_1m` table
    for the specified symbol. It then compares this value against the overbought
    and oversold thresholds, which are retrieved from the dynamic system
    configuration.

    Args:
        symbol: The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        A tuple containing the signal direction ('buy', 'sell', or 'hold')
        and a confidence score. A breach of the thresholds returns a high
        confidence of 0.7, while a neutral RSI returns 'hold' with 0.0 confidence.
    """
    conn = get_db_connection()
    try:
        # Load dynamic thresholds from the central configuration, with fallbacks.
        strategy_config = get_config().get('strategies', {}).get('rsi', {})
        oversold = strategy_config.get('oversold_threshold', DEFAULT_OVERSOLD_THRESHOLD)
        overbought = strategy_config.get('overbought_threshold', DEFAULT_OVERBOUGHT_THRESHOLD)

        # Fetch the most recent RSI value.
        query = """
            SELECT rsi FROM features_1m
            WHERE symbol = %s AND rsi IS NOT NULL
            ORDER BY time DESC LIMIT 1;
        """
        df = pd.read_sql(query, conn, params=(symbol,))

        if df.empty:
            # Not enough data or RSI has not been calculated yet.
            return "hold", 0.0

        rsi_value = df['rsi'].iloc[0]

        if rsi_value > overbought:
            return "sell", 0.7  # High confidence for a clear overbought signal.
        elif rsi_value < oversold:
            return "buy", 0.7   # High confidence for a clear oversold signal.
        else:
            # RSI is in the neutral zone between the thresholds.
            return "hold", 0.0

    except Exception as e:
        print(f"An error occurred in the RSI strategy for {symbol}: {e}")
        return "hold", 0.0
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example Usage:
    # Demonstrates how to generate a signal using this strategy.
    # The result depends on the latest RSI value in the 'features_1m' table.

    print("--- RSI Strategy Example ---")
    direction, confidence = generate_signal('BTC/USDT')
    print(f"Generated Signal for BTC/USDT: {direction.upper()} (Confidence: {confidence:.2f})")
