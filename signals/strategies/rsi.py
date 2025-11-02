"""
RSI Overbought/Oversold Trading Strategy

This module implements a classic mean-reversion strategy based on the Relative
Strength Index (RSI). It generates a 'buy' signal when the RSI indicates that an
asset is oversold (typically below 30) and a 'sell' signal when it indicates the
asset is overbought (typically above 70).
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.manager import get_config

load_dotenv()

# --- Strategy Parameters ---
# These could be moved to the database configuration for dynamic tuning.
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70
# -------------------------

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
    Generates a trading signal based on the RSI indicator.

    This function fetches the most recent RSI value from the feature store and
    compares it against the overbought and oversold thresholds.

    Args:
        symbol (str): The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        Tuple[str, float]: A tuple containing the signal direction ('buy', 'sell',
                           or 'hold') and a confidence score (0.0 to 1.0).
                           A breach of the thresholds returns a high confidence of 0.7.
    """
    conn = get_db_connection()
    try:
        # Load dynamic thresholds from the central configuration
        strategy_config = get_config().get('strategies', {}).get('rsi', {})
        oversold = strategy_config.get('oversold_threshold', OVERSOLD_THRESHOLD)
        overbought = strategy_config.get('overbought_threshold', OVERBOUGHT_THRESHOLD)

        # Fetch the most recent RSI value for the given symbol.
        query = "SELECT rsi FROM features_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
        df = pd.read_sql(query, conn, params=(symbol,))

        if df.empty or pd.isna(df['rsi'].iloc[0]):
            # Not enough data or RSI is not calculated yet
            return "hold", 0.0

        rsi_value = df['rsi'].iloc[0]

        if rsi_value > overbought:
            return "sell", 0.7  # High confidence for a clear overbought signal
        elif rsi_value < oversold:
            return "buy", 0.7   # High confidence for a clear oversold signal
        else:
            # RSI is in the neutral zone
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
