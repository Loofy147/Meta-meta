import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def generate_signal(symbol='BTC/USDT'):
    """
    Generates a trading signal based on the MACD crossover.
    - Buy when MACD crosses above the signal line (MACDh > 0)
    - Sell when MACD crosses below the signal line (MACDh < 0)
    """
    conn = get_db_connection()
    try:
        # Fetch the two latest MACD values from the feature store
        query = "SELECT macd, macds FROM features_1m WHERE symbol = %s ORDER BY time DESC LIMIT 2;"
        df = pd.read_sql(query, conn, params=(symbol,))

        if len(df) < 2:
            return "hold", 0.0

        latest = df.iloc[0]
        previous = df.iloc[1]

        # Check for a bullish crossover
        if latest['macd'] > latest['macds'] and previous['macd'] <= previous['macds']:
            return "buy", 0.6 # Moderate confidence for crossover
        # Check for a bearish crossover
        elif latest['macd'] < latest['macds'] and previous['macd'] >= previous['macds']:
            return "sell", 0.6 # Moderate confidence for crossover
        else:
            return "hold", 0.0

    except Exception as e:
        print(f"An error occurred in the MACD strategy: {e}")
        return "hold", 0.0
    finally:
        if conn is not None:
            conn.close()
