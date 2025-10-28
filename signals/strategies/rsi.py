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
    Generates a trading signal based on the Relative Strength Index (RSI).
    - Buy when RSI is oversold (< 30)
    - Sell when RSI is overbought (> 70)
    """
    conn = get_db_connection()
    try:
        # Fetch the latest RSI value from the feature store
        query = "SELECT rsi FROM features_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
        df = pd.read_sql(query, conn, params=(symbol,))

        if df.empty:
            return "hold", 0.0

        rsi = df['rsi'].iloc[0]
        if rsi > 70:
            return "sell", 0.7 # High confidence for overbought
        elif rsi < 30:
            return "buy", 0.7 # High confidence for oversold
        else:
            return "hold", 0.0

    except Exception as e:
        print(f"An error occurred in the RSI strategy: {e}")
        return "hold", 0.0
    finally:
        if conn is not None:
            conn.close()
