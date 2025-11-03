"""A high-fidelity backtester that replays historical trade data.

This module provides an alternative event-driven backtesting engine that uses
raw, historical trade data (as opposed to candle data) for the highest fidelity
simulation. It queries a historical `trades` table from the database and
publishes each individual trade, in chronological order, onto the `raw_trades`
Redis Stream.

This method is powerful because it allows the entire downstream system—starting
with the `DataResampler`—to be tested with a realistic, high-frequency data
flow, exactly as it would operate in a live environment. Running this backtester
requires all other system services to be active and listening to the event bus.
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from typing import Optional

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection():
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

def run_event_driven_backtest(symbol: str, start_date: str, end_date: str):
    """Runs a backtest by replaying historical raw trades through the event bus.

    This function fetches all historical trades for a given symbol and date
    range from the `trades` table. It then iterates through each trade and
    publishes it as a message to the `raw_trades` Redis Stream. This allows
    the entire system to process the historical data as if it were happening
    in real-time.

    Args:
        symbol: The symbol to backtest (e.g., 'BTC/USDT').
        start_date: The start date for the backtest in 'YYYY-MM-DD' format.
        end_date: The end date for the backtest in 'YYYY-MM-DD' format.
    """
    conn = None
    try:
        conn = get_db_connection()
        publisher = EventPublisher()

        # Fetch all historical trade data within the specified date range.
        query = """
            SELECT time, symbol, price, amount, side
            FROM raw_trades
            WHERE symbol = %s AND time BETWEEN %s AND %s
            ORDER BY time ASC;
        """
        df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))

        if df.empty:
            print("No historical trade data found for the given symbol and date range.")
            return

        print(f"Replaying {len(df)} historical trades for {symbol}...")

        # Replay each trade through the event bus.
        for _, row in df.iterrows():
            event_data = {
                'symbol': row['symbol'],
                'price': row['price'],
                'amount': row['amount'],
                'side': row['side'],
                # Ensure timestamp is in milliseconds format, as expected by consumers.
                'timestamp': int(pd.to_datetime(row['time']).timestamp() * 1000)
            }
            publisher.publish('raw_trades', event_data)

        print("Finished replaying all historical trade data.")
        print("\nNote: Allow running services (resampler, features, etc.) time to process the data.")
        print("After a few moments, you can query the database to see the final portfolio state.")

    except (psycopg2.Error, pd.io.sql.DatabaseError) as e:
        print(f"A database error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # This assumes that all the other services are running and listening to the event bus.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    run_event_driven_backtest('BTC/USDT', start_date.isoformat(), end_date.isoformat())
