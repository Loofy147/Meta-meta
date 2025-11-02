"""
Event-Driven Backtester

This module provides a high-fidelity, event-driven backtesting engine. It simulates
live market conditions by replaying historical tick or candle data from the
database onto the system's event bus (Redis Streams). This allows all downstream
services (feature calculation, signal generation, aggregation, portfolio management)
to operate exactly as they would in a live environment, providing a realistic
assessment of a strategy's performance.
"""

import psycopg2
import redis
import os
import time
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Optional
from psycopg2.extensions import connection

load_dotenv()

def get_db_connection() -> connection:
    """
    Establishes a connection to the PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: A database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_redis_client() -> redis.Redis:
    """
    Returns a Redis client instance.

    Returns:
        redis.Redis: An active client connection to the Redis server.
    """
    return redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

def run_backtest(symbol: str, start_date: str, end_date: str, db_conn: Optional[connection] = None) -> None:
    """
    Runs an event-driven backtest by replaying historical data through the event bus.

    This function queries the database for historical 1-minute candle data for a
    given symbol and date range, then publishes each candle as a trade event to the
    'ingested_trades' Redis Stream, simulating the output of the data ingestion service.

    Args:
        symbol (str): The symbol to backtest (e.g., 'BTC/USDT').
        start_date (str): The start date of the backtest in 'YYYY-MM-DD' format.
        end_date (str): The end date of the backtest in 'YYYY-MM-DD' format.
        db_conn (Optional[psycopg2.extensions.connection]): An optional, existing
            database connection. If not provided, a new one will be created.
    """
    close_conn_after = False
    if db_conn is None:
        db_conn = get_db_connection()
        close_conn_after = True

    redis_client = get_redis_client()
    cursor = db_conn.cursor()

    try:
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles_1m
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC;
        """
        cursor.execute(query, (symbol, start_date, end_date))

        print(f"Starting backtest for {symbol} from {start_date} to {end_date}...")

        count = 0
        for row in cursor.fetchall():
            timestamp, open_price, high, low, close, volume = row
            # The 'ingested_trades' stream expects a format similar to live trade data.
            # We use the 'close' price of the candle as the trade price.
            trade_event = {
                'symbol': symbol,
                'price': close,
                'amount': volume,
                'timestamp': timestamp.isoformat(),
                'side': 'buy' # Placeholder, as candle data doesn't have a side.
            }
            redis_client.xadd('ingested_trades', trade_event)
            count += 1
            # A small sleep to simulate the asynchronous nature of a live feed.
            time.sleep(0.001)

        print(f"Backtest finished. Published {count} events.")
    finally:
        cursor.close()
        if close_conn_after:
            db_conn.close()

if __name__ == '__main__':
    # Example usage: Run a backtest for BTC/USDT for a specific day.
    # Ensure you have the relevant historical data in your 'candles_1m' table.
    run_backtest('BTC/USDT', '2023-01-01', '2023-01-02')
