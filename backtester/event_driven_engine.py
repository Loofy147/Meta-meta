"""Provides a high-fidelity, event-driven backtesting engine.

This module simulates live market conditions by replaying historical data through
the system's event bus. It meticulously queries historical candle data from the
database in chronological order and publishes each candle as a new event onto a
Redis Stream.

This approach allows all downstream services—such as feature calculation,
signal generation, aggregation, and portfolio management—to operate exactly as
they would in a live trading environment. This provides a highly realistic
assessment of a strategy's performance, capturing the sequential and asynchronous
nature of the live system.
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

def get_redis_client() -> redis.Redis:
    """Establishes and returns a connection to the Redis server.

    Returns:
        An active Redis client instance.
    """
    return redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

def run_backtest(symbol: str, start_date: str, end_date: str) -> None:
    """Runs an event-driven backtest by replaying historical data.

    This function serves as the main entry point for the backtester. It queries
    the database for all 1-minute candles for a given symbol and date range.
    It then iterates through each historical candle and publishes its data to
    the `raw_trades` Redis Stream, effectively simulating the real-time data
    ingestion process.

    Args:
        symbol: The symbol to backtest (e.g., 'BTC/USDT').
        start_date: The start date of the backtest in 'YYYY-MM-DD' format.
        end_date: The end date of the backtest in 'YYYY-MM-DD' format.
    """
    db_conn = None
    try:
        db_conn = get_db_connection()
        redis_client = get_redis_client()

        with db_conn.cursor() as cursor:
            # Query for historical 1-minute candle data in chronological order.
            query = """
                SELECT time, open, high, low, close, volume
                FROM candles_1m
                WHERE symbol = %s AND time >= %s AND time < %s
                ORDER BY time ASC;
            """
            cursor.execute(query, (symbol, start_date, end_date))

            print(f"Starting event-driven backtest for {symbol} from {start_date} to {end_date}...")

            record_count = 0
            for row in cursor:
                timestamp, open_price, high, low, close, volume = row
                # To simulate the `raw_trades` stream, we format the candle data
                # as a pseudo-trade event. The `close` price is used as the trade price.
                trade_event = {
                    'symbol': symbol,
                    'price': close,
                    'amount': volume,
                    # Timestamp is converted to milliseconds for consistency.
                    'timestamp': int(timestamp.timestamp() * 1000),
                    # Side is a placeholder as candles are direction-agnostic.
                    'side': 'buy'
                }
                redis_client.xadd('raw_trades', trade_event)
                record_count += 1
                # A minimal sleep prevents overwhelming the CPU and allows other
                # services to process events in a more realistic sequence.
                time.sleep(0.001)

            print(f"Backtest finished. Published {record_count} historical events.")

    except psycopg2.Error as e:
        print(f"Database error during backtest: {e}")
    except redis.RedisError as e:
        print(f"Redis error during backtest: {e}")
    finally:
        if db_conn:
            db_conn.close()

if __name__ == '__main__':
    # Example usage: Run a backtest for BTC/USDT for a specific day.
    # Ensure you have the relevant historical data in your 'candles_1m' table.
    run_backtest('BTC/USDT', '2023-01-01', '2023-01-02')
