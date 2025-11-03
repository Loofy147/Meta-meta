"""An event-driven service to resample raw trades into OHLCV candles.

This service acts as a consumer of the 'raw_trades' Redis Stream. It reads
batches of raw trade data, uses the pandas library to resample them into
multiple timeframes (e.g., 1m, 5m, 1h), and persists the resulting OHLCV
(Open, High, Low, Close, Volume) candles into their respective TimescaleDB
hypertables.

This is a critical component of the data processing pipeline, transforming a
high-frequency, unstructured stream of trades into a structured, multi-timeframe
format that is essential for downstream services like feature calculation and
signal generation. After storing a candle, it also publishes that candle to a
dedicated stream (e.g., 'candles_1m') for other services to consume.
"""

import psycopg2
import pandas as pd
import redis
import json
import os
import sys
import time
from dotenv import load_dotenv
from typing import List, Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

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

def process_trades(trades: List[Dict[str, Any]], conn: connection) -> None:
    """Resamples a batch of trades and upserts the resulting candles.

    This function takes a list of raw trades, converts them into a pandas
    DataFrame, and then performs a time-based resampling for a predefined set
    of timeframes. For each symbol and timeframe, it calculates the OHLCV
    values.

    The resulting candles are then "upserted" into the corresponding
    TimescaleDB table using an 'ON CONFLICT' clause to ensure data integrity.
    Finally, each new candle is published to a dedicated Redis Stream.

    Args:
        trades: A list of trade data dictionaries, where each dictionary
            represents a single trade.
        conn: An active psycopg2 database connection object.
    """
    if not trades:
        return

    publisher = EventPublisher()
    timeframes = ['1m', '5m', '15m', '1h']

    # Convert list of trades to a DataFrame for efficient processing
    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')


    for symbol, group in df.groupby('symbol'):
        for tf in timeframes:
            # Use pandas to resample trade data into OHLCV candles
            ohlcv = group['price'].resample(tf).ohlc()
            ohlcv['volume'] = group['amount'].resample(tf).sum()
            ohlcv.dropna(inplace=True)  # Remove timeframes with no trades

            if ohlcv.empty:
                continue

            with conn.cursor() as cursor:
                for index, row in ohlcv.iterrows():
                    table_name = f"candles_{tf}"
                    # Use ON CONFLICT for an idempotent upsert operation
                    cursor.execute(f"""
                        INSERT INTO {table_name} (time, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol) DO UPDATE
                        SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                            close = EXCLUDED.close, volume = EXCLUDED.volume;
                    """, (index, symbol, row['open'], row['high'], row['low'], row['close'], row['volume']))

                    # Publish the newly formed candle for downstream consumers
                    publisher.publish(f"candles_{tf}", {
                        'time': index.isoformat(), 'symbol': symbol, 'open': row['open'],
                        'high': row['high'], 'low': row['low'], 'close': row['close'],
                        'volume': row['volume']
                    })
            conn.commit()
            print(f"[{symbol}] Upserted and published {len(ohlcv)} {tf} candles.")

def run_resampler() -> None:
    """The main entry point and infinite loop for the resampler service.

    This function establishes connections to Redis and PostgreSQL. It then enters
    an infinite loop to continuously consume trades from the 'raw_trades' stream
    using a consumer group.

    It processes events in batches for efficiency and acknowledges them with
    Redis upon successful insertion into the database, ensuring reliable,
    at-least-once message processing. Basic error handling is included to
    manage database connection issues.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()

    stream_name = 'raw_trades'
    group_name = 'resampler_group'
    worker_name = f'resampler_worker_{os.getpid()}'

    try:
        # Create the consumer group; ignores error if it already exists
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e).lower():
            print(f"Error creating consumer group: {e}")
            raise

    print("Starting the event-driven data resampler...")
    while True:
        try:
            # Block and wait for new messages in the stream
            events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=100, block=5000)
            if not events:
                continue

            trades = []
            message_ids = []
            for _, messages in events:
                for message_id, message_data in messages:
                    # Decode message data from bytes to string
                    trade = {k.decode(): v.decode() for k, v in message_data.items()}
                    trades.append({
                        'time': pd.to_datetime(int(trade['timestamp']), unit='ms'),
                        'symbol': trade['symbol'],
                        'price': float(trade['price']),
                        'amount': float(trade['amount'])
                    })
                    message_ids.append(message_id)

            if trades:
                process_trades(trades, conn)

            # Acknowledge messages only after successful processing
            if message_ids:
                redis_client.xack(stream_name, group_name, *message_ids)

        except Exception as e:
            print(f"An error occurred in the resampler main loop: {e}")
            # On error, rollback DB transaction and wait before retrying
            if conn:
                conn.rollback()
            time.sleep(10)


if __name__ == "__main__":
    run_resampler()
