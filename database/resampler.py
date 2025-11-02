"""
Event-Driven Data Resampler

This service consumes raw trade data from the 'raw_trades' Redis Stream,
resamples it into multiple timeframes (e.g., 1m, 5m, 1h), and persists the
resulting OHLCV (Open, High, Low, Close, Volume) candles into the TimescaleDB
database.

It acts as a critical component in the data processing pipeline, transforming a
high-frequency stream of raw trades into a structured, multi-timeframe format
that is essential for feature calculation and signal generation.
"""

import psycopg2
import pandas as pd
import redis
import json
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

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

def process_trades(trades: List[Dict[str, Any]], conn: connection) -> None:
    """
    Resamples a batch of trades into multiple timeframes and stores them.

    Args:
        trades (List[Dict[str, Any]]): A list of trade data dictionaries.
        conn (psycopg2.extensions.connection): An active database connection.
    """
    if not trades:
        return

    publisher = EventPublisher()
    timeframes = ['1m', '5m', '15m', '1h']

    df = pd.DataFrame(trades).set_index('time')

    for symbol, group in df.groupby('symbol'):
        for tf in timeframes:
            # Resample trade data to OHLCV candles
            ohlcv = group['price'].resample(tf).ohlc()
            ohlcv['volume'] = group['amount'].resample(tf).sum()
            ohlcv.dropna(inplace=True) # Remove timeframes with no trades

            if ohlcv.empty:
                continue

            with conn.cursor() as cursor:
                for index, row in ohlcv.iterrows():
                    table_name = f"candles_{tf}"
                    # Use ON CONFLICT to update existing candles, creating a robust upsert
                    cursor.execute(f"""
                        INSERT INTO {table_name} (time, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol) DO UPDATE
                        SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                            close = EXCLUDED.close, volume = EXCLUDED.volume;
                    """, (index, symbol, row['open'], row['high'], row['low'], row['close'], row['volume']))

                    # Publish the newly formed candle to its own stream for downstream consumers
                    publisher.publish(f"candles_{tf}", {
                        'time': index.isoformat(), 'symbol': symbol, 'open': row['open'],
                        'high': row['high'], 'low': row['low'], 'close': row['close'],
                        'volume': row['volume']
                    })
            conn.commit()
            print(f"[{symbol}] Upserted and published {len(ohlcv)} {tf} candles.")

def run_resampler() -> None:
    """
    The main loop for the resampler service.

    It continuously consumes trades from Redis, processes them in batches, and
    stores the resulting candles.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()

    stream_name = 'raw_trades'
    group_name = 'resampler_group'
    worker_name = f'resampler_worker_{os.getpid()}'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            print(f"Error creating consumer group: {e}")
            raise

    print("Starting the event-driven data resampler...")
    while True:
        try:
            # Read a batch of messages from the stream
            events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=100, block=5000)
            if not events:
                continue

            trades = []
            message_ids = []
            for _, messages in events:
                for message_id, message_data in messages:
                    trade = {k.decode(): v.decode() for k, v in message_data.items()}
                    trades.append({
                        'time': pd.to_datetime(int(trade['timestamp']), unit='ms'),
                        'symbol': trade['symbol'],
                        'price': float(trade['price']),
                        'amount': float(trade['amount'])
                    })
                    message_ids.append(message_id)

            process_trades(trades, conn)

            # Acknowledge messages after successful processing
            if message_ids:
                redis_client.xack(stream_name, group_name, *message_ids)

        except Exception as e:
            print(f"An error occurred in the resampler main loop: {e}")
            # In case of a DB error, rollback and wait before retrying
            conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    run_resampler()
