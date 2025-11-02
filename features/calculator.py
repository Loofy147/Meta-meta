"""
Event-Driven Feature Calculator

This service is a data-driven consumer that listens for new OHLCV candles on the
event bus (e.g., 'candles_1m', 'candles_5m'). Upon receiving a new candle, it
calculates a suite of technical analysis indicators (features) such as RSI,
MACD, and Bollinger Bands.

The calculated features are then persisted to the corresponding 'features' table
in the TimescaleDB database, creating a rich feature store that can be used by
the signal generation engine.
"""

import psycopg2
import pandas as pd
import pandas_ta as ta
import redis
import json
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from psycopg2.extensions import connection

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

def process_candles(candles: List[Dict[str, Any]], timeframe: str, conn: connection) -> None:
    """
    Calculates and stores technical features for a batch of candles.

    Args:
        candles (List[Dict[str, Any]]): A list of candle data dictionaries.
        timeframe (str): The timeframe of the candles (e.g., '1m', '5m').
        conn (psycopg2.extensions.connection): An active database connection.
    """
    if not candles:
        return

    df = pd.DataFrame(candles).set_index('time')

    for symbol, group in df.groupby('symbol'):
        # --- Feature Calculation using pandas_ta ---
        # Ensure data is sorted by time before calculating indicators
        group = group.sort_index()

        # Calculate RSI
        group.ta.rsi(length=14, append=True)
        # Calculate MACD
        group.ta.macd(fast=12, slow=26, signal=9, append=True)
        # Calculate Bollinger Bands
        group.ta.bbands(length=20, std=2, append=True)
        # ----------------------------------------

        # --- Database Upsert ---
        with conn.cursor() as cursor:
            for index, row in group.iterrows():
                # Only store rows where indicators have been successfully calculated
                if not pd.isna(row['RSI_14']):
                    table_name = f"features_{timeframe}"
                    cursor.execute(f"""
                        INSERT INTO {table_name} (time, symbol, rsi, macd, macds, macdh, bb_lower, bb_mid, bb_upper)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol) DO UPDATE
                        SET rsi = EXCLUDED.rsi, macd = EXCLUDED.macd, macds = EXCLUDED.macds,
                            macdh = EXCLUDED.macdh, bb_lower = EXCLUDED.bb_lower,
                            bb_mid = EXCLUDED.bb_mid, bb_upper = EXCLUDED.bb_upper;
                    """, (
                        index, symbol, row.get('RSI_14'), row.get('MACD_12_26_9'),
                        row.get('MACDs_12_26_9'), row.get('MACDh_12_26_9'),
                        row.get('BBL_20_2.0'), row.get('BBM_20_2.0'), row.get('BBU_20_2.0')
                    ))
        conn.commit()
        print(f"[{symbol}] Upserted {len(group)} features for {timeframe} timeframe.")

def run_feature_calculator() -> None:
    """
    The main loop for the feature calculator service.

    It continuously listens for new candles on all configured timeframe streams,
    processes them in batches, and stores the resulting features.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()
    timeframes = ['1m', '5m', '15m', '1h']

    # Create consumer groups for each timeframe stream
    for tf in timeframes:
        stream_name = f'candles_{tf}'
        group_name = f'feature_calculator_group_{tf}'
        try:
            redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "already exists" not in str(e):
                print(f"Error creating consumer group for {stream_name}: {e}")
                raise

    print("Starting the event-driven feature calculator...")
    while True:
        try:
            for tf in timeframes:
                stream_name = f'candles_{tf}'
                group_name = f'feature_calculator_group_{tf}'
                worker_name = f'feature_worker_{tf}_{os.getpid()}'

                events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=100, block=1000)
                if not events:
                    continue

                candles, message_ids = [], []
                for _, messages in events:
                    for message_id, message_data in messages:
                        candle = {k.decode(): v.decode() for k, v in message_data.items()}
                        candles.append({
                            'time': pd.to_datetime(candle['time']), 'symbol': candle['symbol'],
                            'open': float(candle['open']), 'high': float(candle['high']),
                            'low': float(candle['low']), 'close': float(candle['close']),
                            'volume': float(candle['volume'])
                        })
                        message_ids.append(message_id)

                process_candles(candles, tf, conn)

                if message_ids:
                    redis_client.xack(stream_name, group_name, *message_ids)

        except Exception as e:
            print(f"An error occurred in the feature calculator main loop: {e}")
            conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    run_feature_calculator()
