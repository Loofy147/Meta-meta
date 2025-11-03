"""An event-driven service for calculating technical analysis features.

This service is a consumer of the candle data streams (e.g., 'candles_1m',
'candles_5m') on the Redis event bus. When a new OHLCV candle is received, it
calculates a suite of technical analysis indicators (features) using the
`pandas_ta` library. The supported indicators include RSI, MACD, and
Bollinger Bands.

The calculated features are then persisted to the corresponding features table
(e.g., 'features_1m') in the TimescaleDB database. This creates a rich,
multi-timeframe feature store that is used by the downstream signal generation
engine to make trading decisions.
"""

import psycopg2
import pandas as pd
import pandas_ta as ta
import redis
import json
import os
import sys
import time
from dotenv import load_dotenv
from typing import List, Dict, Any
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

def process_candles(candles: List[Dict[str, Any]], timeframe: str, conn: connection) -> None:
    """Calculates and stores technical features for a batch of candles.

    This function takes a list of candle data, converts it into a pandas
    DataFrame, and then uses the `pandas_ta` library to calculate technical
    indicators. It iterates through each symbol's data, calculates features,
    and then upserts the results into the appropriate TimescaleDB table.

    Note: To calculate indicators that require a lookback period (like a
    14-period RSI), this function would need to be extended to fetch historical
    data from the database to prepend to the incoming candle data. The current
    implementation calculates features only on the given batch.

    Args:
        candles: A list of candle data dictionaries.
        timeframe: The timeframe of the candles (e.g., '1m', '5m'), which
            determines the target database table.
        conn: An active psycopg2 database connection object.
    """
    if not candles:
        return

    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')


    for symbol, group in df.groupby('symbol'):
        # --- Feature Calculation using pandas_ta ---
        # For accurate TA, data must be sorted chronologically.
        group = group.sort_index()

        # Calculate a suite of technical indicators and append them as columns.
        group.ta.rsi(length=14, append=True)
        group.ta.macd(fast=12, slow=26, signal=9, append=True)
        group.ta.bbands(length=20, std=2, append=True)
        # ----------------------------------------

        # --- Database Upsert ---
        with conn.cursor() as cursor:
            # Iterate through the DataFrame rows to insert/update feature data.
            for index, row in group.iterrows():
                # Only store rows where indicators have been successfully calculated.
                # RSI is a good proxy for this, as it's one of the first to be calculated.
                if not pd.isna(row.get('RSI_14')):
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
        print(f"[{symbol}] Upserted features for {len(group)} candles in {timeframe} timeframe.")

def run_feature_calculator() -> None:
    """The main entry point and infinite loop for the feature calculator.

    This function establishes connections to Redis and PostgreSQL. It then
    creates the necessary Redis consumer groups for each candle stream timeframe.
    It enters an infinite loop, polling each stream for new candle events,
    processing them in batches, and acknowledging them upon success.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()
    timeframes = ['1m', '5m', '15m', '1h']

    # Create a consumer group for each timeframe stream we need to process.
    for tf in timeframes:
        stream_name = f'candles_{tf}'
        group_name = f'feature_calculator_group_{tf}'
        try:
            redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "already exists" not in str(e).lower():
                print(f"Error creating consumer group for {stream_name}: {e}")
                raise

    print("Starting the event-driven feature calculator...")
    while True:
        try:
            for tf in timeframes:
                stream_name = f'candles_{tf}'
                group_name = f'feature_calculator_group_{tf}'
                worker_name = f'feature_worker_{tf}_{os.getpid()}'

                # Read a batch of messages from one of the timeframe streams.
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

                if candles:
                    process_candles(candles, tf, conn)

                if message_ids:
                    redis_client.xack(stream_name, group_name, *message_ids)

        except Exception as e:
            print(f"An error occurred in the feature calculator main loop: {e}")
            if conn:
                conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    run_feature_calculator()
