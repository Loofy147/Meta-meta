import psycopg2
import pandas as pd
import pandas_ta as ta
import redis
import json
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

def calculate_and_store_features(redis_client):
    """
    Consumes new candles from the event bus, calculates technical indicators,
    and upserts them into the corresponding feature tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    timeframes = ['1m', '5m', '15m', '1h']

    for tf in timeframes:
        stream_name = f'candles_{tf}'
        group_name = f'feature_calculator_group_{tf}'
        try:
            redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "already exists" not in str(e):
                raise

    while True:
        try:
            for tf in timeframes:
                stream_name = f'candles_{tf}'
                group_name = f'feature_calculator_group_{tf}'

                events = redis_client.xreadgroup(group_name, f'worker_{tf}', {stream_name: '>'}, count=100, block=1000)
                if not events:
                    continue

                candles = []
                for _, messages in events:
                    for message_id, message_data in messages:
                        candle = {k.decode(): v.decode() for k, v in message_data.items()}
                        candles.append({
                            'time': pd.to_datetime(candle['time']),
                            'symbol': candle['symbol'],
                            'open': float(candle['open']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'close': float(candle['close']),
                            'volume': float(candle['volume'])
                        })

                df = pd.DataFrame(candles).set_index('time')

                if not df.empty:
                    for symbol, group in df.groupby('symbol'):
                        # Calculate features
                        group.ta.rsi(length=14, append=True)
                        group.ta.macd(append=True)
                        group.ta.bbands(append=True)

                        # Upsert features
                        for index, row in group.iterrows():
                            if not pd.isna(row['RSI_14']):
                                table_name = f"features_{tf}"
                                cursor.execute(f"""
                                    INSERT INTO {table_name} (time, symbol, rsi, macd, macds, macdh, bb_lower, bb_mid, bb_upper)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (time, symbol) DO UPDATE
                                    SET rsi = EXCLUDED.rsi, macd = EXCLUDED.macd, macds = EXCLUDED.macds, macdh = EXCLUDED.macdh,
                                        bb_lower = EXCLUDED.bb_lower, bb_mid = EXCLUDED.bb_mid, bb_upper = EXCLUDED.bb_upper;
                                """, (index, symbol, row['RSI_14'], row['MACD_12_26_9'], row['MACDs_12_26_9'], row['MACDh_12_26_9'], row['BBL_5_2.0'], row['BBM_5_2.0'], row['BBU_5_2.0']))
                        conn.commit()
                        print(f"[{symbol}] Upserted {len(group)} {tf} features.")

        except Exception as e:
            print(f"An error occurred in the feature calculator: {e}")

if __name__ == "__main__":
    print("Starting the event-driven feature calculator...")
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    calculate_and_store_features(redis_client)
