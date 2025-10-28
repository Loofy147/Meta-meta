import psycopg2
import pandas as pd
import redis
import json
import os
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def resample_and_store_candles(redis_client):
    """
    Consumes raw trades from the event bus, resamples them into multiple timeframes,
    upserts them into the corresponding candle tables, and publishes them to the event bus.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    timeframes = ['1m', '5m', '15m', '1h']
    publisher = EventPublisher()

    stream_name = 'raw_trades'
    group_name = 'resampler_group'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    while True:
        try:
            events = redis_client.xreadgroup(group_name, 'resampler_worker', {stream_name: '>'}, count=100, block=5000)
            if not events:
                continue

            trades = []
            for _, messages in events:
                for message_id, message_data in messages:
                    trade = {k.decode(): v.decode() for k, v in message_data.items()}
                    trades.append({
                        'time': pd.to_datetime(int(trade['timestamp']), unit='ms'),
                        'symbol': trade['symbol'],
                        'price': float(trade['price']),
                        'amount': float(trade['amount'])
                    })

            df = pd.DataFrame(trades).set_index('time')

            if not df.empty:
                for symbol, group in df.groupby('symbol'):
                    for tf in timeframes:
                        ohlcv = group['price'].resample(tf).ohlc()
                        ohlcv['volume'] = group['amount'].resample(tf).sum()

                        for index, row in ohlcv.iterrows():
                            if not pd.isna(row['open']):
                                table_name = f"candles_{tf}"
                                cursor.execute(f"""
                                    INSERT INTO {table_name} (time, symbol, open, high, low, close, volume)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (time, symbol) DO UPDATE
                                    SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                                        close = EXCLUDED.close, volume = EXCLUDED.volume;
                                """, (index, symbol, row['open'], row['high'], row['low'], row['close'], row['volume']))

                                # Publish the new candle to a stream
                                publisher.publish(f"candles_{tf}", {
                                    'time': index.isoformat(), 'symbol': symbol, 'open': row['open'],
                                    'high': row['high'], 'low': row['low'], 'close': row['close'],
                                    'volume': row['volume']
                                })
                        conn.commit()
                        print(f"[{symbol}] Upserted and published {len(ohlcv)} {tf} candles.")

        except Exception as e:
            print(f"An error occurred in the resampler: {e}")

if __name__ == "__main__":
    print("Starting the event-driven data resampler...")
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    resample_and_store_candles(redis_client)
