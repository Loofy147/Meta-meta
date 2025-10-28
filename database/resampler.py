import psycopg2
import pandas as pd
import time
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

def resample_and_store_candles():
    """
    Fetches new raw trades, resamples them into 1-minute candles,
    and upserts them into the candles_1m table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    while True:
        try:
            # Get the timestamp of the last processed trade
            cursor.execute("SELECT MAX(time) FROM candles_1m;")
            last_processed_time = cursor.fetchone()[0]

            # Fetch trades that are newer than the last processed trade
            if last_processed_time:
                query = "SELECT time, symbol, price, amount FROM trades WHERE time > %s;"
                df = pd.read_sql(query, conn, params=(last_processed_time,), index_col='time')
            else:
                query = "SELECT time, symbol, price, amount FROM trades;"
                df = pd.read_sql(query, conn, index_col='time')

            if not df.empty:
                # Resample to 1-minute candles for each symbol
                for symbol, group in df.groupby('symbol'):
                    ohlcv = group['price'].resample('1T').ohlc()
                    ohlcv['volume'] = group['amount'].resample('1T').sum()
                    ohlcv['symbol'] = symbol

                    # Upsert data into the candles_1m table
                    for index, row in ohlcv.iterrows():
                        cursor.execute("""
                            INSERT INTO candles_1m (time, symbol, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (time, symbol) DO UPDATE
                            SET open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume;
                        """, (index, row['symbol'], row['open'], row['high'], row['low'], row['close'], row['volume']))
                    conn.commit()
                    print(f"[{symbol}] Upserted {len(ohlcv)} 1-minute candles.")

            # Wait for the next interval
            time.sleep(60) # Run every minute

        except Exception as e:
            print(f"An error occurred in the resampler: {e}")
            time.sleep(60) # Wait before retrying

if __name__ == "__main__":
    print("Starting the data resampler...")
    resample_and_store_candles()
