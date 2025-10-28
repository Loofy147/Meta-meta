import psycopg2
import pandas as pd
import pandas_ta as ta
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

def calculate_and_store_features():
    """
    Fetches new 1-minute candles, calculates technical indicators,
    and upserts them into the features_1m table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    while True:
        try:
            # Get the timestamp of the last processed candle
            cursor.execute("SELECT MAX(time) FROM features_1m;")
            last_processed_time = cursor.fetchone()[0]

            # Fetch candles that are newer than the last processed candle
            if last_processed_time:
                query = "SELECT time, symbol, open, high, low, close, volume FROM candles_1m WHERE time > %s;"
                df = pd.read_sql(query, conn, params=(last_processed_time,), index_col='time')
            else:
                query = "SELECT time, symbol, open, high, low, close, volume FROM candles_1m;"
                df = pd.read_sql(query, conn, index_col='time')


            if not df.empty:
                # Calculate features for each symbol
                for symbol, group in df.groupby('symbol'):
                    # RSI
                    group.ta.rsi(length=14, append=True)
                    # MACD
                    group.ta.macd(append=True)
                    # Bollinger Bands
                    group.ta.bbands(append=True)

                    # Rename columns to match the database schema
                    group.rename(columns={
                        'RSI_14': 'rsi',
                        'MACD_12_26_9': 'macd',
                        'MACDh_12_26_9': 'macdh',
                        'MACDs_12_26_9': 'macds',
                        'BBL_5_2.0': 'bb_lower',
                        'BBM_5_2.0': 'bb_mid',
                        'BBU_5_2.0': 'bb_upper',
                    }, inplace=True)

                    # Upsert data into the features_1m table
                    for index, row in group.iterrows():
                        if not pd.isna(row['rsi']): # Only insert if features are calculated
                            cursor.execute("""
                                INSERT INTO features_1m (time, symbol, rsi, macd, macds, macdh, bb_lower, bb_mid, bb_upper)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (time, symbol) DO UPDATE
                                SET rsi = EXCLUDED.rsi,
                                    macd = EXCLUDED.macd,
                                    macds = EXCLUDED.macds,
                                    macdh = EXCLUDED.macdh,
                                    bb_lower = EXCLUDED.bb_lower,
                                    bb_mid = EXCLUDED.bb_mid,
                                    bb_upper = EXCLUDED.bb_upper;
                            """, (index, symbol, row['rsi'], row['macd'], row['macds'], row['macdh'], row['bb_lower'], row['bb_mid'], row['bb_upper']))
                    conn.commit()
                    print(f"[{symbol}] Upserted features for {len(group)} candles.")

            # Wait for the next interval
            time.sleep(60) # Run every minute

        except Exception as e:
            print(f"An error occurred in the feature calculator: {e}")
            time.sleep(60) # Wait before retrying

if __name__ == "__main__":
    print("Starting the feature calculator...")
    calculate_and_store_features()
