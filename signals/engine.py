import pandas as pd
import pandas_ta as ta
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_trade_data():
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        # Fetch the last 1000 trades for BTC/USDT
        query = "SELECT time, price FROM trades WHERE symbol = 'BTC/USDT' ORDER BY time DESC LIMIT 1000;"
        df = pd.read_sql(query, conn, index_col='time')
        return df.iloc[::-1]  # Reverse to have the latest data at the end
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()

def generate_signal():
    df = get_trade_data()
    if len(df) < 50: # Not enough data for the long MA
        return "hold"

    # Calculate short and long-term moving averages, explicitly using the 'price' column
    df.ta.sma(close=df['price'], length=20, append=True)
    df.ta.sma(close=df['price'], length=50, append=True)

    # Generate signal
    latest = df.iloc[-1]

    # Check if SMA values are valid
    if pd.isna(latest['SMA_20']) or pd.isna(latest['SMA_50']):
        return "hold"

    if latest['SMA_20'] > latest['SMA_50']:
        return "buy"
    elif latest['SMA_20'] < latest['SMA_50']:
        return "sell"
    else:
        return "hold"

if __name__ == "__main__":
    signal = generate_signal()
    print(f"Generated Signal: {signal}")
