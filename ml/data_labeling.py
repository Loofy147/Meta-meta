import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def create_labeled_data(symbol, timeframe='1m', look_forward_periods=15, threshold=0.001):
    """
    Creates a labeled dataset for a given symbol and timeframe.
    The label is determined by the future price change.
    - 1 for buy (price increases by threshold)
    - -1 for sell (price decreases by threshold)
    - 0 for hold
    """
    conn = get_db_connection()

    # Fetch features and close prices
    query = f"""
        SELECT f.time, f.rsi, f.macd, f.macds, f.macdh, c.close
        FROM features_{timeframe} f
        JOIN candles_{timeframe} c ON f.time = c.time AND f.symbol = c.symbol
        WHERE f.symbol = '{symbol}'
        ORDER BY f.time;
    """
    df = pd.read_sql(query, conn, index_col='time')
    conn.close()

    # Calculate future returns
    df['future_return'] = df['close'].pct_change(periods=-look_forward_periods).shift(-look_forward_periods)

    # Create labels
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1
    df.loc[df['future_return'] < -threshold, 'label'] = -1

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df.drop(columns=['future_return', 'close'])

if __name__ == '__main__':
    print("Creating labeled data...")
    try:
        labeled_data = create_labeled_data('BTC/USDT')
        if not labeled_data.empty:
            # Save to a CSV for training
            labeled_data.to_csv('ml/labeled_btc_data.csv')
            print("Labeled data created and saved to 'ml/labeled_btc_data.csv'.")
            print(labeled_data.head())
            print(f"Label distribution:\\n{labeled_data['label'].value_counts()}")
        else:
            print("No data found to create labels.")
    except Exception as e:
        print(f"Could not create labeled data: {e}")
