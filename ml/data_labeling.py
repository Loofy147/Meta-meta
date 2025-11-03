"""Prepares a labeled dataset for training the machine learning model.

This module provides the functionality to create a labeled dataset from the
historical feature and price data stored in the database. This is a crucial
preprocessing step required before the ML model can be trained.

The core logic involves applying a "triple-barrier" labeling method:
1.  **Forward-Looking Returns**: For each timestamp, it calculates the
    percentage price change over a specified future period (the "look-forward"
    window).
2.  **Threshold-Based Labeling**: It then assigns a label based on this future
    return:
    - `1` (buy): If the price increases by more than a defined threshold.
    - `-1` (sell): If the price decreases by more than the threshold.
    - `0` (hold): If the price change is within the neutral threshold.
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_db_connection():
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

def create_labeled_data(symbol: str, timeframe: str = '1m', look_forward_periods: int = 15, threshold: float = 0.001) -> pd.DataFrame:
    """Creates a labeled dataset for ML model training.

    This function fetches historical features and prices, calculates future
    price returns, and assigns a label (buy, sell, or hold) to each data point
    based on whether the future return exceeds a specified threshold.

    Args:
        symbol: The trading symbol to create data for (e.g., 'BTC/USDT').
        timeframe: The candle timeframe to use (e.g., '1m').
        look_forward_periods: The number of future periods to look ahead to
            calculate the return for labeling (e.g., 15 periods for a 15-minute
            lookahead on 1m data).
        threshold: The percentage change (e.g., 0.001 for 0.1%) required to
            assign a 'buy' or 'sell' label.

    Returns:
        A pandas DataFrame containing the features and the corresponding
        'label' column. Rows where a label could not be determined (e.g., at
        the end of the dataset) are dropped.
    """
    conn = get_db_connection()

    try:
        # Fetch features and their corresponding close prices.
        query = f"""
            SELECT f.*, c.close
            FROM features_{timeframe} f
            JOIN candles_{timeframe} c ON f.time = c.time AND f.symbol = c.symbol
            WHERE f.symbol = %s
            ORDER BY f.time;
        """
        df = pd.read_sql(query, conn, index_col='time', params=(symbol,))
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Calculate the percentage change N periods into the future.
    # `pct_change` with a negative period looks forward. `shift` aligns the
    # future return with the current timestamp.
    df['future_return'] = df['close'].pct_change(periods=look_forward_periods).shift(-look_forward_periods)

    # Assign labels based on the threshold.
    df['label'] = 0  # Default to 'hold'
    df.loc[df['future_return'] > threshold, 'label'] = 1  # 'buy'
    df.loc[df['future_return'] < -threshold, 'label'] = -1 # 'sell'

    # Remove rows where the future return couldn't be calculated (the last N rows).
    df.dropna(inplace=True)

    # Clean up the final DataFrame before returning.
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
