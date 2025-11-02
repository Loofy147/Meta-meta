"""
ML-Powered Trading Strategy

This module implements a trading strategy that uses a pre-trained machine
learning model to generate trading signals. It loads the latest features from the
database, feeds them to the model, and interprets the model's output to produce
a 'buy', 'sell', or 'hold' signal.
"""

import joblib
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple

load_dotenv()

MODEL_PATH = "ml/model.pkl"

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_latest_features(symbol: str, timeframe: str = '1m') -> pd.DataFrame:
    """
    Retrieves the most recent set of calculated features for a given symbol.
    """
    conn = get_db_connection()
    query = f"SELECT * FROM features_{timeframe} WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    return df

def generate_signal(symbol: str) -> Tuple[str, float]:
    """
    Generates a trading signal using the trained ML model.

    Args:
        symbol (str): The symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        Tuple[str, float]: A tuple containing the signal direction ('buy', 'sell',
                           'hold') and the model's confidence score.
    """
    if not os.path.exists(MODEL_PATH):
        print("ML model not found. Please train the model first.")
        return 'hold', 0.0

    model = joblib.load(MODEL_PATH)
    features_df = get_latest_features(symbol)

    if features_df.empty:
        return 'hold', 0.0

    # Prepare features for the model
    X = features_df.drop(columns=['time', 'symbol'])

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    direction = 'hold'
    if prediction == 1:
        direction = 'buy'
    elif prediction == -1:
        direction = 'sell'

    return direction, confidence

if __name__ == '__main__':
    direction, confidence = generate_signal('BTC/USDT')
    print(f"ML Strategy Signal for BTC/USDT: {direction} (Confidence: {confidence:.2f})")
