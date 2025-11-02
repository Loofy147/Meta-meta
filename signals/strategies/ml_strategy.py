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
from typing import Tuple, Optional
from lightgbm import LGBMClassifier
from psycopg2.extensions import connection

load_dotenv()

MODEL_PATH = "ml/model.pkl"
_model: Optional[LGBMClassifier] = None

def load_model() -> Optional[LGBMClassifier]:
    """
    Loads the trained machine learning model from the disk.

    Returns:
        Optional[LGBMClassifier]: The loaded model object, or None if the file
                                  is not found.
    """
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        print("Loading ML model...")
        _model = joblib.load(MODEL_PATH)
    return _model

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

def get_latest_features(symbol: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
    """
    Retrieves the most recent set of calculated features for a given symbol.

    Args:
        symbol (str): The symbol to fetch features for.
        timeframe (str): The timeframe to use (e.g., '1m').

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the latest features, or None.
    """
    conn = get_db_connection()
    try:
        query = f"SELECT * FROM features_{timeframe} WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
        df = pd.read_sql(query, conn, params=(symbol,))
        return df if not df.empty else None
    finally:
        conn.close()

def generate_signal(symbol: str) -> Tuple[str, float]:
    """
    Generates a trading signal using the trained ML model.

    This function fetches the latest features, passes them to the loaded model,
    and converts the model's prediction and probability into a standardized
    signal format.

    Args:
        symbol (str): The symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        Tuple[str, float]: A tuple containing the signal direction ('buy', 'sell',
                           'hold') and the model's confidence score (0.0 to 1.0).
    """
    model = load_model()
    if model is None:
        print("ML model not found. Please train the model first. Skipping ML strategy.")
        return 'hold', 0.0

    features_df = get_latest_features(symbol)
    if features_df is None:
        print(f"No features found for {symbol}. Skipping ML strategy.")
        return 'hold', 0.0

    # Ensure the feature columns match what the model was trained on
    model_features = model.feature_name_
    X = features_df[model_features]

    # Get prediction and confidence (probability of the predicted class)
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    if prediction == 1:
        return 'buy', confidence
    elif prediction == -1:
        return 'sell', confidence
    else: # prediction == 0
        return 'hold', 0.0

if __name__ == '__main__':
    # Example Usage:
    # Demonstrates how to generate a signal using the ML strategy.

    print("--- ML Strategy Example ---")
    direction, confidence = generate_signal('BTC/USDT')
    print(f"Generated Signal for BTC/USDT: {direction.upper()} (Confidence: {confidence:.2f})")
