"""Implements a trading strategy powered by a machine learning model.

This module provides a signal generation function that uses a pre-trained
LightGBM classification model to predict market direction. It fetches the
latest set of technical analysis features from the database, feeds them into the
model, and interprets the model's output probability as a confidence score.

The model is expected to be saved at `ml/model.pkl` and should be trained to
predict three classes:
- `1`: 'buy' (upward price movement expected)
- `-1`: 'sell' (downward price movement expected)
- `0`: 'hold' (no significant movement expected)
"""

import joblib
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from typing import Tuple, Optional, Any
from lightgbm import LGBMClassifier
from psycopg2.extensions import connection

load_dotenv()

MODEL_PATH = "ml/model.pkl"
# A global cache for the loaded model to avoid repeated file I/O.
_model: Optional[LGBMClassifier] = None

def load_model() -> Optional[LGBMClassifier]:
    """Loads the trained ML model from disk into a global cache.

    On the first call, it loads the model from the path specified by
    `MODEL_PATH` using `joblib`. Subsequent calls return the cached model
    instance directly.

    Returns:
        The loaded LightGBM classifier model, or None if the model file
        is not found.
    """
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print("Loading ML model for the first time...")
            _model = joblib.load(MODEL_PATH)
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    return _model

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

def get_latest_features(symbol: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
    """Retrieves the most recent row of features for a given symbol.

    Args:
        symbol: The symbol to fetch features for (e.g., 'BTC/USDT').
        timeframe: The timeframe to use (e.g., '1m'), corresponding to the
            `features_{timeframe}` table.

    Returns:
        A single-row DataFrame containing the latest features, or None if no
        features are found.
    """
    conn = get_db_connection()
    try:
        query = f"SELECT * FROM features_{timeframe} WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
        df = pd.read_sql(query, conn, params=(symbol,))
        return df if not df.empty else None
    finally:
        conn.close()

def generate_signal(symbol: str) -> Tuple[str, float]:
    """Generates a trading signal using the trained ML model.

    This function orchestrates the process of loading the ML model, fetching the
    latest feature data, preparing the data to match the model's expected
    input format, and making a prediction. The model's predicted class is
    mapped to a 'buy'/'sell'/'hold' signal, and the prediction probability is
    used as the confidence score.

    Args:
        symbol: The symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        A tuple containing the signal direction ('buy', 'sell', or 'hold')
        and the model's confidence score (the probability of the predicted
        class). Returns ('hold', 0.0) if the model or features are unavailable.
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
