import pandas as pd
import psycopg2
import os
import joblib
from dotenv import load_dotenv

load_dotenv()

# Load the trained model
try:
    model = joblib.load('ml/lgbm_model.pkl')
except FileNotFoundError:
    model = None

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def generate_signal(symbol='BTC/USDT', timeframe='1m'):
    """
    Generates a trading signal using the trained machine learning model.
    """
    if model is None:
        print("ML model not found. Please train the model first.")
        return 'hold', 0.0

    conn = get_db_connection()
    try:
        # Fetch the latest features for the model
        query = f"""
            SELECT rsi, macd, macds, macdh
            FROM features_{timeframe}
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT 1;
        """
        df = pd.read_sql(query, conn, params=(symbol,))

        if df.empty:
            return 'hold', 0.0

        # Predict the label
        prediction = model.predict(df)[0]

        # Get the prediction probabilities
        probabilities = model.predict_proba(df)[0]
        confidence = max(probabilities)

        if prediction == 1:
            return 'buy', confidence
        elif prediction == -1:
            return 'sell', confidence
        else:
            return 'hold', 0.0

    except Exception as e:
        print(f"An error occurred in the ML strategy: {e}")
        return 'hold', 0.0
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    # Example usage
    direction, confidence = generate_signal('BTC/USDT')
    print(f"ML Strategy Signal: {direction}, Confidence: {confidence}")
