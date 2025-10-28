import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def create_schema():
    """Establishes a database connection and creates/updates the necessary tables and hypertables."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        cursor = conn.cursor()

        # --- Data Tables ---
        cursor.execute("CREATE TABLE IF NOT EXISTS trades (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, price DOUBLE PRECISION, amount DOUBLE PRECISION, side TEXT);")
        cursor.execute("SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles_1m (
                time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION,
                low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION,
                CONSTRAINT unique_candle_1m UNIQUE (time, symbol)
            );
        """)
        cursor.execute("SELECT create_hypertable('candles_1m', 'time', if_not_exists => TRUE);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features_1m (
                time TIMESTAMTz NOT NULL, symbol TEXT NOT NULL, rsi DOUBLE PRECISION, macd DOUBLE PRECISION,
                macds DOUBLE PRECISION, macdh DOUBLE PRECISION, bb_lower DOUBLE PRECISION,
                bb_mid DOUBLE PRECISION, bb_upper DOUBLE PRECISION,
                CONSTRAINT unique_feature_1m UNIQUE (time, symbol)
            );
        """)
        cursor.execute("SELECT create_hypertable('features_1m', 'time', if_not_exists => TRUE);")

        # --- Portfolio Management Tables ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cash_balance DOUBLE PRECISION NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id),
                symbol TEXT NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                average_entry_price DOUBLE PRECISION NOT NULL,
                UNIQUE (portfolio_id, symbol)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executed_trades (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id),
                symbol TEXT NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                side TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL
            );
        """)

        conn.commit()

        # --- Seed a default portfolio if it doesn't exist ---
        cursor.execute("INSERT INTO portfolios (name, cash_balance) VALUES ('default', 100000) ON CONFLICT (name) DO NOTHING;")
        conn.commit()

        cursor.close()
        print("Schema created/updated successfully.")
    except Exception as e:
        print(f"An error occurred during schema creation: {e}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
