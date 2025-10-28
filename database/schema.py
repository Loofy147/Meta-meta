import psycopg2
import os
from dotenv import load_dotenv
import json

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

        timeframes = ['1m', '5m', '15m', '1h']
        for tf in timeframes:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS candles_{tf} (
                    time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION,
                    low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION,
                    CONSTRAINT unique_candle_{tf} UNIQUE (time, symbol)
                );
            """)
            cursor.execute(f"SELECT create_hypertable('candles_{tf}', 'time', if_not_exists => TRUE);")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS features_{tf} (
                    time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, rsi DOUBLE PRECISION, macd DOUBLE PRECISION,
                    macds DOUBLE PRECISION, macdh DOUBLE PRECISION, bb_lower DOUBLE PRECISION,
                    bb_mid DOUBLE PRECISION, bb_upper DOUBLE PRECISION,
                    CONSTRAINT unique_feature_{tf} UNIQUE (time, symbol)
                );
            """)
            cursor.execute(f"SELECT create_hypertable('features_{tf}', 'time', if_not_exists => TRUE);")

        # --- Portfolio, Performance, and Signal Tables ---
        cursor.execute("CREATE TABLE IF NOT EXISTS portfolios (id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, cash_balance DOUBLE PRECISION NOT NULL);")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY, portfolio_id INTEGER REFERENCES portfolios(id), symbol TEXT NOT NULL,
                quantity DOUBLE PRECISION NOT NULL, average_entry_price DOUBLE PRECISION NOT NULL,
                UNIQUE (portfolio_id, symbol)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executed_trades (
                id SERIAL PRIMARY KEY, portfolio_id INTEGER REFERENCES portfolios(id), symbol TEXT NOT NULL,
                quantity DOUBLE PRECISION NOT NULL, price DOUBLE PRECISION NOT NULL, side TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT PRIMARY KEY,
                hit_rate DOUBLE PRECISION NOT NULL,
                total_pnl DOUBLE PRECISION NOT NULL,
                trade_count INTEGER NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id UUID PRIMARY KEY,
                asset TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                meta JSONB,
                timestamp TIMESTAMPTZ NOT NULL
            );
        """)
        cursor.execute("CREATE TABLE IF NOT EXISTS system_parameters (key TEXT PRIMARY KEY, value JSONB NOT NULL);")

        conn.commit()

        # --- Seeding ---
        default_config = {
            "strategies": {
                "rsi": {"enabled": True, "overbought_threshold": 70, "oversold_threshold": 30},
                "macd": {"enabled": True}, "sentiment": {"enabled": False}, "ml": {"enabled": False}
            },
            "risk_management": {"max_position_size_pct": 0.1, "max_portfolio_exposure_pct": 0.5}
        }
        cursor.execute("INSERT INTO system_parameters (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING;", ('config', json.dumps(default_config)))
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
