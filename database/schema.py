"""Defines and initializes the complete database schema.

This script is responsible for creating all necessary tables, enabling the
TimescaleDB extension, and converting time-series tables into hypertables
for efficient data handling. It is designed to be idempotent, meaning it can
be safely run multiple times without causing errors or data loss.

The script also seeds the database with initial essential data, such as a
default system configuration and a default trading portfolio, ensuring the
application can start in a known state.
"""

import psycopg2
import os
from dotenv import load_dotenv
import json
from psycopg2.extensions import connection

load_dotenv()

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

def create_schema() -> None:
    """Creates/updates all tables, hypertables, and seeds initial data.

    This function serves as the main entry point for database schema management.
    It connects to the database and executes a series of SQL commands to define
    the required tables for:
    - Time-series market data (raw trades, candles, features).
    - Application state (portfolios, positions).
    - Performance tracking (strategy performance).
    - System metadata (signals, open trades, system parameters).

    It ensures the TimescaleDB extension is active and converts relevant tables
    into hypertables. Finally, it seeds the `system_parameters` and `portfolios`
    tables with default values.
    """
    conn = None  # Initialize conn to None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            print("Creating schema and tables...")

            # --- Time-Series Data Tables (Hypertables) ---
            print("Creating time-series data tables...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_trades (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    side TEXT
                );
            """)
            cursor.execute("SELECT create_hypertable('raw_trades', 'time', if_not_exists => TRUE);")

            timeframes = ['1m', '5m', '15m', '1h']
            for tf in timeframes:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS candles_{tf} (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        CONSTRAINT unique_candle_{tf} UNIQUE (time, symbol)
                    );
                """)
                cursor.execute(f"SELECT create_hypertable('candles_{tf}', 'time', if_not_exists => TRUE);")

                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS features_{tf} (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        rsi DOUBLE PRECISION,
                        macd DOUBLE PRECISION,
                        macds DOUBLE PRECISION,
                        macdh DOUBLE PRECISION,
                        bb_lower DOUBLE PRECISION,
                        bb_mid DOUBLE PRECISION,
                        bb_upper DOUBLE PRECISION,
                        CONSTRAINT unique_feature_{tf} UNIQUE (time, symbol)
                    );
                """)
                cursor.execute(f"SELECT create_hypertable('features_{tf}', 'time', if_not_exists => TRUE);")

            # --- Application State and Metadata Tables ---
            print("Creating application state and metadata tables...")
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
                    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
                    symbol TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    average_entry_price DOUBLE PRECISION NOT NULL,
                    UNIQUE (portfolio_id, symbol)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT PRIMARY KEY,
                    hit_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    trade_count INTEGER NOT NULL DEFAULT 0,
                    total_hits INTEGER NOT NULL DEFAULT 0
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS open_trades (
                    trade_id UUID PRIMARY KEY,
                    signal_id UUID NOT NULL REFERENCES signals(id),
                    asset TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    contributing_strategies JSONB
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_parameters (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL
                );
            """)

            # --- Seeding Initial Data ---
            print("Seeding initial data...")
            default_config = {
                "ingestion": {
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                },
                "strategies": {
                    "rsi": {"enabled": True, "overbought_threshold": 70, "oversold_threshold": 30},
                    "macd": {"enabled": True},
                    "sentiment": {"enabled": False},
                    "ml": {"enabled": False}
                },
                "risk_management": {
                    "max_position_size_pct": 0.1,
                    "max_portfolio_exposure_pct": 0.5
                }
            }
            cursor.execute(
                "INSERT INTO system_parameters (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING;",
                ('config', json.dumps(default_config))
            )
            cursor.execute(
                "INSERT INTO portfolios (name, cash_balance) VALUES ('default', 100000) ON CONFLICT (name) DO NOTHING;"
            )

            conn.commit()
            print("Schema created/updated and seeded successfully.")

    except Exception as e:
        print(f"An error occurred during schema creation: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_schema()
