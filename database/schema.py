import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

load_dotenv()

def create_schema():
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        cursor = conn.cursor()

        # Create the trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                price DOUBLE PRECISION,
                amount DOUBLE PRECISION,
                side TEXT
            );
        """)

        # Create the hypertable
        cursor.execute("SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);")

        conn.commit()
        cursor.close()
        print("Schema created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
