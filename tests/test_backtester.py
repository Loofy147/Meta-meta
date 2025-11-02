import unittest
import psycopg2
import redis
import os
from datetime import datetime
from unittest.mock import patch
from backtester.event_driven_engine import run_backtest

class TestEventDrivenBacktester(unittest.TestCase):

    def setUp(self):
        """Set up the test database and Redis client."""
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres_test"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        self.cursor = self.conn.cursor()
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

        # Create schema for testing
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles_1m (
                time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION,
                high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION
            );
        """)
        # Clear the table to ensure a clean state for the test
        self.cursor.execute("DELETE FROM candles_1m;")
        self.conn.commit()

        # Seed with test data
        self.cursor.execute(
            "INSERT INTO candles_1m (time, symbol, close, volume) VALUES (%s, %s, %s, %s);",
            (datetime(2023, 1, 1, 0, 0), 'BTC/USDT', 50000.0, 1.0)
        )
        self.cursor.execute(
            "INSERT INTO candles_1m (time, symbol, close, volume) VALUES (%s, %s, %s, %s);",
            (datetime(2023, 1, 1, 0, 1), 'BTC/USDT', 50001.0, 1.5)
        )
        self.conn.commit()

        # Clear the Redis stream before the test
        self.redis_client.delete('ingested_trades')

    def tearDown(self):
        """Clean up the test database and Redis."""
        self.cursor.execute("DROP TABLE candles_1m;")
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        self.redis_client.delete('ingested_trades')

    def test_run_backtest(self):
        """Test that the backtester correctly replays data to the event bus."""
        run_backtest('BTC/USDT', '2023-01-01', '2023-01-02', db_conn=self.conn)

        # Verify data was published to Redis
        stream_data = self.redis_client.xrange('ingested_trades')
        self.assertEqual(len(stream_data), 2)

        first_event = stream_data[0][1]
        self.assertEqual(first_event[b'symbol'], b'BTC/USDT')
        self.assertEqual(float(first_event[b'price']), 50000.0)

        second_event = stream_data[1][1]
        self.assertEqual(second_event[b'symbol'], b'BTC/USDT')
        self.assertEqual(float(second_event[b'price']), 50001.0)

if __name__ == '__main__':
    unittest.main()
