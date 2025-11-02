import unittest
import psycopg2
import os
import json
import uuid
from datetime import datetime
from performance.tracker import handle_trade_opening, handle_trade_closing

class TestPerformanceTracker(unittest.TestCase):
    def setUp(self):
        """Set up the test database and create necessary tables."""
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres_test"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        self.cursor = self.conn.cursor()

        # Create schema for testing
        self.cursor.execute("CREATE TABLE IF NOT EXISTS signals (id UUID PRIMARY KEY, meta JSONB);")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS open_trades (
                trade_id UUID PRIMARY KEY,
                signal_id UUID REFERENCES signals(id),
                asset TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                contributing_strategies JSONB
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT PRIMARY KEY,
                hit_rate DOUBLE PRECISION NOT NULL,
                total_pnl DOUBLE PRECISION NOT NULL,
                trade_count INTEGER NOT NULL,
                total_hits INTEGER NOT NULL DEFAULT 0
            );
        """)
        self.conn.commit()

    def tearDown(self):
        """Clean up the test database."""
        self.cursor.execute("DROP TABLE IF EXISTS open_trades;")
        self.cursor.execute("DROP TABLE IF EXISTS strategy_performance;")
        self.cursor.execute("DROP TABLE IF EXISTS signals;")
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def test_performance_tracking_cycle(self):
        """Test the full cycle of opening and closing a trade."""
        # 1. Setup mock data
        signal_id = str(uuid.uuid4())
        trade_open = {
            'signal_id': signal_id,
            'symbol': 'BTC/USD',
            'price': '50000.0',
            'timestamp': datetime.now().isoformat()
        }
        signal_meta_open = {
            'direction': 'buy',
            'contributing_signals': [{'strategy': 'rsi'}, {'strategy': 'macd'}]
        }
        trade_close = {
            'signal_id': str(uuid.uuid4()),
            'symbol': 'BTC/USD',
            'price': '51000.0',
            'timestamp': datetime.now().isoformat()
        }

        # 2. Open a trade
        self.cursor.execute("INSERT INTO signals (id, meta) VALUES (%s, %s);", (signal_id, json.dumps(signal_meta_open)))
        handle_trade_opening(self.cursor, trade_open, signal_meta_open)
        self.conn.commit()

        # 3. Verify open trade is recorded
        self.cursor.execute("SELECT * FROM open_trades;")
        open_trade_data = self.cursor.fetchone()
        self.assertIsNotNone(open_trade_data)
        open_trade = {
            "trade_id": open_trade_data[0],
            "direction": open_trade_data[3],
            "entry_price": open_trade_data[4],
            "contributing_strategies": open_trade_data[6]
        }


        # 4. Close the trade
        handle_trade_closing(self.cursor, trade_close, open_trade)
        self.conn.commit()

        # 5. Verify performance metrics
        for strategy in ['rsi', 'macd']:
            self.cursor.execute("SELECT hit_rate, total_pnl, trade_count, total_hits FROM strategy_performance WHERE strategy_name = %s;", (strategy,))
            perf = self.cursor.fetchone()
            self.assertIsNotNone(perf, f"Performance data for {strategy} not found.")
            self.assertEqual(perf[0], 1.0, f"Hit rate for {strategy} is incorrect.") # 1.0 because 1 hit / 1 trade
            self.assertEqual(perf[1], 1000.0, f"P&L for {strategy} is incorrect.")      # 51000 - 50000
            self.assertEqual(perf[2], 1, f"Trade count for {strategy} is incorrect.")
            self.assertEqual(perf[3], 1, f"Total hits for {strategy} is incorrect.")

        # 6. Verify the open trade is deleted
        self.cursor.execute("SELECT * FROM open_trades;")
        self.assertIsNone(self.cursor.fetchone())

if __name__ == '__main__':
    unittest.main()
