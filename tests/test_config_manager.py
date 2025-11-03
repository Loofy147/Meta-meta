import unittest
import psycopg2
import json
import os
import time
import redis
from unittest.mock import patch
from config.manager import get_config, update_config, _clear_cache, CONFIG_UPDATE_CHANNEL

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        """Set up a real test database and Redis connection."""
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres_test"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

        # Ensure the table exists and is clean
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_parameters (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL
            );
        """)
        cursor.execute("DELETE FROM system_parameters WHERE key = 'config';")
        self.initial_config = {'strategies': {'rsi': {'enabled': True}}}
        cursor.execute(
            "INSERT INTO system_parameters (key, value) VALUES ('config', %s);",
            (json.dumps(self.initial_config),)
        )
        self.conn.commit()
        cursor.close()

        _clear_cache()

    def tearDown(self):
        """Close the database connection."""
        self.conn.close()

    @patch('config.manager.get_db_connection')
    def test_hot_reloading_via_redis(self, mock_get_db_connection):
        """
        Test that the configuration cache is invalidated via a Redis pub/sub message.
        """
        # Point the config manager to the test database
        mock_get_db_connection.return_value = self.conn

        # 1. First call to get_config() should fetch from the DB and cache it.
        config1 = get_config(db_conn=self.conn)
        self.assertEqual(config1, self.initial_config)

        # 2. Verify it's cached by calling again.
        get_config(db_conn=self.conn)
        # Since we are passing the connection, get_db_connection is not called
        mock_get_db_connection.assert_not_called()

        # 3. Directly update the database, simulating an external change.
        new_config = {'strategies': {'rsi': {'enabled': False}}}
        cursor = self.conn.cursor()
        cursor.execute("UPDATE system_parameters SET value = %s WHERE key = 'config';", (json.dumps(new_config),))
        self.conn.commit()
        cursor.close()

        # 4. Call get_config() again. It should STILL return the old, cached value.
        cached_config = get_config()
        self.assertEqual(cached_config, self.initial_config)

        # 5. Publish a message to the update channel to trigger hot-reloading.
        self.redis_client.publish(CONFIG_UPDATE_CHANNEL, "updated")
        time.sleep(0.1)  # Give the subscriber thread a moment to process the message

        # 6. Now, get_config() should re-fetch from the DB and return the new value.
        config3 = get_config()
        self.assertEqual(config3, new_config)

if __name__ == '__main__':
    unittest.main()
