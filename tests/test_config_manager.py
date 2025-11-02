import unittest
from unittest.mock import patch
from config.manager import get_config, update_config, _config_cache

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        # Reset the cache before each test
        global _config_cache
        _config_cache = None

    @patch('config.manager.get_db_connection')
    def test_caching_logic(self, mock_get_db_connection):
        """
        Test that the configuration is cached after the first call
        and the cache is invalidated after an update.
        """
        # Mock the database connection and cursor
        mock_conn = mock_get_db_connection.return_value
        mock_cursor = mock_conn.cursor.return_value

        # 1. First call to get_config() should hit the database
        mock_cursor.fetchone.return_value = [{'strategies': {'rsi': {'enabled': True}}}]
        config1 = get_config()
        mock_get_db_connection.assert_called_once()
        self.assertEqual(config1, {'strategies': {'rsi': {'enabled': True}}})

        # 2. Second call to get_config() should use the cache
        config2 = get_config()
        mock_get_db_connection.assert_called_once() # Should not be called again
        self.assertEqual(config2, config1)

        # 3. Call update_config() to invalidate the cache
        new_config = {'strategies': {'rsi': {'enabled': False}}}
        update_config(new_config)

        # 4. Third call to get_config() should hit the database again
        mock_cursor.fetchone.return_value = [new_config]
        config3 = get_config()
        self.assertEqual(mock_get_db_connection.call_count, 3) # Called for get, update, and get again
        self.assertEqual(config3, new_config)

if __name__ == '__main__':
    unittest.main()
