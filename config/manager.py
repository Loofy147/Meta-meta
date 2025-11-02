"""
Configuration Manager

This module provides a centralized system for managing the application's
configuration. It retrieves settings from a dedicated 'system_parameters' table
in the database, allowing for dynamic updates without requiring a service restart.

To improve performance, it uses an in-memory cache to store the configuration
after the first read. The cache is automatically invalidated upon any updates.
"""

import psycopg2
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from psycopg2.extensions import connection

load_dotenv()

# Global in-memory cache for the configuration
_config_cache: Optional[Dict[str, Any]] = None

def get_db_connection() -> connection:
    """
    Establishes and returns a connection to the PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: A database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_config() -> Dict[str, Any]:
    """
    Retrieves the system configuration, utilizing an in-memory cache.

    On the first call, it fetches the configuration from the 'system_parameters'
    table in the database. Subsequent calls return the cached version. The cache
    is invalidated by calling update_config().

    Returns:
        Dict[str, Any]: The system configuration as a dictionary.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_parameters WHERE key = 'config';")
        result = cursor.fetchone()
        if result:
            config = result[0]
            _config_cache = config
            return config
        else:
            raise ValueError("Configuration not found in the database.")
    finally:
        conn.close()


def update_config(new_config: Dict[str, Any]) -> None:
    """
    Updates the system configuration in the database and invalidates the cache.

    Args:
        new_config (Dict[str, Any]): The new configuration dictionary to be saved.
                                     This will overwrite the existing config.
    """
    global _config_cache
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE system_parameters SET value = %s WHERE key = 'config';",
            (json.dumps(new_config),)
        )
        conn.commit()
    finally:
        conn.close()

    # Invalidate the cache to ensure the next get_config() call fetches fresh data
    _config_cache = None
    print("System configuration updated successfully and cache invalidated.")

def _clear_cache() -> None:
    """
    Forces a clear of the in-memory configuration cache.
    """
    global _config_cache
    _config_cache = None

if __name__ == '__main__':
    # Example Usage:
    # This demonstrates the caching behavior of the configuration manager.

    print("--- Configuration Manager Example ---")

    # Clear cache for a clean run
    _clear_cache()

    print("\n1. Fetching config for the first time (should hit DB)...")
    config = get_config()
    print("   Current Configuration:")
    print(json.dumps(config, indent=4))

    print("\n2. Fetching config for the second time (should use cache)...")
    cached_config = get_config()
    # In a real app, you wouldn't see a DB hit here.

    # 3. Example of updating the config
    print("\n3. Updating configuration...")
    config['strategies']['rsi']['overbought_threshold'] = 75
    update_config(config)

    print("\n4. Fetching config after update (should hit DB again)...")
    new_config = get_config()
    print("   Updated Configuration:")
    print(json.dumps(new_config, indent=4))
