"""Manages dynamic system configuration from a database source.

This module provides a centralized interface for retrieving and updating
the application's configuration. Settings are stored in a dedicated
'system_parameters' table in the database, enabling dynamic updates without
requiring a service restart.

To enhance performance, it employs a global in-memory cache. The configuration
is fetched from the database on the first request and subsequently served from
the cache. The cache is automatically invalidated upon any updates.
"""

import psycopg2
import json
import os
import redis
import threading
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from psycopg2.extensions import connection

load_dotenv()

# Global in-memory cache for the configuration.
_config_cache: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()
_subscriber_thread_started = False
CONFIG_UPDATE_CHANNEL = "config_updates"

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

def get_redis_client() -> redis.Redis:
    """Establishes and returns a connection to the Redis server."""
    return redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)

def _subscribe_to_updates():
    """Worker function to listen for config updates on a Redis channel."""
    redis_client = get_redis_client()
    pubsub = redis_client.pubsub()
    pubsub.subscribe(CONFIG_UPDATE_CHANNEL)
    print("Configuration manager subscribed to Redis updates...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            print("Received configuration update notification. Invalidating cache.")
            _clear_cache()

def _start_subscriber_thread():
    """Starts the Redis subscriber thread if it's not already running."""
    global _subscriber_thread_started
    if not _subscriber_thread_started:
        thread = threading.Thread(target=_subscribe_to_updates, daemon=True)
        thread.start()
        _subscriber_thread_started = True

def get_config(db_conn: Optional[connection] = None) -> Dict[str, Any]:
    """Retrievess the system configuration, utilizing a thread-safe, hot-reloadable cache.

    On the first call, this function fetches the configuration from the database
    and starts a background thread to listen for update notifications on Redis.
    Subsequent calls return the cached version. If the cache is invalidated by the
    Redis listener, the next call will automatically re-fetch from the database.

    Args:
        db_conn: An optional existing database connection. If provided, the function
                 will use it instead of creating a new one. The caller is
                 responsible for managing the connection's lifecycle.

    Returns:
        The system configuration as a dictionary.

    Raises:
        ValueError: If the 'config' key is not found in the database.
    """
    global _config_cache
    _start_subscriber_thread()  # Ensure the listener is running

    with _config_lock:
        if _config_cache is not None:
            return _config_cache

        conn_provided = db_conn is not None
        conn = db_conn if conn_provided else get_db_connection()

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
            if not conn_provided and conn:
                conn.close()

def update_config(new_config: Dict[str, Any]) -> None:
    """Updates the config in the DB and publishes a notification to Redis.

    This function serializes the provided dictionary to JSON, overwrites the
    existing configuration in the database, and then publishes a message to the
    `config_updates` Redis channel. This broadcast triggers all subscribed
    services to invalidate their local caches.

    Args:
        new_config: The new configuration dictionary to be saved. This will
            completely overwrite the existing configuration.
    """
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

    # Publish an update notification to the Redis channel
    redis_client = get_redis_client()
    redis_client.publish(CONFIG_UPDATE_CHANNEL, "updated")
    print("System configuration updated and notification published.")


def _clear_cache() -> None:
    """Forces a clear of the in-memory configuration cache in a thread-safe way.

    This is primarily a helper function for testing purposes to ensure a clean
    state between test cases.
    """
    global _config_cache
    with _config_lock:
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
