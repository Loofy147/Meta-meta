import psycopg2
import json
import os
from dotenv import load_dotenv

load_dotenv()

# In-memory cache for the configuration
_config_cache = None

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_config():
    """
    Retrieves the system configuration from the database, using an in-memory cache.
    """
    global _config_cache
    if _config_cache:
        return _config_cache

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM system_parameters WHERE key = 'config';")
    config = cursor.fetchone()[0]

    conn.close()
    _config_cache = config
    return config

def update_config(new_config):
    """
    Updates the system configuration in the database and invalidates the cache.
    """
    global _config_cache
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE system_parameters SET value = %s WHERE key = 'config';",
        (json.dumps(new_config),)
    )

    conn.commit()
    conn.close()

    # Invalidate the cache
    _config_cache = None
    print("System configuration updated successfully and cache invalidated.")

if __name__ == '__main__':
    # Example Usage
    print("Fetching config for the first time (should hit DB)...")
    config = get_config()
    print("Current Configuration:")
    print(json.dumps(config, indent=4))

    print("\nFetching config for the second time (should use cache)...")
    config = get_config()

    # Example of updating the config
    print("\nUpdating configuration...")
    config['strategies']['rsi']['overbought_threshold'] = 75
    update_config(config)

    print("\nFetching config after update (should hit DB again)...")
    config = get_config()
    print("\nUpdated Configuration:")
    print(json.dumps(config, indent=4))
