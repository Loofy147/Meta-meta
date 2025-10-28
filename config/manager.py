import psycopg2
import json
import os
from dotenv import load_dotenv

load_dotenv()

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
    Retrieves the system configuration from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM system_parameters WHERE key = 'config';")
    config = cursor.fetchone()[0]

    conn.close()
    return config

def update_config(new_config):
    """
    Updates the system configuration in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE system_parameters SET value = %s WHERE key = 'config';",
        (json.dumps(new_config),)
    )

    conn.commit()
    conn.close()
    print("System configuration updated successfully.")

if __name__ == '__main__':
    # Example Usage
    config = get_config()
    print("Current Configuration:")
    print(json.dumps(config, indent=4))

    # # Example of updating the config
    # config['strategies']['rsi']['overbought_threshold'] = 75
    # update_config(config)

    # config = get_config()
    # print("\\nUpdated Configuration:")
    # print(json.dumps(config, indent=4))
