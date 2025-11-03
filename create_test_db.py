import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="password"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE postgres_test")
    print("Database 'postgres_test' created successfully.")
except Exception as e:
    # If the database already exists, it will throw an error.
    # We can ignore this error.
    if "already exists" in str(e):
        print("Database 'postgres_test' already exists.")
    else:
        print(f"Error creating database: {e}")
finally:
    if 'conn' in locals() and conn is not None:
        conn.close()
