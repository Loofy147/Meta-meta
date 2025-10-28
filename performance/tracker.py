import redis
import json
import os
import psycopg2
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

def track_performance(redis_client):
    """
    Consumes executed trades from the event bus and updates the performance of the contributing strategies.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    stream_name = 'executed_trades'
    group_name = 'performance_tracker_group'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    while True:
        try:
            events = redis_client.xreadgroup(group_name, 'tracker_worker', {stream_name: '>'}, count=1, block=0)
            if not events:
                continue

            for _, messages in events:
                for message_id, message_data in messages:
                    trade = {k.decode(): json.loads(v.decode()) if v.decode().startswith('{') else v.decode() for k, v in message_data.items()}

                    # Get the signal that led to this trade
                    cursor.execute("SELECT meta FROM signals WHERE id = %s;", (trade['signal_id'],))
                    signal_meta = cursor.fetchone()[0]
                    contributing_strategies = [s['strategy'] for s in signal_meta['contributing_signals']]

                    # --- This is a simplified performance tracking logic ---
                    # A real system would need to track the outcome of the trade over time.
                    # For now, we'll just increment the trade count.

                    for strategy_name in contributing_strategies:
                        cursor.execute("""
                            INSERT INTO strategy_performance (strategy_name, hit_rate, total_pnl, trade_count)
                            VALUES (%s, 0, 0, 1)
                            ON CONFLICT (strategy_name) DO UPDATE
                            SET trade_count = strategy_performance.trade_count + 1;
                        """, (strategy_name,))
                    conn.commit()
                    print(f"Updated performance for strategies: {contributing_strategies}")

        except Exception as e:
            print(f"An error occurred in the performance tracker: {e}")

if __name__ == "__main__":
    print("Starting the performance tracker...")
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    track_performance(redis_client)
