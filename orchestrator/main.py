import redis
import json
import os
from dotenv import load_dotenv
import psycopg2
import uuid

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio.manager import execute_trade, get_portfolio_status
from risk.validator import RiskValidator
from broker.alpaca_broker import AlpacaBroker

load_dotenv()

def get_redis_client():
    """Establishes and returns a Redis client connection."""
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0
    )

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def main():
    """
    The main loop of the orchestrator.
    Listens for aggregated signals and manages the trade lifecycle.
    """
    redis_client = get_redis_client()

    stream_name = 'aggregated_signals'
    group_name = 'orchestrator_group'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    print("Orchestrator is running and listening for signals...")
    while True:
        try:
            events = redis_client.xreadgroup(group_name, 'orchestrator_worker', {stream_name: '>'}, count=1, block=0)
            if not events:
                continue

            for _, messages in events:
                for message_id, message_data in messages:
                    signal = {k.decode(): json.loads(v.decode()) if v.decode().startswith('{') else v.decode() for k, v in message_data.items()}

                    print(f"Orchestrator received signal: {signal}")

                    if float(signal['confidence']) > 0.5:
                        # Store the signal in the database
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO signals (id, asset, direction, confidence, meta, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s);
                        """, (uuid.UUID(signal['signal_id']), signal['asset'], signal['direction'], signal['confidence'], json.dumps(signal['meta']), signal['timestamp']))
                        conn.commit()
                        conn.close()

                        # ... rest of the logic ...

                        execute_trade(
                            portfolio_name='default',
                            symbol=signal['asset'],
                            side=signal['direction'],
                            quantity=0.01, # Simplified for now
                            signal_id=signal['signal_id']
                        )

        except Exception as e:
            print(f"An error occurred in the orchestrator: {e}")

if __name__ == "__main__":
    main()
