"""
Trade Orchestrator

This is the central nervous system of the autonomous trading loop. The orchestrator
listens for high-level, aggregated signals from the event bus. For each signal
that meets the confidence threshold, it manages the entire trade lifecycle:

1.  **Signal Persistence:** Stores the signal in the database for auditing and
    performance tracking.
2.  **Risk Management:** Validates the potential trade against a set of predefined
    risk rules (e.g., max position size, portfolio exposure).
3.  **Execution:** If the trade is deemed safe, it instructs the portfolio
    manager to execute the trade via the connected broker.
"""

import redis
import json
import os
import uuid
import psycopg2
from dotenv import load_dotenv
from typing import Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio.manager import execute_trade
from risk.validator import RiskValidator
# The broker is not directly used here but is called by the portfolio manager.

load_dotenv()

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.55 # Minimum confidence to consider a signal for execution
TRADE_QUANTITY = 0.01 # Simplified fixed quantity for this example

def get_redis_client() -> redis.Redis:
    """
    Establishes and returns a connection to the Redis server.

    Returns:
        redis.Redis: An active Redis client instance.
    """
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0
    )

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

def persist_signal(signal: Dict[str, Any], conn: connection) -> None:
    """
    Stores a received signal in the database for auditing and analysis.

    Args:
        signal (Dict[str, Any]): The aggregated signal data.
        conn (psycopg2.extensions.connection): An active database connection.
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO signals (id, asset, direction, confidence, meta, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (
            uuid.UUID(signal['signal_id']),
            signal['asset'],
            signal['direction'],
            signal['confidence'],
            json.dumps(signal['meta']),
            signal['timestamp']
        ))
    conn.commit()

def process_signal(signal: Dict[str, Any]) -> None:
    """
    Processes a single aggregated signal, handling the full trade lifecycle.

    Args:
        signal (Dict[str, Any]): The aggregated signal to process.
    """
    print(f"Orchestrator received signal: {signal['asset']} -> {signal['direction']} (Confidence: {signal['confidence']})")

    # 1. Confidence Check
    if float(signal['confidence']) < CONFIDENCE_THRESHOLD:
        print("Signal confidence below threshold. Ignoring.")
        return

    # 2. Persist the Signal
    db_conn = get_db_connection()
    try:
        persist_signal(signal, db_conn)
        print(f"Signal {signal['signal_id']} persisted to database.")
    finally:
        db_conn.close()

    # 3. Risk Validation
    risk_validator = RiskValidator()
    if not risk_validator.is_trade_safe(portfolio_name='default', symbol=signal['asset'], quantity=TRADE_QUANTITY):
        print(f"Risk validation failed for signal {signal['signal_id']}. Trade aborted.")
        return
    print("Risk validation passed.")

    # 4. Execute Trade
    print(f"Proceeding with trade execution for signal {signal['signal_id']}...")
    execute_trade(
        portfolio_name='default',
        symbol=signal['asset'],
        side=signal['direction'],
        quantity=TRADE_QUANTITY,
        signal_id=signal['signal_id']
    )

def run_orchestrator() -> None:
    """
    The main loop of the orchestrator service.

    It listens for aggregated signals on a Redis Stream and manages the trade
    lifecycle for each valid signal.
    """
    redis_client = get_redis_client()
    stream_name = 'aggregated_signals'
    group_name = 'orchestrator_group'
    worker_name = f'orchestrator_worker_{os.getpid()}'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    print("Orchestrator is running and listening for signals...")
    while True:
        try:
            events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=1, block=0)
            if not events:
                continue

            for _, messages in events:
                for message_id, message_data in messages:
                    # Decode and parse the signal data from the stream
                    signal = {
                        k.decode(): json.loads(v.decode()) if v.decode().startswith('{') else v.decode()
                        for k, v in message_data.items()
                    }

                    process_signal(signal)

                    # Acknowledge the message after processing
                    redis_client.xack(stream_name, group_name, message_id)

        except Exception as e:
            print(f"An error occurred in the orchestrator main loop: {e}")

if __name__ == "__main__":
    run_orchestrator()
