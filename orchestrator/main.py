"""The central trade orchestrator for the autonomous trading loop.

This module acts as the central nervous system of the trading system. It is an
event-driven service that listens for high-confidence, aggregated signals from
the `aggregated_signals` Redis Stream.

For each incoming signal, the orchestrator manages the complete trade lifecycle:
1.  **Confidence Check**: It first verifies if the signal's confidence score
    meets a configurable threshold, discarding low-confidence signals.
2.  **Signal Persistence**: The valid signal is stored in the `signals` database
    table for auditing, analysis, and performance tracking.
3.  **Pre-Execution Risk Management**: It invokes the `RiskValidator` to ensure
    the potential trade complies with all predefined risk rules (e.g., maximum
    position size, portfolio exposure limits).
4.  **Trade Execution**: If the risk checks pass, it calls the `PortfolioManager`
    to execute the trade through the designated broker API.
"""

import redis
import json
import os
import uuid
import psycopg2
from dotenv import load_dotenv
from typing import Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio.manager import execute_trade
from risk.validator import RiskValidator
# The broker module is not directly used here; it is abstracted away by the
# portfolio manager, which is the sole execution gateway.

load_dotenv()

# --- Orchestrator Configuration ---
# Minimum confidence score required to consider a signal for execution.
CONFIDENCE_THRESHOLD = 0.55
# For simplicity, this example uses a fixed trade size. A real system would
# calculate this dynamically based on risk, volatility, etc.
TRADE_QUANTITY = 0.01

def get_redis_client() -> redis.Redis:
    """Establishes and returns a connection to the Redis server.

    Returns:
        An active Redis client instance.
    """
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0
    )

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

def persist_signal(signal: Dict[str, Any], conn: connection) -> None:
    """Stores a received signal in the database for auditing and analysis.

    Args:
        signal: The aggregated signal data received from the event bus.
        conn: An active psycopg2 database connection object.
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
            json.dumps(signal['meta']),  # Persist contributing signals as JSON
            signal['timestamp']
        ))
    conn.commit()

def process_signal(signal: Dict[str, Any]) -> None:
    """Processes a single aggregated signal through the full trade lifecycle.

    This function is the core logic of the orchestrator. It applies the
    confidence filter, persists the signal, runs risk checks, and triggers
    trade execution if all conditions are met.

    Args:
        signal: A dictionary representing the aggregated signal to process.
    """
    print(f"Orchestrator received signal: {signal['asset']} -> {signal['direction']} (Confidence: {signal['confidence']})")

    # 1. Confidence Check
    if float(signal['confidence']) < CONFIDENCE_THRESHOLD:
        print("Signal confidence below threshold. Ignoring.")
        return

    # 2. Persist the Signal for Auditing
    db_conn = get_db_connection()
    try:
        persist_signal(signal, db_conn)
        print(f"Signal {signal['signal_id']} persisted to database.")
    finally:
        db_conn.close()

    # 3. Pre-Execution Risk Validation
    risk_validator = RiskValidator()
    if not risk_validator.is_trade_safe(portfolio_name='default', symbol=signal['asset'], quantity=TRADE_QUANTITY):
        print(f"Risk validation failed for signal {signal['signal_id']}. Trade aborted.")
        return
    print("Pre-execution risk validation passed.")

    # 4. Instruct Portfolio Manager to Execute Trade
    print(f"Proceeding with trade execution for signal {signal['signal_id']}...")
    execute_trade(
        portfolio_name='default',
        symbol=signal['asset'],
        side=signal['direction'],
        quantity=TRADE_QUANTITY,
        signal_id=signal['signal_id']
    )
    print("Trade execution instruction sent to portfolio manager.")

def run_orchestrator() -> None:
    """The main entry point and infinite loop for the orchestrator service.

    This function establishes a connection to Redis, creates a consumer group
    for the `aggregated_signals` stream, and enters an infinite loop. Inside the
    loop, it blocks and waits for new signals, processing each one as it arrives
    and acknowledging it to ensure reliable, at-least-once processing.
    """
    redis_client = get_redis_client()
    stream_name = 'aggregated_signals'
    group_name = 'orchestrator_group'
    worker_name = f'orchestrator_worker_{os.getpid()}'

    try:
        # Create the consumer group; ignores error if it already exists.
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e).lower():
            raise

    print("Orchestrator is running and listening for aggregated signals...")
    while True:
        try:
            # Block and wait indefinitely for a new message.
            events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=1, block=0)
            if not events:
                continue

            for _, messages in events:
                for message_id, message_data in messages:
                    # Decode and deserialize the signal data from the stream.
                    signal = {
                        k.decode(): json.loads(v.decode()) if v.decode().startswith(('{', '[')) else v.decode()
                        for k, v in message_data.items()
                    }

                    process_signal(signal)

                    # Acknowledge the message to remove it from the pending entries list.
                    redis_client.xack(stream_name, group_name, message_id)

        except Exception as e:
            print(f"An unexpected error occurred in the orchestrator main loop: {e}")
            # In a production system, add a delay before retrying.
            time.sleep(10)

if __name__ == "__main__":
    run_orchestrator()
