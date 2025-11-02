"""
Strategy Performance Tracker

This service is responsible for evaluating the performance of individual trading
strategies in real-time. It listens for executed trades on the event bus,
determines the outcome of each trade (profit or loss), and attributes that
outcome to the strategies that contributed to the original signal.

The calculated performance metrics (hit rate, total P&L) are stored in the
database and used by the adaptive signal aggregator to dynamically weight
strategies based on their historical effectiveness.
"""

import redis
import json
import os
import uuid
import psycopg2
from dotenv import load_dotenv
from typing import Dict, Any
from psycopg2.extensions import connection, cursor

load_dotenv()

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

def handle_trade_opening(cur: cursor, trade: Dict[str, Any], signal_meta: Dict[str, Any]) -> None:
    """
    Records a new open trade in the database.

    Args:
        cur (psycopg2.extensions.cursor): The database cursor.
        trade (Dict[str, Any]): The executed trade data from the event bus.
        signal_meta (Dict[str, Any]): The metadata of the signal that triggered the trade.
    """
    contributing_strategies = [s['strategy'] for s in signal_meta['contributing_signals']]
    cur.execute(
        """
        INSERT INTO open_trades (trade_id, signal_id, asset, direction, entry_price, timestamp, contributing_strategies)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """,
        (
            str(uuid.uuid4()),
            trade['signal_id'],
            trade['symbol'],
            signal_meta['direction'],
            float(trade['price']),
            trade['timestamp'],
            json.dumps(contributing_strategies)
        )
    )

def handle_trade_closing(cur: cursor, closing_trade: Dict[str, Any], open_trade: Dict[str, Any]) -> None:
    """
    Closes an existing trade, calculates its P&L, and updates strategy performance.

    Args:
        cur (psycopg2.extensions.cursor): The database cursor.
        closing_trade (Dict[str, Any]): The new trade that closes the position.
        open_trade (Dict[str, Any]): The original trade that opened the position.
    """
    # --- P&L Calculation ---
    pnl = 0.0
    if open_trade['direction'] == 'buy':
        pnl = float(closing_trade['price']) - open_trade['entry_price']
    else:  # sell
        pnl = open_trade['entry_price'] - float(closing_trade['price'])

    is_hit = 1 if pnl > 0 else 0
    # ---------------------

    # --- Update Performance Metrics ---
    for strategy_name in open_trade['contributing_strategies']:
        # This SQL query atomically updates the performance metrics for a strategy.
        # It calculates the new hit rate based on the updated total hits and trade count.
        cur.execute(
            """
            INSERT INTO strategy_performance (strategy_name, hit_rate, total_pnl, trade_count, total_hits)
            VALUES (%s, %s, %s, 1, %s)
            ON CONFLICT (strategy_name) DO UPDATE
            SET
                trade_count = strategy_performance.trade_count + 1,
                total_pnl = strategy_performance.total_pnl + EXCLUDED.total_pnl,
                total_hits = strategy_performance.total_hits + EXCLUDED.total_hits,
                hit_rate = (strategy_performance.total_hits + EXCLUDED.total_hits)::float / (strategy_performance.trade_count + 1.0);
            """,
            (strategy_name, is_hit, pnl, is_hit)
        )
    # --------------------------------

    # Remove the trade from the open_trades table
    cur.execute("DELETE FROM open_trades WHERE trade_id = %s;", (open_trade['trade_id'],))

def run_performance_tracker() -> None:
    """
    The main loop for the performance tracking service.

    It listens for executed trades, determines if they open or close a position,
    and updates performance metrics accordingly.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()

    stream_name = 'executed_trades'
    group_name = 'performance_tracker_group'
    worker_name = f'perf_worker_{os.getpid()}'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    print("Performance tracker started. Waiting for trades...")
    while True:
        try:
            with conn.cursor() as cur:
                events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=1, block=0)
                if not events:
                    continue

                for _, messages in events:
                    for message_id, message_data in messages:
                        trade = {k.decode(): v.decode() for k, v in message_data.items()}

                        # Fetch the original signal to get metadata
                        cur.execute("SELECT meta FROM signals WHERE id = %s;", (trade['signal_id'],))
                        signal_meta = cur.fetchone()[0]

                        # Check if there's an open position for this asset
                        cur.execute("SELECT trade_id, direction, entry_price, contributing_strategies FROM open_trades WHERE asset = %s;", (trade['symbol'],))
                        open_trade_data = cur.fetchone()

                        if open_trade_data:
                            open_trade = {
                                "trade_id": open_trade_data[0], "direction": open_trade_data[1],
                                "entry_price": open_trade_data[2],
                                "contributing_strategies": open_trade_data[3]
                            }
                            handle_trade_closing(cur, trade, open_trade)
                            print(f"Closed trade for {trade['symbol']}. Performance updated.")
                        else:
                            handle_trade_opening(cur, trade, signal_meta)
                            print(f"Opened new trade for {trade['symbol']}.")

                        conn.commit()
                        redis_client.xack(stream_name, group_name, message_id)

        except Exception as e:
            print(f"An error occurred in the performance tracker: {e}")
            conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    run_performance_tracker()
