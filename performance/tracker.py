"""Monitors executed trades to track and update strategy performance metrics.

This service is a crucial component of the system's adaptive learning loop. It
listens for messages on the `executed_trades` Redis Stream. For each trade, it
determines whether it opens a new position or closes an existing one.

When a trade closes a position, this service calculates the Profit or Loss (P&L)
and attributes this outcome to all the strategies that contributed to the
original signal. It then atomically updates the performance metrics (hit rate,
total P&L, trade count) for each of those strategies in the
`strategy_performance` database table. These metrics are then used by the
adaptive signal aggregator to weight strategies based on their historical
effectiveness.
"""

import redis
import json
import os
import uuid
import psycopg2
import time
from dotenv import load_dotenv
from typing import Dict, Any
from psycopg2.extensions import connection, cursor

load_dotenv()

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

def handle_trade_opening(cur: cursor, trade: Dict[str, Any], signal_meta: Dict[str, Any]) -> None:
    """Records a new trade that opens a position in the `open_trades` table.

    Args:
        cur: An active psycopg2 database cursor.
        trade: The executed trade data from the event bus.
        signal_meta: The metadata of the signal that triggered the trade,
            containing the list of contributing strategies.
    """
    contributing_strategies = [s['strategy'] for s in signal_meta.get('contributing_signals', [])]
    cur.execute(
        """
        INSERT INTO open_trades (trade_id, signal_id, asset, direction, entry_price, timestamp, contributing_strategies)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """,
        (
            str(uuid.uuid4()),
            trade['signal_id'],
            trade['symbol'],
            signal_meta.get('direction', trade.get('side')), # Use signal direction
            float(trade['price']),
            trade['timestamp'],
            json.dumps(contributing_strategies)
        )
    )
    print(f"Opened new trade for {trade['symbol']} from signal {trade['signal_id']}.")


def handle_trade_closing(cur: cursor, closing_trade: Dict[str, Any], open_trade: Dict[str, Any]) -> None:
    """Closes a trade, calculates P&L, and updates strategy performance.

    This function calculates the profit or loss of the closed trade. It then
    atomically updates the `strategy_performance` table for every strategy
    that contributed to the original opening signal. The `ON CONFLICT` clause
    ensures that the updates are safe for concurrent operations.

    Args:
        cur: An active psycopg2 database cursor.
        closing_trade: The new trade event that closes the position.
        open_trade: The record of the original opening trade from the
            `open_trades` table.
    """
    # --- P&L Calculation ---
    pnl = 0.0
    # Assumes equal quantity for opening and closing trades.
    if open_trade['direction'] == 'buy':
        pnl = float(closing_trade['price']) - open_trade['entry_price']
    else:  # 'sell'
        pnl = open_trade['entry_price'] - float(closing_trade['price'])

    is_hit = 1 if pnl > 0 else 0
    # ---------------------

    # --- Atomically Update Performance Metrics for each contributing strategy ---
    for strategy_name in open_trade['contributing_strategies']:
        # This SQL query atomically creates or updates the performance metrics.
        # It correctly recalculates the hit rate based on the new totals.
        cur.execute(
            """
            INSERT INTO strategy_performance (strategy_name, hit_rate, total_pnl, trade_count, total_hits)
            VALUES (%s, %s, %s, 1, %s)
            ON CONFLICT (strategy_name) DO UPDATE
            SET
                trade_count = strategy_performance.trade_count + 1,
                total_pnl = strategy_performance.total_pnl + EXCLUDED.total_pnl,
                total_hits = strategy_performance.total_hits + EXCLUDED.total_hits,
                hit_rate = (strategy_performance.total_hits + EXCLUDED.total_hits)::float / (strategy_performance.trade_count + 1.0)
            WHERE strategy_performance.strategy_name = EXCLUDED.strategy_name;
            """,
            (strategy_name, float(is_hit), pnl, is_hit)
        )
    # --------------------------------

    # Remove the trade from the open_trades table now that it's closed.
    cur.execute("DELETE FROM open_trades WHERE trade_id = %s;", (open_trade['trade_id'],))
    print(f"Closed trade for {closing_trade['symbol']}. P&L: {pnl:.4f}. Performance updated.")


def run_performance_tracker() -> None:
    """The main entry point and infinite loop for the performance tracker service.

    This function establishes connections to Redis and PostgreSQL and enters an
    infinite loop to consume messages from the `executed_trades` stream. For
    each message, it determines if the trade is opening or closing a position
    and calls the appropriate handler function. All database operations for a
    single trade are performed within a transaction.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()

    stream_name = 'executed_trades'
    group_name = 'performance_tracker_group'
    worker_name = f'perf_worker_{os.getpid()}'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e).lower():
            raise

    print("Performance tracker started. Waiting for executed trades...")
    while True:
        try:
            with conn.cursor() as cur:
                events = redis_client.xreadgroup(group_name, worker_name, {stream_name: '>'}, count=1, block=0)
                if not events:
                    time.sleep(1) # Avoid busy-looping when no events
                    continue

                for _, messages in events:
                    for message_id, message_data in messages:
                        trade = {k.decode(): v.decode() for k, v in message_data.items()}

                        # Fetch the original signal to get its metadata.
                        cur.execute("SELECT meta FROM signals WHERE id = %s;", (trade['signal_id'],))
                        result = cur.fetchone()
                        if not result:
                            print(f"Warning: Signal ID {trade['signal_id']} not found. Cannot process trade.")
                            redis_client.xack(stream_name, group_name, message_id)
                            continue
                        signal_meta = result[0]

                        # Check if an open position for this asset already exists.
                        cur.execute("SELECT trade_id, direction, entry_price, contributing_strategies FROM open_trades WHERE asset = %s;", (trade['symbol'],))
                        open_trade_data = cur.fetchone()

                        if open_trade_data:
                            open_trade = {
                                "trade_id": open_trade_data[0], "direction": open_trade_data[1],
                                "entry_price": open_trade_data[2],
                                "contributing_strategies": open_trade_data[3]
                            }
                            # If the new trade is in the opposite direction, it's a closing trade.
                            if trade['side'] != open_trade['direction']:
                                handle_trade_closing(cur, trade, open_trade)
                            else:
                                # Logic for scaling into a position could be added here.
                                print(f"Received same-direction trade for open position on {trade['symbol']}. Ignoring for P&L.")
                        else:
                            handle_trade_opening(cur, trade, signal_meta)

                        conn.commit()
                        redis_client.xack(stream_name, group_name, message_id)

        except Exception as e:
            print(f"An error occurred in the performance tracker: {e}")
            if conn:
                conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    run_performance_tracker()
