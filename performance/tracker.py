import redis
import json
import os
import psycopg2
from dotenv import load_dotenv
import uuid

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def handle_trade_opening(cursor, trade, signal_meta):
    """Handles the logic for opening a new trade."""
    contributing_strategies = [s['strategy'] for s in signal_meta['contributing_signals']]
    cursor.execute(
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

def handle_trade_closing(cursor, trade, open_trade):
    """Handles the logic for closing an existing trade."""
    pnl = 0
    if open_trade['direction'] == 'buy':
        pnl = float(trade['price']) - open_trade['entry_price']
    else: # sell
        pnl = open_trade['entry_price'] - float(trade['price'])

    is_hit = 1 if pnl > 0 else 0

    for strategy_name in open_trade['contributing_strategies']:
        cursor.execute(
            """
            INSERT INTO strategy_performance (strategy_name, hit_rate, total_pnl, trade_count, total_hits)
            VALUES (%s, %s, %s, 1, %s)
            ON CONFLICT (strategy_name) DO UPDATE
            SET
                trade_count = strategy_performance.trade_count + 1,
                total_pnl = strategy_performance.total_pnl + EXCLUDED.total_pnl,
                total_hits = strategy_performance.total_hits + EXCLUDED.total_hits,
                hit_rate = (strategy_performance.total_hits + EXCLUDED.total_hits) / (strategy_performance.trade_count + 1.0);
            """,
            (strategy_name, is_hit, pnl, is_hit)
        )

    cursor.execute("DELETE FROM open_trades WHERE trade_id = %s;", (open_trade['trade_id'],))

def track_performance():
    """
    Consumes executed trades from the event bus and updates the performance of the contributing strategies.
    """
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    conn = get_db_connection()
    cursor = conn.cursor()

    stream_name = 'executed_trades'
    group_name = 'performance_tracker_group'

    try:
        redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "already exists" not in str(e):
            raise

    print("Performance tracker started. Waiting for trades...")
    while True:
        try:
            events = redis_client.xreadgroup(group_name, 'tracker_worker', {stream_name: '>'}, count=1, block=0)
            if not events:
                continue

            for _, messages in events:
                for message_id, message_data in messages:
                    trade = {k.decode(): v.decode() for k, v in message_data.items()}

                    cursor.execute("SELECT meta FROM signals WHERE id = %s;", (trade['signal_id'],))
                    signal_meta = cursor.fetchone()[0]

                    cursor.execute("SELECT trade_id, direction, entry_price, contributing_strategies FROM open_trades WHERE asset = %s;", (trade['symbol'],))
                    open_trade_data = cursor.fetchone()

                    if open_trade_data:
                        open_trade = {
                            "trade_id": open_trade_data[0],
                            "direction": open_trade_data[1],
                            "entry_price": open_trade_data[2],
                            "contributing_strategies": open_trade_data[3]
                        }
                        handle_trade_closing(cursor, trade, open_trade)
                        print(f"Closed trade for {trade['symbol']}. Performance updated.")
                    else:
                        handle_trade_opening(cursor, trade, signal_meta)
                        print(f"Opened new trade for {trade['symbol']}.")

                    conn.commit()
                    redis_client.xack(stream_name, group_name, message_id)

        except Exception as e:
            print(f"An error occurred in the performance tracker: {e}")
            conn.rollback()
            time.sleep(10)

if __name__ == "__main__":
    track_performance()
