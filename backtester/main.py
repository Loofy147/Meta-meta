import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def run_event_driven_backtest(symbol, start_date, end_date):
    """
    Runs a high-fidelity, event-driven backtest by replaying historical data.
    """
    conn = get_db_connection()
    publisher = EventPublisher()

    # Fetch historical trade data
    query = """
        SELECT time, symbol, price, amount, side
        FROM trades
        WHERE symbol = %s AND time BETWEEN %s AND %s
        ORDER BY time;
    """
    df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
    conn.close()

    if df.empty:
        print("No historical data found for the given symbol and date range.")
        return

    print(f"Replaying {len(df)} historical trades for {symbol}...")

    # Replay trades through the event bus
    for _, row in df.iterrows():
        event_data = {
            'symbol': row['symbol'],
            'price': row['price'],
            'amount': row['amount'],
            'side': row['side'],
            'timestamp': int(row['time'].timestamp() * 1000) # Convert to ms timestamp
        }
        publisher.publish('raw_trades', event_data)

        # In a real-time simulation, you might add a small delay here
        # For a fast backtest, we'll publish as quickly as possible

    print("Finished replaying historical data.")

    # --- The rest of the system (resampler, feature calculator, etc.) ---
    # --- will now process this replayed data, and the orchestrator   ---
    # --- will execute trades against the portfolio.                  ---
    # --- After the replay is done, we can query the database to      ---
    # --- see the final state of the portfolio and calculate the KPIs. ---

if __name__ == '__main__':
    # This assumes that all the other services are running and listening to the event bus.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    run_event_driven_backtest('BTC/USDT', start_date.isoformat(), end_date.isoformat())
