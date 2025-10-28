import sys
import os
import uuid
import json
from datetime import datetime, timezone
import pandas as pd
import psycopg2

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.engine import generate_signals

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_higher_timeframe_trend(symbol, timeframe='1h'):
    """
    Determines the trend on a higher timeframe by checking the MACD.
    Returns 'buy' for uptrend, 'sell' for downtrend, 'hold' otherwise.
    """
    conn = get_db_connection()
    try:
        query = f"SELECT macd, macds FROM features_{timeframe} WHERE symbol = %s ORDER BY time DESC LIMIT 1;"
        df = pd.read_sql(query, conn, params=(symbol,))
        if not df.empty and df['macd'].iloc[0] > df['macds'].iloc[0]:
            return 'buy'
        elif not df.empty and df['macd'].iloc[0] < df['macds'].iloc[0]:
            return 'sell'
        return 'hold'
    finally:
        conn.close()

def aggregate_signals_for_symbol(symbol='BTC/USDT'):
    """
    Aggregates signals with a multi-timeframe consensus score.
    """
    # 1. Get signals from the primary (1m) timeframe
    signals_1m = generate_signals(symbol)

    if not signals_1m:
        final_direction = 'hold'
        final_confidence = 0.0
    else:
        # 2. Check for conflicts on the primary timeframe
        directions_1m = {s['direction'] for s in signals_1m}
        if 'buy' in directions_1m and 'sell' in directions_1m:
            final_direction = 'hold'
            final_confidence = 0.0
        else:
            # 3. Get trends from higher timeframes
            trend_15m = get_higher_timeframe_trend(symbol, '15m')
            trend_1h = get_higher_timeframe_trend(symbol, '1h')

            primary_direction = signals_1m[0]['direction']

            # 4. Calculate consensus score
            weights = {'1m': 0.4, '15m': 0.3, '1h': 0.3}
            score = 0

            # Add score for primary signal
            if primary_direction != 'hold':
                score += weights['1m']

            # Add score for 15m trend alignment
            if trend_15m == primary_direction:
                score += weights['15m']

            # Add score for 1h trend alignment
            if trend_1h == primary_direction:
                score += weights['1h']

            final_direction = primary_direction
            final_confidence = score

    final_signal = {
        "signal_id": str(uuid.uuid4()),
        "asset": symbol,
        "direction": final_direction,
        "confidence": round(final_confidence, 4),
        "timeframe": "1m_consensus",
        "origin": "multi_timeframe_aggregator",
        "meta": {"contributing_signals_1m": signals_1m},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return final_signal

if __name__ == "__main__":
    with open('config/main.json', 'r') as f:
        config = json.load(f)['ingestion']

    for symbol in config['symbols']:
        print(json.dumps(aggregate_signals_for_symbol(symbol), indent=4))
