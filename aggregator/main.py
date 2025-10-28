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
from event_bus.publisher import EventPublisher
from config.manager import get_config

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_strategy_weights():
    """
    Fetches strategy performance and calculates dynamic weights.
    """
    conn = get_db_connection()
    try:
        query = "SELECT strategy_name, hit_rate FROM strategy_performance;"
        df = pd.read_sql(query, conn)

        if df.empty:
            return {}

        weights = {row['strategy_name']: row['hit_rate'] for _, row in df.iterrows()}
        total_hit_rate = sum(weights.values())
        if total_hit_rate > 0:
            for name in weights:
                weights[name] /= total_hit_rate
        return weights
    finally:
        conn.close()

def aggregate_signals(symbol, strategy_weights):
    """
    Aggregates signals for a single symbol using adaptive weights.
    """
    signals = generate_signals(symbol)

    if not signals:
        final_direction = 'hold'
        final_confidence = 0.0
    else:
        directions = {s['direction'] for s in signals}
        if 'buy' in directions and 'sell' in directions:
            final_direction = 'hold'
            final_confidence = 0.0
        else:
            weighted_confidence = 0
            total_weight = 0
            for s in signals:
                weight = strategy_weights.get(s['strategy'], 0.1)
                weighted_confidence += s['confidence'] * weight
                total_weight += weight

            final_direction = signals[0]['direction']
            final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

    return {
        "signal_id": str(uuid.uuid4()),
        "asset": symbol,
        "direction": final_direction,
        "confidence": round(final_confidence, 4),
        "origin": "adaptive_aggregator",
        "meta": {"contributing_signals": signals},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def aggregate_and_publish_signals():
    """
    Aggregates signals for all configured symbols and publishes them to the event bus.
    """
    config = get_config()
    ingestion_config = config['ingestion']

    publisher = EventPublisher()
    strategy_weights = get_strategy_weights()

    for symbol in ingestion_config['symbols']:
        final_signal = aggregate_signals(symbol, strategy_weights)
        publisher.publish('aggregated_signals', final_signal)

if __name__ == "__main__":
    import time
    print("Starting the adaptive signal aggregator...")
    while True:
        aggregate_and_publish_signals()
        time.sleep(60)
