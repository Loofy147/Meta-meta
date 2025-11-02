"""
The Adaptive Signal Aggregator consumes raw signals from various strategies,
weights them based on historical performance, and produces a final, aggregated
trading signal.
"""

import sys
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd
import psycopg2
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.engine import generate_signals
from event_bus.publisher import EventPublisher
from config.manager import get_config

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

def get_strategy_weights() -> Dict[str, float]:
    """
    Fetches strategy performance from the database and calculates dynamic weights.

    The weight of each strategy is proportional to its historical hit rate.
    Strategies with no performance data are implicitly given a weight of 0.

    Returns:
        Dict[str, float]: A dictionary mapping strategy names to their calculated weights.
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
            # Normalize weights so they sum to 1
            for name in weights:
                weights[name] /= total_hit_rate
        return weights
    finally:
        conn.close()

def aggregate_signals(symbol: str, strategy_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Generates and aggregates signals for a single symbol using adaptive weights.

    This function fetches raw signals from the signal engine, resolves conflicts
    (e.g., simultaneous buy and sell signals), and calculates a final,
    confidence-weighted signal.

    Args:
        symbol (str): The trading symbol to generate signals for (e.g., 'BTC/USDT').
        strategy_weights (Dict[str, float]): A dictionary of weights for each strategy.

    Returns:
        Dict[str, Any]: The final aggregated signal, including direction, confidence,
                        and metadata about the contributing raw signals.
    """
    signals = generate_signals(symbol)

    if not signals:
        final_direction = 'hold'
        final_confidence = 0.0
    else:
        directions = {s['direction'] for s in signals}
        if 'buy' in directions and 'sell' in directions:
            # Conflict resolution: if strategies disagree, produce a neutral signal.
            final_direction = 'hold'
            final_confidence = 0.0
        else:
            weighted_confidence = 0.0
            total_weight = 0.0
            for s in signals:
                # Use a default weight of 0.1 for strategies without historical data
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

def aggregate_and_publish_signals() -> None:
    """
    Periodically aggregates signals for all configured symbols and publishes them.

    This function acts as the main loop of the aggregator service.
    """
    config = get_config()
    ingestion_config = config.get('ingestion', {})
    symbols = ingestion_config.get('symbols', [])

    if not symbols:
        print("No symbols configured for ingestion. Aggregator is idle.")
        return

    publisher = EventPublisher()
    strategy_weights = get_strategy_weights()

    print(f"Aggregating signals for symbols: {symbols}")
    for symbol in symbols:
        final_signal = aggregate_signals(symbol, strategy_weights)
        if final_signal['direction'] != 'hold':
            publisher.publish('aggregated_signals', final_signal)
            print(f"Published signal for {symbol}: {final_signal['direction']} (Confidence: {final_signal['confidence']})")

if __name__ == "__main__":
    import time
    print("Starting the adaptive signal aggregator...")
    while True:
        aggregate_and_publish_signals()
        print("Sleeping for 60 seconds...")
        time.sleep(60)
