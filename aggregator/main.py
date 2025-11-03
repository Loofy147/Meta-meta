"""An adaptive signal aggregator that weights strategies by performance.

This module is responsible for consuming raw signals from various trading
strategies, intelligently weighting them based on their historical performance
(hit rate), and producing a single, high-confidence, aggregated trading signal.

The core logic is as follows:
1.  **Fetch Performance**: Retrieve the historical hit rate of each strategy
    from the `strategy_performance` database table.
2.  **Calculate Weights**: Normalize the hit rates to create a set of weights
    that sum to 1. This ensures that more successful strategies have a greater
    influence on the final signal.
3.  **Generate Raw Signals**: Invoke the signal engine to get the latest raw
    signals from all enabled strategies.
4.  **Aggregate**: Calculate a final, weighted-average confidence score. If
    strategies produce conflicting signals (both 'buy' and 'sell'), it resolves
    this by issuing a neutral 'hold' signal.
5.  **Publish**: The final aggregated signal is published to the
    `aggregated_signals` Redis Stream for the orchestrator to consume.
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

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.engine import generate_signals
from event_bus.publisher import EventPublisher
from config.manager import get_config

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

def get_strategy_weights() -> Dict[str, float]:
    """Fetches performance data and calculates normalized strategy weights.

    This function queries the `strategy_performance` table to get the historical
    hit rate of each strategy. The hit rates are then normalized so that they
    sum to 1, effectively representing each strategy's contribution to the
    total confidence.

    Returns:
        A dictionary mapping strategy names to their normalized performance
        weights (e.g., {'rsi': 0.6, 'macd': 0.4}). Returns an empty
        dictionary if no performance data is available.
    """
    conn = get_db_connection()
    try:
        query = "SELECT strategy_name, hit_rate FROM strategy_performance WHERE trade_count > 10;"
        df = pd.read_sql(query, conn)

        if df.empty:
            return {}

        weights = {row['strategy_name']: row['hit_rate'] for _, row in df.iterrows()}
        total_hit_rate = sum(weights.values())

        if total_hit_rate > 0:
            # Normalize weights to ensure they sum to 1.
            for name in weights:
                weights[name] /= total_hit_rate
        return weights
    finally:
        conn.close()

def aggregate_signals(symbol: str, strategy_weights: Dict[str, float]) -> Dict[str, Any]:
    """Generates and aggregates signals for a symbol using adaptive weights.

    This function first fetches all raw signals from the signal engine. It then
    applies the performance-based weights to the confidence score of each
    signal. A key feature is its conflict resolution: if both 'buy' and 'sell'
    signals are present, it defaults to a neutral 'hold' signal to avoid
    indecisive actions.

    Args:
        symbol: The trading symbol to generate signals for (e.g., 'BTC/USDT').
        strategy_weights: A dictionary of normalized weights for each strategy.

    Returns:
        A dictionary representing the final aggregated signal, complete with a
        unique ID, direction, confidence, and metadata about the raw signals
        that contributed to it.
    """
    signals = generate_signals(symbol)

    if not signals:
        return _create_final_signal(symbol, 'hold', 0.0, [])

    directions = {s['direction'] for s in signals}
    if 'buy' in directions and 'sell' in directions:
        # Conflict resolution: if strategies disagree, produce a neutral signal.
        return _create_final_signal(symbol, 'hold', 0.0, signals)

    # If all signals agree on the direction, calculate the weighted confidence.
    weighted_confidence = 0.0
    total_weight = 0.0
    for s in signals:
        # Use a small default weight for new strategies without historical data.
        weight = strategy_weights.get(s['strategy'], 0.1)
        weighted_confidence += s['confidence'] * weight
        total_weight += weight

    final_direction = signals[0]['direction']
    # Normalize the final confidence score by the total weight used.
    final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

    return _create_final_signal(symbol, final_direction, final_confidence, signals)

def _create_final_signal(symbol: str, direction: str, confidence: float, contributing: List[Dict]) -> Dict[str, Any]:
    """Helper function to format the final aggregated signal dictionary."""
    return {
        "signal_id": str(uuid.uuid4()),
        "asset": symbol,
        "direction": direction,
        "confidence": round(confidence, 4),
        "origin": "adaptive_aggregator",
        "meta": {"contributing_signals": contributing},
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
