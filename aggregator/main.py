import sys
import os
import uuid
import json
from datetime import datetime, timezone

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.engine import generate_signals

def aggregate_signals_for_symbol(symbol='BTC/USDT'):
    """
    Aggregates signals from all strategies for a single symbol
    into a final trading decision.
    """
    signals = generate_signals(symbol)

    if not signals:
        final_direction = 'hold'
        final_confidence = 0.0
    else:
        # Simple conflict resolution: if there are opposing signals, hold.
        directions = {s['direction'] for s in signals}
        if 'buy' in directions and 'sell' in directions:
            final_direction = 'hold'
            final_confidence = 0.0 # Confidence is zero due to conflict
        else:
            # Average the confidences of the signals
            final_direction = signals[0]['direction']
            total_confidence = sum(s['confidence'] for s in signals)
            final_confidence = total_confidence / len(signals)

    final_signal = {
        "signal_id": str(uuid.uuid4()),
        "asset": symbol,
        "direction": final_direction,
        "confidence": round(final_confidence, 4),
        "timeframe": "1m",
        "origin": "multi_strategy_aggregator",
        "meta": {"contributing_signals": signals},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return final_signal

if __name__ == "__main__":
    # Example of aggregating signals for all symbols in the config
    with open('config/main.json', 'r') as f:
        config = json.load(f)['ingestion']

    aggregated_signals = []
    for symbol in config['symbols']:
        aggregated_signals.append(aggregate_signals_for_symbol(symbol))

    print("Aggregated Signals:")
    print(json.dumps(aggregated_signals, indent=4))
