import sys
import os
import uuid
from datetime import datetime, timezone

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.engine import generate_signal

def aggregate_signal():
    raw_signal = generate_signal()

    # The confidence for this basic signal is static for now.
    confidence = 0.5 if raw_signal != "hold" else 0.0

    signal = {
        "signal_id": str(uuid.uuid4()),
        "asset": "BTC/USDT",
        "direction": raw_signal,
        "confidence": confidence,
        "timeframe": "1m",  # This is a placeholder for now
        "origin": "sma_crossover_engine",
        "meta": {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return signal

if __name__ == "__main__":
    final_signal = aggregate_signal()
    print(final_signal)
