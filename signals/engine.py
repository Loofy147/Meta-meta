from signals.strategies import rsi, macd, sentiment, ml_strategy
import json
import os

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import get_config

def generate_signals(symbol='BTC/USDT'):
    """
    Generates signals from all available strategies.
    """
    strategies = {
        'rsi': rsi.generate_signal,
        'macd': macd.generate_signal,
        'sentiment': sentiment.generate_signal,
        'ml': ml_strategy.generate_signal,
    }

    signals = []
    # Load strategy configuration from the database
    config = get_config()['strategies']

    for name, generate_func in strategies.items():
        if config.get(name, {}).get('enabled', True):
            direction, confidence = generate_func(symbol)
            if direction != 'hold':
                signals.append({
                    'strategy': name,
                    'direction': direction,
                    'confidence': confidence,
                    'symbol': symbol
                })

    return signals

if __name__ == "__main__":
    # This will now use the config from the database to determine which symbols to process
    # For simplicity, this example will just use the default
    all_signals = generate_signals('BTC/USDT')

    print("Generated Signals:")
    print(json.dumps(all_signals, indent=4))
