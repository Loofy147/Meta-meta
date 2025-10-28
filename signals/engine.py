from signals.strategies import rsi, macd, sentiment
import json
import os

def generate_signals(symbol='BTC/USDT'):
    """
    Generates signals from all available strategies.
    """
    strategies = {
        'rsi': rsi.generate_signal,
        'macd': macd.generate_signal,
        'sentiment': sentiment.generate_signal,
    }

    signals = []
    # Load strategy configuration
    with open('config/main.json', 'r') as f:
        config = json.load(f)['strategies']

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
    with open('config/main.json', 'r') as f:
        ingestion_config = json.load(f)['ingestion']

    all_signals = []
    for symbol in ingestion_config['symbols']:
        all_signals.extend(generate_signals(symbol))

    print("Generated Signals:")
    print(json.dumps(all_signals, indent=4))
