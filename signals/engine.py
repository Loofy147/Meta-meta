from signals.strategies import rsi, macd
import json

def generate_signals(symbol='BTC/USDT'):
    """
    Generates signals from all available strategies.
    """
    strategies = {
        'rsi': rsi.generate_signal,
        'macd': macd.generate_signal,
    }

    signals = []
    for name, generate_func in strategies.items():
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
    # Example of generating signals for all symbols in the config
    with open('config/main.json', 'r') as f:
        config = json.load(f)['ingestion']

    all_signals = []
    for symbol in config['symbols']:
        all_signals.extend(generate_signals(symbol))

    print("Generated Signals:")
    print(json.dumps(all_signals, indent=4))
