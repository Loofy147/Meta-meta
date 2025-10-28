import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signal

PORTFOLIO_FILE = 'paper_trader/portfolio.json'

def get_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {'cash': 100000, 'btc': 0}
    with open(PORTFOLIO_FILE, 'r') as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def execute_trade(signal):
    portfolio = get_portfolio()

    # For now, we'll use a fixed trade size of 1 BTC
    trade_size_btc = 1

    # A real implementation would need to fetch the current price.
    # For this simulation, we'll use a placeholder price.
    current_price = 90000 # Lowered placeholder price

    if signal['direction'] == 'buy' and portfolio['cash'] >= current_price * trade_size_btc:
        portfolio['cash'] -= current_price * trade_size_btc
        portfolio['btc'] += trade_size_btc
        print(f"Executed BUY of {trade_size_btc} BTC at {current_price}")
    elif signal['direction'] == 'sell' and portfolio['btc'] >= trade_size_btc:
        portfolio['cash'] += current_price * trade_size_btc
        portfolio['btc'] -= trade_size_btc
        print(f"Executed SELL of {trade_size_btc} BTC at {current_price}")
    else:
        print("Holding position or insufficient funds.")

    save_portfolio(portfolio)

if __name__ == "__main__":
    signal = aggregate_signal()
    if signal['confidence'] > 0.4: # A simple threshold
        execute_trade(signal)
    else:
        print("Signal confidence too low, holding.")
