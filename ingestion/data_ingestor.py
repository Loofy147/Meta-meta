"""Ingests real-time trade data from a cryptocurrency exchange using ccxt.pro.

This module provides a robust data ingestion service that connects to a specified
cryptocurrency exchange via WebSockets, subscribes to the public trades stream
for multiple symbols, and publishes the incoming data to the 'raw_trades'
Redis Stream.

It is designed to be configurable through a JSON file (`config/main.json`) and
relies on the `ccxt.pro` library, which provides a unified API for interacting
with the WebSocket feeds of numerous exchanges.
"""

import ccxt.pro
import asyncio
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Add the parent directory to the path to allow imports from sibling modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

load_dotenv()

async def watch_and_publish_trades(exchange: ccxt.pro.Exchange, symbol: str, publisher: EventPublisher):
    """Watches for new trades for a symbol and publishes them.

    This asynchronous coroutine enters an infinite loop to continuously listen
    for new trade data from the exchange's WebSocket feed for a single symbol.
    When a batch of trades is received, it iterates through them, formats them
    into a standardized dictionary format, and publishes each trade to the
    'raw_trades' Redis Stream.

    Args:
        exchange: An initialized `ccxt.pro` exchange instance.
        symbol: The market symbol to watch for trades (e.g., 'BTC/USDT').
        publisher: An `EventPublisher` instance to publish the trades.
    """
    while True:
        try:
            trades: List[Dict[str, Any]] = await exchange.watch_trades(symbol)
            for trade in trades:
                event_data = {
                    'symbol': trade['symbol'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'timestamp': trade['timestamp']  # ccxt provides ms timestamp
                }
                publisher.publish('raw_trades', event_data)
                print(f"Published trade for {symbol}: {trade['price']:.2f} @ {trade['timestamp']}")
        except Exception as e:
            print(f"An error occurred while watching trades for {symbol}: {e}")
            # In a production system, you might add a delay and attempt to reconnect.
            break

async def main():
    """Initializes the exchange, creates tasks, and runs the data ingestion.

    This is the main entry point for the service. It reads the configuration
    from `config/main.json` to determine the exchange and symbols. It then
    instantiates the appropriate `ccxt.pro` exchange class, creates an
    `EventPublisher`, and spawns an asynchronous task for each symbol using
    `watch_and_publish_trades`. It runs these tasks concurrently until the
    program is interrupted.
    """
    try:
        with open('config/main.json', 'r') as f:
            config = json.load(f).get('ingestion', {})
    except FileNotFoundError:
        print("Error: `config/main.json` not found. Please create it.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode `config/main.json`.")
        return

    exchange_name = config.get('exchange', 'kucoin')
    symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT'])

    try:
        exchange_class = getattr(ccxt.pro, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' is not supported by ccxt.pro.")
        return

    publisher = EventPublisher()
    tasks = [watch_and_publish_trades(exchange, symbol, publisher) for symbol in symbols]

    print(f"Starting data ingestion for {len(symbols)} symbol(s) on '{exchange_name}'...")
    await asyncio.gather(*tasks)

    # This part will only be reached if the loops in watch_and_publish_trades break.
    await exchange.close()
    print("Data ingestion stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nData ingestion stopped by user.")
