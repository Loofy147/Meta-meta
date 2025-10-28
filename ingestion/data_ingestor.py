import ccxt.pro
import asyncio
import json
import os
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher

load_dotenv()

async def watch_and_publish_trades(exchange, symbol, publisher):
    """Watches trades for a single symbol and publishes them to the event bus."""
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                event_data = {
                    'symbol': trade['symbol'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'timestamp': trade['timestamp']
                }
                publisher.publish('raw_trades', event_data)
        except Exception as e:
            print(f"An error occurred while watching {symbol}: {e}")
            break

async def main():
    """Main function to run the data ingestion."""
    with open('config/main.json', 'r') as f:
        config = json.load(f)['ingestion']

    exchange_name = config.get('exchange', 'kucoin')
    symbols = config.get('symbols', ['BTC/USDT'])

    try:
        exchange_class = getattr(ccxt.pro, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' not found.")
        return

    publisher = EventPublisher()
    tasks = [watch_and_publish_trades(exchange, symbol, publisher) for symbol in symbols]

    print(f"Starting data ingestion for {len(symbols)} symbols on {exchange_name}...")
    await asyncio.gather(*tasks)

    await exchange.close()
    print("Data ingestion stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nData ingestion stopped by user.")
