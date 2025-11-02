"""
Data Ingestion Service

This service is responsible for connecting to real-time market data sources,
subscribing to trade or quote streams, and publishing the raw data onto the
system's event bus ('raw_trades' Redis Stream).

It is the primary entry point for all live market data into the system.
"""

import asyncio
import websockets
import json
import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher
from config.manager import get_config

load_dotenv()

async def subscribe_to_market_data(publisher: EventPublisher, symbols: list):
    """
    Connects to a WebSocket-based market data feed and subscribes to trades.

    Args:
        publisher (EventPublisher): An instance of the event publisher.
        symbols (list): A list of symbols to subscribe to.
    """
    # This is a placeholder for a real WebSocket endpoint (e.g., from Binance, Polygon)
    # For this example, we'll simulate a simple feed.
    uri = "wss://echo.websocket.events" # A public echo server for testing

    async with websockets.connect(uri) as websocket:
        # In a real implementation, you would send a subscription message here.
        # e.g., await websocket.send(json.dumps({"method": "SUBSCRIBE", "params": symbols}))
        print(f"Subscribed to trade data for symbols: {symbols}")

        while True:
            try:
                # In a real implementation, you would receive real trade data here.
                # We will simulate receiving a trade every second.
                for symbol in symbols:
                    trade_event = {
                        'symbol': symbol,
                        'price': 50000.0 + (os.urandom(1)[0] / 255.0 - 0.5), # Simulate price fluctuation
                        'amount': os.urandom(1)[0] / 255.0,
                        'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000)
                    }
                    publisher.publish('raw_trades', trade_event)
                await asyncio.sleep(1)

            except websockets.ConnectionClosed:
                print("WebSocket connection closed. Reconnecting...")
                await asyncio.sleep(5)
                # Reconnect logic would be here
                break

def run_ingestion_service():
    """
    The main entry point for the data ingestion service.
    """
    print("Starting data ingestion service...")
    try:
        publisher = EventPublisher()
        config = get_config()
        symbols = config.get('ingestion', {}).get('symbols', [])

        if not symbols:
            print("No symbols configured for ingestion. Service is idle.")
            return

        asyncio.run(subscribe_to_market_data(publisher, symbols))

    except Exception as e:
        print(f"An error occurred in the ingestion service: {e}")

if __name__ == "__main__":
    run_ingestion_service()
