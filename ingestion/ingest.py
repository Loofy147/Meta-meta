"""A service for ingesting real-time market data via WebSockets.

This service is the primary entry point for all live market data into the
trading system. It connects to a WebSocket-based market data provider,
subscribes to trade streams for a configurable list of symbols, and publishes
the received raw trade data onto the 'raw_trades' Redis Stream for downstream
processing.

Note: The current implementation uses a public echo WebSocket server for
demonstration purposes and simulates trade data. To connect to a real data
source, the `subscribe_to_market_data` function must be modified.
"""

import asyncio
import websockets
import json
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, Any, List

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_bus.publisher import EventPublisher
from config.manager import get_config

load_dotenv()

async def subscribe_to_market_data(publisher: EventPublisher, symbols: List[str]):
    """Connects to a WebSocket feed and streams trade data.

    This function establishes a connection to a market data WebSocket endpoint.
    Upon connection, it would typically send a subscription message for the
    specified symbols. It then enters an infinite loop, listening for incoming
    messages, formatting them into a standard trade event format, and publishing
    them using the provided EventPublisher.

    Args:
        publisher: An initialized `EventPublisher` instance for sending
            events to the Redis Stream.
        symbols: A list of asset symbols to subscribe to (e.g., ["BTC/USDT"]).
    """
    # Note: This URI is a public echo server for testing. Replace with a real
    # market data endpoint (e.g., from Binance, Polygon.io, or Alpaca).
    uri = "wss://echo.websocket.events"

    while True:  # Outer loop for handling reconnections
        try:
            async with websockets.connect(uri) as websocket:
                # In a real-world scenario, you would send a subscription message here.
                # Example:
                # sub_message = json.dumps({"method": "SUBSCRIBE", "params": symbols})
                # await websocket.send(sub_message)
                print(f"Successfully connected and subscribed to data for: {symbols}")

                while True:
                    # The following block simulates receiving trade data.
                    # In a real implementation, you would replace this with:
                    # message = await websocket.recv()
                    # trade_data = json.loads(message)
                    # And then parse `trade_data` to create `trade_event`.
                    for symbol in symbols:
                        trade_event = {
                            'symbol': symbol,
                            # Simulate minor price fluctuations around a base
                            'price': 50000.0 + (os.urandom(1)[0] / 255.0 - 0.5) * 100,
                            'amount': os.urandom(1)[0] / 255.0,
                            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                            'side': 'buy' if os.urandom(1)[0] > 128 else 'sell'
                        }
                        publisher.publish('raw_trades', trade_event)
                        print(f"Published trade for {symbol}: {trade_event['price']:.2f}")
                    await asyncio.sleep(1)

        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            print(f"WebSocket connection error: {e}. Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred in WebSocket handler: {e}")
            await asyncio.sleep(10) # Wait longer for unexpected errors


def run_ingestion_service():
    """Initializes and runs the data ingestion service.

    This function is the main entry point for the service. It performs the
    following steps:
    1. Initializes the `EventPublisher` for communication with the event bus.
    2. Fetches the system configuration to get the list of symbols to track.
    3. If symbols are configured, it starts the asynchronous WebSocket
       subscription client.
    4. If no symbols are configured, it prints a message and exits gracefully.
    """
    print("Starting data ingestion service...")
    try:
        publisher = EventPublisher()
        config = get_config()
        symbols = config.get('ingestion', {}).get('symbols', [])

        if not symbols:
            print("Warning: No symbols configured for ingestion in 'system_parameters' table. Service is idle.")
            return

        asyncio.run(subscribe_to_market_data(publisher, symbols))

    except Exception as e:
        print(f"A critical error occurred while starting the ingestion service: {e}")

if __name__ == "__main__":
    run_ingestion_service()
