"""
Event Bus Publisher

This module provides a simple, reusable class for publishing events to the
system's event bus, which is implemented using Redis Streams. It abstracts away
the details of the Redis connection and data serialization.
"""

import redis
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class EventPublisher:
    """
    A class for publishing events to a Redis Stream.

    This publisher handles the connection to Redis and ensures that event data
    is correctly serialized before being published.
    """
    def __init__(self):
        """
        Initializes the EventPublisher and connects to the Redis server.

        Connection parameters are sourced from environment variables.
        """
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=False # Store bytes to handle JSON correctly
            )
            self.redis_client.ping() # Verify connection
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def publish(self, stream_name: str, event_data: Dict[str, Any]) -> None:
        """
        Publishes an event to the specified Redis Stream.

        The event data dictionary is serialized into a format suitable for Redis
        Streams (a flat dictionary of bytes). Nested dictionaries and lists are
        JSON-encoded.

        Args:
            stream_name (str): The name of the Redis Stream to publish to.
            event_data (Dict[str, Any]): The event data to publish.
        """
        try:
            # Redis streams require a flat dictionary of bytes or strings.
            # We serialize complex types (dicts, lists) as JSON strings.
            serialized_data = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in event_data.items()
            }
            self.redis_client.xadd(stream_name, serialized_data)
            # print(f"Published event to '{stream_name}': {event_data}")
        except Exception as e:
            print(f"An error occurred while publishing to Redis Stream '{stream_name}': {e}")

if __name__ == '__main__':
    # Example Usage:
    # This demonstrates how to instantiate the publisher and send different types of events.

    try:
        publisher = EventPublisher()
        print("Successfully connected to Redis.")

        # Example 1: A raw trade event
        trade_event = {
            'symbol': 'BTC/USDT',
            'price': 60000.0,
            'amount': 0.1,
            'timestamp': 1672531200000, # Milliseconds for consistency
            'side': 'buy'
        }
        publisher.publish('raw_trades', trade_event)
        print("Published sample trade event to 'raw_trades'.")

        # Example 2: An aggregated signal event
        signal_event = {
            "signal_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            "asset": "BTC/USDT",
            "direction": "buy",
            "confidence": 0.75,
            "meta": {"contributing_signals": ["rsi", "macd"]}
        }
        publisher.publish('aggregated_signals', signal_event)
        print("Published sample signal event to 'aggregated_signals'.")

    except ConnectionError as e:
        print(e)
