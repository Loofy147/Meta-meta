import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

class EventPublisher:
    """
    A class for publishing events to a Redis Stream.
    """
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )

    def publish(self, stream_name, event_data):
        """
        Publishes an event to the specified Redis Stream.
        """
        try:
            # The event data should be a dictionary of strings
            stringified_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in event_data.items()}
            self.redis_client.xadd(stream_name, stringified_data)
            print(f"Published event to '{stream_name}': {event_data}")
        except Exception as e:
            print(f"An error occurred while publishing to Redis Stream '{stream_name}': {e}")

if __name__ == '__main__':
    # Example Usage
    publisher = EventPublisher()

    # Example trade event
    trade_event = {
        'symbol': 'BTC/USDT',
        'price': 60000.0,
        'amount': 0.1,
        'side': 'buy'
    }
    publisher.publish('raw_trades', trade_event)

    # Example signal event
    signal_event = {
        "asset": "BTC/USDT",
        "direction": "buy",
        "confidence": 0.75
    }
    publisher.publish('aggregated_signals', signal_event)
