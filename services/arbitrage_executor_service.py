"""
Arbitrage Executor Service

This service listens to the 'arbitrage_opportunities' stream on the event bus
and executes the identified arbitrage trades.
"""

import asyncio
import sys
import os
import json
import redis

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.alpaca_broker import AlpacaBroker

class ArbitrageExecutor:
    """
    Executes arbitrage opportunities.
    """
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379,
            db=0,
            decode_responses=True
        )
        self.broker = AlpacaBroker()

    async def listen_for_opportunities(self):
        """
        Listens for arbitrage opportunities on the Redis stream.
        """
        stream_name = 'arbitrage_opportunities'
        last_id = '0'

        print(f"Listening for arbitrage opportunities on stream '{stream_name}'...")

        while True:
            try:
                response = self.redis_client.xread(
                    {stream_name: last_id}, count=1, block=0
                )

                if response:
                    for _, messages in response:
                        for message_id, data in messages:
                            print(f"Received arbitrage opportunity: {data}")
                            await self.execute_opportunity(data)
                            last_id = message_id

            except Exception as e:
                print(f"Error reading from Redis stream: {e}")
                await asyncio.sleep(5)

    async def execute_opportunity(self, opportunity_data: dict):
        """
        Executes an arbitrage opportunity.
        """
        try:
            # In a real system, you would have separate broker instances for each exchange.
            # Here, we'll simulate the execution with a single broker.

            symbol = opportunity_data['symbol']
            buy_price = float(opportunity_data['buy_price'])
            sell_price = float(opportunity_data['sell_price'])
            volume_limit = float(opportunity_data['volume_limit'])

            # Simple execution logic
            # Buy on the 'buy_exchange'
            self.broker.place_order(
                symbol, 'buy', volume_limit, 'market'
            )

            # Sell on the 'sell_exchange'
            self.broker.place_order(
                symbol, 'sell', volume_limit, 'market'
            )

            print(f"Executed arbitrage trade for {symbol}: BUY {volume_limit} @ {buy_price}, SELL {volume_limit} @ {sell_price}")

        except Exception as e:
            print(f"Error executing arbitrage opportunity: {e}")


async def run_arbitrage_executor():
    """
    Runs the arbitrage executor service.
    """
    executor = ArbitrageExecutor()
    await executor.listen_for_opportunities()

if __name__ == '__main__':
    print("Starting Arbitrage Executor Service...")
    asyncio.run(run_arbitrage_executor())
