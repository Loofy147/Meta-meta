"""
Arbitrage Scanner Service

This service runs in the background to continuously scan for cross-exchange
arbitrage opportunities.
"""

import asyncio
import sys
import os
import redis
from typing import List

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbitrage_detector import ArbitrageOrchestrator, ExchangeData

class ArbitrageIntegration:
    """
    Integrates Arbitrage Detection into the system.

    Runs as a separate service that publishes arbitrage opportunities.
    """

    def __init__(self):
        self.orchestrator = ArbitrageOrchestrator()
        self.ExchangeData = ExchangeData
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379, db=0
        )

    async def run_scanner(self):
        """
        Continuously scans for arbitrage opportunities.
        """
        print("Starting arbitrage scanner...")

        while True:
            try:
                exchange_data = await self._collect_exchange_data()

                opportunities = self.orchestrator.scan_all_opportunities(exchange_data)

                for opp in opportunities:
                    if opp.confidence > 0.7 and opp.profit_pct > 0.5:
                        self.orchestrator.publish_opportunity(opp)
                        print(f"Arbitrage opportunity: {opp.symbol} | "
                              f"{opp.profit_pct:.2f}% | "
                              f"Confidence: {opp.confidence:.2f}")

                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in arbitrage scanner: {e}")
                await asyncio.sleep(5)

    async def _collect_exchange_data(self) -> List:
        """Collects orderbook data from all monitored exchanges"""
        return []


async def run_arbitrage_scanner():
    """
    Continuously scans for arbitrage opportunities.
    """
    arbitrage_integration = ArbitrageIntegration()
    await arbitrage_integration.run_scanner()

if __name__ == '__main__':
    print("Starting Arbitrage Scanner Service...")
    asyncio.run(run_arbitrage_scanner())
