"""
Order Book Analytics Signal Strategy

This strategy uses market microstructure data from the order book to generate
trading signals. It leverages the OrderBookStrategyIntegration class to
interface with the order book analytics engine.
"""

import sys
import os
import json
import psycopg2
import redis
import pandas as pd
from typing import Dict, Optional


# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orderbook_analytics import OrderBookAnalyzer, OrderBookSnapshot

class OrderBookStrategyIntegration:
    """
    Integrates Order Book Analytics as a new signal strategy.

    Adds microstructure signals to signals/strategies/ directory.
    """

    def __init__(self):
        from orderbook_analytics import OrderBookAnalyzer, OrderBookSnapshot

        self.analyzer = OrderBookAnalyzer(window_size=50, depth_levels=10)
        self.OrderBookSnapshot = OrderBookSnapshot
        self.db_conn = self._get_db_connection()
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379, db=0
        )

    def _get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )

    def generate_signal(self, symbol: str) -> tuple[str, float]:
        """
        Generates trading signal from order book analysis.

        This function signature matches existing strategies for easy integration.
        """
        # Fetch latest order book snapshot from Redis stream
        snapshot_data = self._fetch_latest_orderbook(symbol)

        if not snapshot_data:
            return 'hold', 0.0

        # Create snapshot object
        snapshot = self.OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bids=snapshot_data['bids'],
            asks=snapshot_data['asks'],
            mid_price=snapshot_data['mid_price'],
            spread=snapshot_data['spread']
        )

        # Process snapshot and get metrics
        metrics = self.analyzer.process_snapshot(snapshot)

        # Generate signal
        direction, confidence = self.analyzer.generate_alpha_signal(metrics)

        # Log metrics for monitoring
        self._log_orderbook_metrics(symbol, metrics)

        return direction, confidence

    def _fetch_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetches latest order book snapshot from Redis"""
        try:
            # In production, this would read from a dedicated orderbook stream
            # For now, we'll construct from latest trades
            messages = self.redis_client.xrevrange(
                f'orderbook_{symbol}',
                count=1
            )

            if not messages:
                return None

            _, data = messages[0]
            return {
                'bids': json.loads(data[b'bids'].decode()),
                'asks': json.loads(data[b'asks'].decode()),
                'mid_price': float(data[b'mid_price'].decode()),
                'spread': float(data[b'spread'].decode())
            }
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return None

    def _log_orderbook_metrics(self, symbol: str, metrics):
        """Logs order flow metrics to database for analysis"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO orderbook_metrics
                (time, symbol, order_imbalance, vpin, pressure_index, liquidity_score, toxicity_score)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s);
            """, (
                symbol,
                metrics.order_imbalance,
                metrics.vpin,
                metrics.pressure_index,
                metrics.liquidity_score,
                metrics.toxicity_score
            ))
            self.db_conn.commit()
        except Exception as e:
            print(f"Error logging orderbook metrics: {e}")
            self.db_conn.rollback()

class OrderBookStrategy:
    """
    A signal strategy based on order book analytics.
    """
    def __init__(self):
        self.integration = OrderBookStrategyIntegration()

    def generate_signal(self, symbol: str) -> tuple[str, float]:
        """
        Generates a trading signal for the given symbol.

        Args:
            symbol (str): The symbol to generate a signal for.

        Returns:
            tuple[str, float]: A tuple containing the signal direction ('buy', 'sell', or 'hold')
                               and the confidence level (0.0 to 1.0).
        """
        return self.integration.generate_signal(symbol)

if __name__ == '__main__':
    # Example Usage:
    print("--- Order Book Strategy Example ---")
    strategy = OrderBookStrategy()
    signal, confidence = strategy.generate_signal('BTC/USDT')
    print(f"Generated Signal for BTC/USDT: {signal.upper()} (Confidence: {confidence:.2f})")
