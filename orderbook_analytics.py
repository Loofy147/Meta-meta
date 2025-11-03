"""Provides an advanced engine for market microstructure analysis.

This module implements a sophisticated `OrderBookAnalyzer` class that processes
real-time Level 2 order book data to derive alpha-generating insights. It goes
beyond simple price and volume indicators to model the underlying dynamics of
supply and demand.

The key metrics calculated include:
- **Order Flow Imbalance**: Measures the net buying or selling pressure by
  comparing the volume on the bid and ask sides of the book.
- **VPIN (Volume-synchronized Probability of Informed Trading)**: An advanced
  indicator that estimates the probability of trading activity being driven by
  informed traders, which often precedes significant price moves.
- **Trade Flow Toxicity**: Measures the risk of adverse selection by analyzing
  the price impact of recent trades.
- **Liquidity Score**: A composite score that quantifies the current state of
  market liquidity based on depth, spread, and depth distribution.
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import psycopg2
from scipy.stats import zscore
import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OrderBookSnapshot:
    """Represents a single, point-in-time snapshot of an order book."""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # List of (price, size) tuples
    asks: List[Tuple[float, float]]  # List of (price, size) tuples
    mid_price: float
    spread: float


@dataclass
class OrderFlowMetrics:
    """A collection of advanced order flow and microstructure indicators."""
    order_imbalance: float
    vpin: float
    pressure_index: float
    liquidity_score: float
    toxicity_score: float


class OrderBookAnalyzer:
    """Analyzes real-time order book data to extract market microstructure signals.

    This class maintains a rolling history of order book snapshots and trades
    to calculate a variety of advanced metrics. These metrics can then be used
    to generate a trading signal based on the underlying order flow dynamics.
    """

    def __init__(self, window_size: int = 50, depth_levels: int = 10):
        """Initializes the OrderBookAnalyzer.

        Args:
            window_size: The number of recent order book snapshots to keep in
                the history for time-series calculations.
            depth_levels: The number of price levels on both the bid and ask
                sides to include in the analysis.
        """
        self.window_size = window_size
        self.depth_levels = depth_levels
        self.snapshot_history: deque = deque(maxlen=window_size)
        self.trade_history: deque = deque(maxlen=1000)

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )

    def process_snapshot(self, snapshot: OrderBookSnapshot) -> OrderFlowMetrics:
        """Processes a new snapshot and calculates all microstructure metrics.

        This is the main entry point for the analyzer. When a new order book
        update is received, this method is called to compute the latest set
        of order flow metrics.

        Args:
            snapshot: The latest `OrderBookSnapshot` to be processed.

        Returns:
            An `OrderFlowMetrics` object containing all the calculated indicators.
        """
        self.snapshot_history.append(snapshot)

        if len(self.snapshot_history) < 2:
            return self._default_metrics()

        return OrderFlowMetrics(
            order_imbalance=self._calculate_order_imbalance(snapshot),
            vpin=self._calculate_vpin(),
            pressure_index=self._calculate_pressure_index(),
            liquidity_score=self._calculate_liquidity_score(snapshot),
            toxicity_score=self._calculate_toxicity_score()
        )

    def _calculate_order_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """Calculates the volume-weighted order book imbalance.

        Formula: OI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)

        Returns a value in the range [-1, 1], where a positive value indicates
        stronger buying pressure and a negative value indicates stronger
        selling pressure.
        """
        bid_volume = sum(size for _, size in snapshot.bids[:self.depth_levels])
        ask_volume = sum(size for _, size in snapshot.asks[:self.depth_levels])

        total_volume = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0

    def _calculate_vpin(self) -> float:
        """Calculates the Volume-synchronized Probability of Informed Trading (VPIN).

        VPIN is a sophisticated measure that estimates the probability of
        informed traders being active in the market by analyzing the imbalance
        of buyer-initiated versus seller-initiated volume. A high VPIN value
        is often considered a leading indicator of increased volatility or a
        potential market reversal due to "toxic" order flow.
        """
        if len(self.trade_history) < 50:
            return 0.0

        trades = list(self.trade_history)[-50:]
        buy_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
        sell_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')

        total_volume = buy_volume + sell_volume
        return abs(buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

    def _calculate_pressure_index(self) -> float:
        """Calculates the net buying/selling pressure from recent trade flow.

        This method uses an exponentially weighted moving average of trade
        volumes, signed by their direction (buy/sell), to give more weight to
        the most recent trades. The result is normalized to the range [-1, 1].
        """
        if len(self.trade_history) < 10:
            return 0.0

        recent_trades = list(self.trade_history)[-50:]
        weights = np.exp(np.linspace(-2, 0, len(recent_trades))) # Exponential weights

        pressure = 0.0
        for i, trade in enumerate(recent_trades):
            multiplier = 1.0 if trade['side'] == 'buy' else -1.0
            pressure += multiplier * trade['amount'] * weights[i]

        return np.tanh(pressure / (np.sum(weights) + 1e-9)) # Normalize with tanh

    def _calculate_liquidity_score(self, snapshot: OrderBookSnapshot) -> float:
        """Calculates a composite score representing market liquidity.

        The score combines three factors:
        1.  **Depth**: The total volume available at the best bid and ask prices.
        2.  **Spread**: The tightness of the bid-ask spread.
        3.  **Distribution**: The evenness of volume distribution across the top
            price levels of the order book.

        A higher score indicates better, more stable liquidity.
        """
        bid_depth = snapshot.bids[0][1] if snapshot.bids else 0
        ask_depth = snapshot.asks[0][1] if snapshot.asks else 0
        depth_score = min((bid_depth + ask_depth) / 100.0, 1.0) # Normalize against a reference value

        spread_bps = (snapshot.spread / snapshot.mid_price) * 10000
        spread_score = 1.0 / (1.0 + spread_bps)

        bid_sizes = [size for _, size in snapshot.bids[:self.depth_levels]]
        ask_sizes = [size for _, size in snapshot.asks[:self.depth_levels]]
        distribution_score = 0.5
        if len(bid_sizes) > 1 and len(ask_sizes) > 1:
            bid_cv = np.std(bid_sizes) / (np.mean(bid_sizes) + 1e-9)
            ask_cv = np.std(ask_sizes) / (np.mean(ask_sizes) + 1e-9)
            distribution_score = 1.0 / (1.0 + (bid_cv + ask_cv) / 2)

        return 0.4 * depth_score + 0.4 * spread_score + 0.2 * distribution_score

    def _calculate_toxicity_score(self) -> float:
        """Estimates trade flow toxicity (adverse selection risk).

        This score measures the price impact of recent trades. High toxicity
        suggests that informed traders are active, which can be a precursor to
        significant price movements as the market absorbs new information.
        """
        if len(self.snapshot_history) < 10 or len(self.trade_history) < 10:
            return 0.0

        impacts = []
        recent_trades = list(self.trade_history)[-20:]
        for i, trade in enumerate(recent_trades):
            if i < len(recent_trades) - 1:
                next_trade = recent_trades[i+1]
                price_change = next_trade['price'] - trade['price']
                impact = (price_change if trade['side'] == 'buy' else -price_change) / trade['price']
                impacts.append(abs(impact))

        if not impacts:
            return 0.0

        toxicity_bps = np.mean(impacts) * 10000  # Average impact in basis points
        return min(toxicity_bps / 5.0, 1.0) # Normalize against a reference value of 5 bps

    def _default_metrics(self) -> OrderFlowMetrics:
        """Returns a set of neutral default metrics when there is not enough data."""
        return OrderFlowMetrics(0.0, 0.0, 0.0, 0.5, 0.0)

    def record_trade(self, trade: Dict):
        """Records a new trade to the internal history for flow analysis."""
        self.trade_history.append(trade)

    def generate_alpha_signal(self, metrics: OrderFlowMetrics) -> Tuple[str, float]:
        """Generates a trading signal from the combined microstructure metrics.

        This method uses a multi-factor model to combine the various order flow
        indicators into a single trading score, which is then converted into a
        directional signal ('buy'/'sell'/'hold') and a confidence level.

        Returns:
            A tuple containing the signal direction and a confidence score.
        """
        score = 0.0

        # High liquidity implies order imbalance is more likely to be momentum.
        # Low liquidity implies it might be a temporary exhaustion to be faded.
        score += metrics.order_imbalance * (0.3 if metrics.liquidity_score > 0.7 else -0.2)

        # The pressure index is a strong indicator of short-term direction.
        score += metrics.pressure_index * 0.4

        # High VPIN or toxicity suggests the current trend is driven by informed
        # traders and may be prone to a reversal.
        if metrics.vpin > 0.6:
            score -= np.sign(metrics.pressure_index) * 0.2
        if metrics.toxicity_score > 0.5:
            score -= np.sign(metrics.pressure_index) * 0.15

        direction = 'buy' if score > 0 else 'sell' if score < 0 else 'hold'
        confidence = min(abs(score), 1.0)

        return ('hold', 0.0) if confidence < 0.3 else (direction, confidence)


def run_orderbook_strategy(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """A placeholder main entry point for an order book-based strategy.

    In a real implementation, this function would be integrated into the main
    signal generation engine and would consume real-time L2 data.
    """
    analyzer = OrderBookAnalyzer()
    # This is a placeholder; real-time data would be fed to the analyzer here.
    return 'hold', 0.0


if __name__ == '__main__':
    print("=== Order Book Analytics Engine ===")
    
    # Example usage with mock data
    analyzer = OrderBookAnalyzer(window_size=50, depth_levels=10)
    
    # Mock snapshot
    snapshot = OrderBookSnapshot(
        timestamp=pd.Timestamp.now(),
        bids=[(50000, 1.5), (49999, 2.0), (49998, 1.0)],
        asks=[(50001, 1.0), (50002, 2.5), (50003, 1.5)],
        mid_price=50000.5,
        spread=1.0
    )
    
    # Mock trade
    analyzer.record_trade({
        'symbol': 'BTC/USDT',
        'price': 50001,
        'amount': 0.5,
        'side': 'buy',
        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
    })
    
    metrics = analyzer.process_snapshot(snapshot)
    print(f"\nOrder Flow Metrics:")
    print(f"  Order Imbalance: {metrics.order_imbalance:.3f}")
    print(f"  VPIN: {metrics.vpin:.3f}")
    print(f"  Pressure Index: {metrics.pressure_index:.3f}")
    print(f"  Liquidity Score: {metrics.liquidity_score:.3f}")
    print(f"  Toxicity Score: {metrics.toxicity_score:.3f}")
    
    direction, confidence = analyzer.generate_alpha_signal(metrics)
    print(f"\nGenerated Signal: {direction.upper()} (Confidence: {confidence:.2f})")
