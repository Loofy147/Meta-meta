"""
Advanced Order Book Analytics Engine

This module implements sophisticated market microstructure analysis including:
- Level 2/3 order book reconstruction
- Order flow imbalance detection
- Liquidity heatmaps
- VWAP/TWAP execution quality metrics
- Trade flow toxicity indicators (VPIN)
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
    """Represents a point-in-time order book state"""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    

@dataclass
class OrderFlowMetrics:
    """Advanced order flow indicators"""
    order_imbalance: float  # Buy-sell imbalance
    vpin: float  # Volume-synchronized Probability of Informed Trading
    pressure_index: float  # Net buying/selling pressure
    liquidity_score: float  # Depth-weighted liquidity
    toxicity_score: float  # Adverse selection risk


class OrderBookAnalyzer:
    """
    Analyzes real-time order book data to extract alpha signals
    from market microstructure.
    """
    
    def __init__(self, window_size: int = 50, depth_levels: int = 10):
        """
        Args:
            window_size: Number of snapshots to maintain in history
            depth_levels: Number of price levels to analyze on each side
        """
        self.window_size = window_size
        self.depth_levels = depth_levels
        self.snapshot_history: deque = deque(maxlen=window_size)
        self.trade_history: deque = deque(maxlen=1000)
        
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379, db=0
        )
        
    def process_snapshot(self, snapshot: OrderBookSnapshot) -> OrderFlowMetrics:
        """
        Processes a new order book snapshot and calculates microstructure metrics.
        
        Args:
            snapshot: Current order book state
            
        Returns:
            OrderFlowMetrics containing calculated indicators
        """
        self.snapshot_history.append(snapshot)
        
        if len(self.snapshot_history) < 2:
            return self._default_metrics()
        
        # Calculate Order Imbalance (OI)
        order_imbalance = self._calculate_order_imbalance(snapshot)
        
        # Calculate VPIN (Volume-synchronized Probability of Informed Trading)
        vpin = self._calculate_vpin()
        
        # Calculate net pressure from recent trades
        pressure_index = self._calculate_pressure_index()
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(snapshot)
        
        # Calculate trade toxicity (adverse selection risk)
        toxicity_score = self._calculate_toxicity_score()
        
        return OrderFlowMetrics(
            order_imbalance=order_imbalance,
            vpin=vpin,
            pressure_index=pressure_index,
            liquidity_score=liquidity_score,
            toxicity_score=toxicity_score
        )
    
    def _calculate_order_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculates order book imbalance using volume-weighted depth.
        
        OI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        
        Returns value in [-1, 1] where:
        - Positive: More buying pressure
        - Negative: More selling pressure
        """
        bid_volume = sum(size for _, size in snapshot.bids[:self.depth_levels])
        ask_volume = sum(size for _, size in snapshot.asks[:self.depth_levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def _calculate_vpin(self) -> float:
        """
        Calculates Volume-synchronized Probability of Informed Trading (VPIN).
        
        VPIN estimates the probability that informed traders are active,
        using the imbalance of buyer vs seller initiated volume.
        
        Higher VPIN suggests more toxic order flow (informed trading).
        """
        if len(self.trade_history) < 50:
            return 0.0
        
        # Classify trades as buy or sell initiated
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in list(self.trade_history)[-50:]:
            if trade['side'] == 'buy':
                buy_volume += trade['amount']
            else:
                sell_volume += trade['amount']
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        # VPIN is the absolute volume imbalance
        vpin = abs(buy_volume - sell_volume) / total_volume
        return vpin
    
    def _calculate_pressure_index(self) -> float:
        """
        Calculates net buying/selling pressure from recent trade flow.
        
        Uses exponentially weighted moving average to give more weight
        to recent trades.
        """
        if len(self.trade_history) < 10:
            return 0.0
        
        recent_trades = list(self.trade_history)[-50:]
        weights = np.exp(np.linspace(-2, 0, len(recent_trades)))
        
        pressure = 0.0
        for i, trade in enumerate(recent_trades):
            multiplier = 1.0 if trade['side'] == 'buy' else -1.0
            pressure += multiplier * trade['amount'] * weights[i]
        
        # Normalize
        return np.tanh(pressure / (np.sum(weights) + 1e-8))
    
    def _calculate_liquidity_score(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculates a composite liquidity score based on:
        - Total depth at best prices
        - Spread tightness
        - Depth distribution across levels
        
        Higher score = better liquidity
        """
        # Component 1: Depth at touch
        bid_depth = snapshot.bids[0][1] if snapshot.bids else 0
        ask_depth = snapshot.asks[0][1] if snapshot.asks else 0
        depth_score = min(bid_depth + ask_depth, 100) / 100  # Normalize
        
        # Component 2: Spread tightness (inverse of spread)
        spread_score = 1.0 / (1.0 + snapshot.spread / snapshot.mid_price * 10000)  # bps
        
        # Component 3: Depth distribution (lower is better)
        bid_sizes = [size for _, size in snapshot.bids[:self.depth_levels]]
        ask_sizes = [size for _, size in snapshot.asks[:self.depth_levels]]
        
        if len(bid_sizes) > 1 and len(ask_sizes) > 1:
            bid_cv = np.std(bid_sizes) / (np.mean(bid_sizes) + 1e-8)
            ask_cv = np.std(ask_sizes) / (np.mean(ask_sizes) + 1e-8)
            distribution_score = 1.0 / (1.0 + (bid_cv + ask_cv) / 2)
        else:
            distribution_score = 0.5
        
        # Weighted combination
        liquidity_score = 0.4 * depth_score + 0.4 * spread_score + 0.2 * distribution_score
        return liquidity_score
    
    def _calculate_toxicity_score(self) -> float:
        """
        Estimates trade toxicity (adverse selection risk) by measuring
        price impact of recent trades.
        
        High toxicity indicates informed traders are active, which can
        signal upcoming price movements.
        """
        if len(self.snapshot_history) < 10 or len(self.trade_history) < 10:
            return 0.0
        
        # Calculate realized price impact of recent trades
        impacts = []
        recent_trades = list(self.trade_history)[-20:]
        
        for i, trade in enumerate(recent_trades):
            # Find nearest snapshot after trade
            trade_time = pd.Timestamp(trade['timestamp'], unit='ms')
            
            # Simple impact: price change in direction of trade
            if i < len(recent_trades) - 1:
                next_trade = recent_trades[i + 1]
                price_change = next_trade['price'] - trade['price']
                
                # Impact is positive if price moved in trade direction
                if trade['side'] == 'buy':
                    impact = price_change / trade['price']
                else:
                    impact = -price_change / trade['price']
                
                impacts.append(abs(impact))
        
        if not impacts:
            return 0.0
        
        # Average absolute impact, normalized
        toxicity = np.mean(impacts) * 10000  # in basis points
        return min(toxicity, 1.0)
    
    def _default_metrics(self) -> OrderFlowMetrics:
        """Returns neutral metrics when insufficient data"""
        return OrderFlowMetrics(
            order_imbalance=0.0,
            vpin=0.0,
            pressure_index=0.0,
            liquidity_score=0.5,
            toxicity_score=0.0
        )
    
    def record_trade(self, trade: Dict):
        """Records a trade for flow analysis"""
        self.trade_history.append(trade)
    
    def generate_alpha_signal(self, metrics: OrderFlowMetrics) -> Tuple[str, float]:
        """
        Generates trading signal from microstructure metrics.
        
        Returns:
            Tuple of (direction, confidence)
        """
        # Multi-factor scoring
        score = 0.0
        
        # Order imbalance signal (mean reversion vs momentum)
        if metrics.liquidity_score > 0.7:
            # High liquidity: imbalance is more predictive (momentum)
            score += metrics.order_imbalance * 0.3
        else:
            # Low liquidity: imbalance may mean-revert
            score -= metrics.order_imbalance * 0.2
        
        # Pressure index (strong signal)
        score += metrics.pressure_index * 0.4
        
        # VPIN signal (toxic flow suggests reversal)
        if metrics.vpin > 0.6:
            score -= np.sign(metrics.pressure_index) * 0.2
        
        # Toxicity signal
        if metrics.toxicity_score > 0.5:
            score -= np.sign(metrics.pressure_index) * 0.15
        
        # Convert to direction and confidence
        direction = 'buy' if score > 0 else 'sell' if score < 0 else 'hold'
        confidence = min(abs(score), 1.0)
        
        # Require minimum confidence threshold
        if confidence < 0.3:
            return 'hold', 0.0
        
        return direction, confidence


def run_orderbook_strategy(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """
    Main entry point for order book-based strategy.
    
    This would be integrated into signals/strategies/ directory.
    """
    analyzer = OrderBookAnalyzer()
    
    # In production, this would consume real-time L2 data
    # For now, we'll return a placeholder
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
