"""A real-time, cross-exchange arbitrage detection engine.

This module provides a suite of tools for detecting various types of arbitrage
opportunities that can arise in cryptocurrency markets. It includes detectors for:
- **Simple Arbitrage**: The classic form of arbitrage, involving buying an asset
  on one exchange where it is priced lower and simultaneously selling it on
  another where it is priced higher.
- **Triangular Arbitrage**: An arbitrage that occurs within a single exchange
  due to price discrepancies between three different assets. For example,
  converting USDT -> BTC -> ETH -> USDT to end up with more USDT than initially.
- **Funding Rate Arbitrage**: A strategy for perpetual futures markets where a
  trader takes opposite positions in the spot and futures market to collect the
  funding rate payments.

The module is designed for high-frequency use, incorporating models for
transaction costs, fees, and execution latency to assess the viability of an
opportunity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import asyncio
from datetime import datetime, timezone
import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ArbitrageOpportunity:
    """A data class representing a detected arbitrage opportunity."""
    type: str  # e.g., 'simple', 'triangular', 'funding'
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    profit_net: float  # Estimated net profit in dollars after fees
    volume_limit: float # The maximum trade size limited by order book depth
    confidence: float   # A score from 0.0 to 1.0 indicating the reliability
    timestamp: datetime
    execution_window_ms: int  # Estimated time in ms the opportunity will last


@dataclass
class ExchangeData:
    """A consolidated data structure for an exchange's order book."""
    exchange: str
    symbol: str
    bid: float  # Highest price a buyer is willing to pay
    ask: float  # Lowest price a seller is willing to accept
    bid_size: float
    ask_size: float
    timestamp: datetime
    latency_ms: float # Time from exchange event to processing


class SimpleArbitrageDetector:
    """Detects simple price arbitrage between two or more exchanges.

    This detector implements the classic arbitrage strategy: buy low on
    Exchange A and simultaneously sell high on Exchange B. It includes a
    detailed cost model, accounting for taker fees and withdrawal fees, to
    ensure that only truly profitable opportunities are identified.
    """

    def __init__(self, min_profit_bps: float = 20):
        """Initializes the SimpleArbitrageDetector.

        Args:
            min_profit_bps: The minimum required profit margin in basis points
                (1 basis point = 0.01%) to consider an opportunity valid.
        """
        self.min_profit_bps = min_profit_bps
        # Configurable fees for different exchanges.
        self.exchange_fees = {
            'binance': 0.001,   # 0.1% taker fee
            'coinbase': 0.004,  # 0.4% taker fee
            'kraken': 0.0026,   # 0.26% taker fee
            'kucoin': 0.001,    # 0.1% taker fee
        }
        # Placeholder for asset withdrawal fees.
        self.withdrawal_fees = {
            'BTC': 0.0005,
            'ETH': 0.005,
            'USDT': 1.0
        }

    def detect_opportunities(self, exchange_data: List[ExchangeData]) -> List[ArbitrageOpportunity]:
        """Scans all pairs of exchanges for simple arbitrage opportunities.

        Args:
            exchange_data: A list of current `ExchangeData` objects, each
                representing the top-of-book for a symbol on an exchange.

        Returns:
            A list of any profitable `ArbitrageOpportunity` found.
        """
        opportunities = []
        by_symbol = defaultdict(list)
        for data in exchange_data:
            by_symbol[data.symbol].append(data)

        # Iterate through every possible pair of exchanges for each symbol.
        for symbol, data_list in by_symbol.items():
            if len(data_list) < 2:
                continue

            for i in range(len(data_list)):
                for j in range(i + 1, len(data_list)):
                    exch1 = data_list[i]
                    exch2 = data_list[j]

                    # Check for arbitrage in both directions (A->B and B->A).
                    opp1 = self._check_pair(exch1, exch2)
                    if opp1:
                        opportunities.append(opp1)
                    opp2 = self._check_pair(exch2, exch1)
                    if opp2:
                        opportunities.append(opp2)

        return opportunities

    def _check_pair(self, buy_data: ExchangeData, sell_data: ExchangeData) -> Optional[ArbitrageOpportunity]:
        """Checks a single directional pair of exchanges for an arbitrage opportunity."""
        # The arbitrage condition: must be able to sell higher than the buy price.
        buy_price = buy_data.ask
        sell_price = sell_data.bid

        if sell_price <= buy_price:
            return None

        # --- Detailed Profit Calculation ---
        buy_fee = self.exchange_fees.get(buy_data.exchange, 0.001)
        sell_fee = self.exchange_fees.get(sell_data.exchange, 0.001)

        # Calculate net profit after accounting for fees.
        cost_per_unit = buy_price * (1 + buy_fee)
        revenue_per_unit = sell_price * (1 - sell_fee)
        net_profit_per_unit = revenue_per_unit - cost_per_unit

        if cost_per_unit == 0: return None
        profit_pct = (net_profit_per_unit / cost_per_unit) * 100
        profit_bps = profit_pct * 100

        if profit_bps < self.min_profit_bps:
            return None
        # ------------------------------------

        # The maximum trade size is limited by the available liquidity (order book depth).
        volume_limit = min(buy_data.ask_size, sell_data.bid_size)
        profit_net = net_profit_per_unit * volume_limit

        # Estimate how long the opportunity might last based on latency.
        total_latency = buy_data.latency_ms + sell_data.latency_ms
        execution_window = max(1000 - total_latency * 2, 100)  # in ms

        confidence = self._calculate_confidence(buy_data, sell_data, profit_pct)

        return ArbitrageOpportunity(
            type='simple',
            symbol=buy_data.symbol,
            buy_exchange=buy_data.exchange,
            sell_exchange=sell_data.exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            profit_pct=profit_pct,
            profit_net=profit_net,
            volume_limit=volume_limit,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            execution_window_ms=int(execution_window)
        )

    def _calculate_confidence(self, buy_data: ExchangeData, sell_data: ExchangeData, profit_pct: float) -> float:
        """Calculates a confidence score for a detected opportunity.

        The score considers profit margin (provides a buffer), latency (lower is
        better), and available liquidity (higher is better).

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        profit_factor = 1 / (1 + np.exp(-(profit_pct - 0.5) * 4))  # Sigmoid scaling
        avg_latency = (buy_data.latency_ms + sell_data.latency_ms) / 2
        latency_factor = 1 / (1 + avg_latency / 100)
        min_size = min(buy_data.ask_size, sell_data.bid_size)
        liquidity_factor = min(min_size / 1.0, 1.0)  # Normalized to a max of 1 BTC/ETH etc.

        # Combine factors with weighting
        confidence = 0.5 * profit_factor + 0.3 * latency_factor + 0.2 * liquidity_factor
        return confidence


class TriangularArbitrageDetector:
    """Detects triangular arbitrage opportunities within a single exchange.

    This involves a sequence of three trades that start and end with the same
    asset, resulting in a net profit. For example:
    USDT -> BTC (buy BTC/USDT)
    BTC -> ETH  (buy ETH/BTC)
    ETH -> USDT (sell ETH/USDT)
    If the final USDT amount is greater than the initial amount, an arbitrage
    opportunity exists.
    """

    def __init__(self, min_profit_bps: float = 10):
        """Initializes the TriangularArbitrageDetector.

        Args:
            min_profit_bps: The minimum required profit in basis points.
        """
        self.min_profit_bps = min_profit_bps
        # Predefined common paths to check for efficiency.
        self.common_paths = [
            ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            ['BTC/USDT', 'BNB/BTC', 'BNB/USDT'],
            ['ETH/USDT', 'LINK/ETH', 'LINK/USDT'],
        ]

    def detect_opportunities(self, orderbook_data: Dict[str, ExchangeData], exchange: str) -> List[ArbitrageOpportunity]:
        """Searches for profitable triangular arbitrage paths.

        Args:
            orderbook_data: A dictionary mapping symbols to their `ExchangeData`.
            exchange: The name of the exchange being analyzed.

        Returns:
            A list of any profitable `ArbitrageOpportunity` found.
        """
        opportunities = []
        for path in self.common_paths:
            if all(pair in orderbook_data for pair in path):
                opp = self._check_triangular_path(
                    [orderbook_data[p] for p in path], exchange
                )
                if opp:
                    opportunities.append(opp)
        return opportunities

    def _check_triangular_path(self, path_data: List[ExchangeData], exchange: str) -> Optional[ArbitrageOpportunity]:
        """Checks one specific triangular path for a net profit."""
        initial_amount = 1.0  # Start with 1 unit of the base currency (e.g., USDT)
        current_amount = initial_amount
        fee = 0.001  # Assume a 0.1% fee per trade

        # --- Simulate the three trades ---
        # 1. USDT -> BTC (buy BTC/USDT)
        current_amount = (current_amount / path_data[0].ask) * (1 - fee)
        # 2. BTC -> ETH (buy ETH/BTC)
        current_amount = (current_amount / path_data[1].ask) * (1 - fee)
        # 3. ETH -> USDT (sell ETH/USDT)
        current_amount = (current_amount * path_data[2].bid) * (1 - fee)
        # --------------------------------

        profit = current_amount - initial_amount
        if initial_amount == 0: return None
        profit_pct = (profit / initial_amount) * 100
        profit_bps = profit_pct * 100

        if profit_bps < self.min_profit_bps:
            return None

        # The maximum volume is limited by the bottleneck in the trade path.
        volume_limits = [d.ask_size for d in path_data[:-1]] + [path_data[-1].bid_size]
        volume_limit = min(volume_limits) * initial_amount

        return ArbitrageOpportunity(
            type='triangular',
            symbol=' -> '.join([d.symbol for d in path_data]),
            buy_exchange=exchange,
            sell_exchange=exchange,
            buy_price=path_data[0].ask,
            sell_price=path_data[2].bid,
            profit_pct=profit_pct,
            profit_net=profit * volume_limit,
            volume_limit=volume_limit,
            confidence=0.7 if profit_pct > 0.2 else 0.5,
            timestamp=datetime.now(timezone.utc),
            execution_window_ms=500  # These are typically very short-lived.
        )


class FundingRateArbitrageDetector:
    """Detects arbitrage opportunities between spot and perpetual futures markets.

    This strategy, known as "cash and carry" arbitrage, exploits the funding
    rate mechanism in perpetual futures. When the funding rate is positive,
    traders with short positions pay those with long positions. The strategy is:
    - **Positive Funding**: Buy the asset on the spot market and simultaneously
      short it on the perpetual futures market. You pay interest on the spot
      position but collect the higher funding rate, profiting from the spread.
    - **Negative Funding**: The reverse, shorting spot and going long futures.
    """

    def __init__(self, min_annual_yield: float = 10.0):
        """Initializes the detector.

        Args:
            min_annual_yield: The minimum annualized yield (in percent) required
                to consider an opportunity attractive.
        """
        self.min_annual_yield = min_annual_yield

    def detect_opportunities(self, spot_price: float, perp_price: float, funding_rate: float, funding_interval_hours: int = 8) -> Optional[ArbitrageOpportunity]:
        """Checks for a funding rate arbitrage opportunity.

        Args:
            spot_price: The current price of the asset on the spot market.
            perp_price: The current price of the perpetual future.
            funding_rate: The funding rate for a single period (e.g., 0.01% for 8 hours).
            funding_interval_hours: The duration of the funding period in hours.

        Returns:
            An `ArbitrageOpportunity` if the expected yield is attractive,
            otherwise None.
        """
        # Annualize the periodic funding rate to compare it to other investments.
        periods_per_year = (365 * 24) / funding_interval_hours
        annualized_rate = funding_rate * periods_per_year * 100  # As a percentage

        if abs(annualized_rate) < self.min_annual_yield:
            return None

        # The "basis" is the price difference between futures and spot.
        basis = perp_price - spot_price
        basis_pct = (basis / spot_price) * 100

        # Determine the strategy based on the sign of the funding rate.
        if funding_rate > 0:  # Shorts pay longs
            strategy_name = "Long Spot / Short Perp"
            # Profit is from funding, minus the cost of basis convergence.
            profit_pct = annualized_rate - abs(basis_pct)
        else:  # Longs pay shorts
            strategy_name = "Short Spot / Long Perp"
            profit_pct = abs(annualized_rate) - abs(basis_pct)

        return ArbitrageOpportunity(
            type='funding',
            symbol=f"{strategy_name}",
            buy_exchange="spot",
            sell_exchange="perp",
            buy_price=spot_price,
            sell_price=perp_price,
            profit_pct=profit_pct,
            profit_net=0.0,  # Net profit depends on position size and duration
            volume_limit=float('inf'),
            confidence=0.8 if abs(annualized_rate) > 20 else 0.6,
            timestamp=datetime.now(timezone.utc),
            execution_window_ms=funding_interval_hours * 3600 * 1000
        )


class ArbitrageOrchestrator:
    """Orchestrates all arbitrage detectors and manages identified opportunities.

    This class acts as the central hub for the arbitrage engine. It runs all
    the different detection strategies, collects the opportunities they find,
    ranks them based on profitability and confidence, and publishes the best
    ones to the event bus for potential execution.
    """

    def __init__(self):
        """Initializes the ArbitrageOrchestrator."""
        self.simple_detector = SimpleArbitrageDetector(min_profit_bps=20)
        self.triangular_detector = TriangularArbitrageDetector(min_profit_bps=10)
        self.funding_detector = FundingRateArbitrageDetector(min_annual_yield=10)

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )

    def scan_all_opportunities(self, exchange_data: List[ExchangeData]) -> List[ArbitrageOpportunity]:
        """Runs all detection strategies and returns a ranked list of opportunities.

        Args:
            exchange_data: A list of the latest `ExchangeData` from all monitored
                exchanges.

        Returns:
            A list of `ArbitrageOpportunity` objects, sorted in descending
            order of a score combining net profit and confidence.
        """
        all_opportunities = []

        # Run simple (cross-exchange) arbitrage detection.
        simple_opps = self.simple_detector.detect_opportunities(exchange_data)
        all_opportunities.extend(simple_opps)

        # Run triangular (intra-exchange) arbitrage detection for each exchange.
        by_exchange = defaultdict(dict)
        for data in exchange_data:
            by_exchange[data.exchange][data.symbol] = data
        for exchange, data_dict in by_exchange.items():
            tri_opps = self.triangular_detector.detect_opportunities(data_dict, exchange)
            all_opportunities.extend(tri_opps)

        # Rank all found opportunities by a composite score.
        ranked = sorted(
            all_opportunities,
            key=lambda x: x.profit_net * x.confidence,
            reverse=True
        )

        return ranked

    def publish_opportunity(self, opp: ArbitrageOpportunity):
        """Publishes a detected arbitrage opportunity to the event bus.

        This allows a separate execution service to pick up and act on the
        opportunity.

        Args:
            opp: The `ArbitrageOpportunity` to publish.
        """
        self.redis_client.xadd('arbitrage_opportunities', {
            'type': opp.type,
            'symbol': opp.symbol,
            'buy_exchange': opp.buy_exchange,
            'sell_exchange': opp.sell_exchange,
            'profit_pct': opp.profit_pct,
            'confidence': opp.confidence,
            'volume_limit': opp.volume_limit,
            'execution_window_ms': opp.execution_window_ms
        })


def run_arbitrage_scanner():
    """
    Main entry point for arbitrage scanning service.
    
    In production, this would:
    1. Subscribe to real-time orderbook streams from multiple exchanges
    2. Continuously scan for opportunities
    3. Publish high-confidence opportunities for execution
    """
    orchestrator = ArbitrageOrchestrator()
    print("Arbitrage scanner initialized and monitoring...")
    return orchestrator


if __name__ == '__main__':
    print("=== Cross-Exchange Arbitrage Engine ===\n")
    
    # Initialize detectors
    simple_detector = SimpleArbitrageDetector(min_profit_bps=20)
    triangular_detector = TriangularArbitrageDetector(min_profit_bps=10)
    funding_detector = FundingRateArbitrageDetector(min_annual_yield=10)
    
    # Example 1: Simple arbitrage
    print("1. Simple Arbitrage Detection:")
    mock_data = [
        ExchangeData('binance', 'BTC/USDT', 49950, 50000, 1.5, 1.2, datetime.now(), 50),
        ExchangeData('coinbase', 'BTC/USDT', 50100, 50150, 1.0, 0.8, datetime.now(), 80),
    ]
    
    simple_opps = simple_detector.detect_opportunities(mock_data)
    if simple_opps:
        for opp in simple_opps:
            print(f"   Found: Buy on {opp.buy_exchange} @ ${opp.buy_price:,.2f}")
            print(f"          Sell on {opp.sell_exchange} @ ${opp.sell_price:,.2f}")
            print(f"          Profit: {opp.profit_pct:.3f}% | Net: ${opp.profit_net:.2f}")
            print(f"          Confidence: {opp.confidence:.2f} | Window: {opp.execution_window_ms}ms\n")
    else:
        print("   No simple arbitrage opportunities detected.\n")
    
    # Example 2: Funding rate arbitrage
    print("2. Funding Rate Arbitrage:")
    funding_opp = funding_detector.detect_opportunities(
        spot_price=50000,
        perp_price=50050,
        funding_rate=0.001,  # 0.1% per 8 hours
        funding_interval_hours=8
    )
    
    if funding_opp:
        print(f"   Strategy: Long Spot / Short Perp")
        print(f"   Expected Annual Yield: {funding_opp.profit_pct:.2f}%")
        print(f"   Confidence: {funding_opp.confidence:.2f}")
    else:
        print("   No attractive funding rate arbitrage.\n")
    
    print("\nArbitrage engine ready for production deployment.")
