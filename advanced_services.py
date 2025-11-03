"""Provides standalone, runnable microservices for advanced trading features.

This module contains the main entry points for running the advanced components
of the trading system as independent, long-running services. Each service is
designed to be launched in its own container or process and communicates with
the rest of the system via the central event bus.

The services include:
- **RiskMonitorService**: Continuously monitors portfolio risk and publishes
  reports and alerts.
- **RLOptimizerService**: A Reinforcement Learning agent that periodically
  optimizes strategy parameters and learns from performance data.
- **OrderBookService**: Analyzes market microstructure data to generate its own
  class of trading signals.
- **ArbitrageService**: Scans multiple exchanges in real-time to detect and
  report arbitrage opportunities.
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
import signal

# Add parent directory to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.advanced_integration import (
    AdvancedRiskIntegration,
    OrderBookStrategyIntegration,
    RLOptimizerIntegration,
    ArbitrageIntegration
)
from event_bus.publisher import EventPublisher


# =====================================================
# RISK MONITORING SERVICE
# =====================================================

class RiskMonitorService:
    """A service for continuous, real-time portfolio risk monitoring.

    This service runs in a loop, periodically invoking the advanced risk
    analysis tools to generate a comprehensive risk report. It checks for limit
    breaches (e.g., daily loss, VaR), publishes alerts to the event bus, and
    sends out a regular heartbeat to indicate it is operational.
    """

    def __init__(self):
        """Initializes the RiskMonitorService."""
        self.integration = AdvancedRiskIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.check_interval = int(os.getenv('RISK_CHECK_INTERVAL', 300))

    async def run(self):
        """The main asynchronous event loop for the service."""
        self.running = True
        print(f"Risk Monitor Service started. Performing checks every {self.check_interval} seconds.")

        while self.running:
            try:
                # Generate and publish a comprehensive risk report for the default portfolio.
                self.integration.publish_risk_report('default')

                # Publish a heartbeat to the event bus for system health monitoring.
                self.publisher.publish('service_heartbeat', {
                    'service': 'risk_monitor',
                    'status': 'healthy',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                print(f"An error occurred in the risk monitoring loop: {e}")
                self.publisher.publish('system_events', {
                    'service': 'risk_monitor',
                    'event_type': 'error',
                    'message': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                await asyncio.sleep(60)  # Wait longer after an error before retrying.

    def stop(self):
        """Initiates a graceful shutdown of the service."""
        print("Risk Monitor Service stopping...")
        self.running = False


# =====================================================
# RL OPTIMIZER SERVICE
# =====================================================

class RLOptimizerService:
    """A service that uses Reinforcement Learning to optimize strategy parameters.

    This service implements a learning loop where an RL agent periodically
    adjusts the parameters of the trading strategies (e.g., RSI thresholds,
    MACD periods) based on a learned policy. It then observes the performance
    of these new parameters and uses the results to further train its policy,
    aiming to maximize a reward function (e.g., Sharpe ratio).
    """

    def __init__(self):
        """Initializes the RLOptimizerService."""
        training_mode = os.getenv('TRAINING_MODE', 'false').lower() == 'true'
        self.integration = RLOptimizerIntegration(training_mode=training_mode)
        self.publisher = EventPublisher()
        self.running = False
        self.optimization_interval = int(os.getenv('OPTIMIZATION_INTERVAL', 3600))

    async def run(self):
        """The main asynchronous event loop for the service."""
        self.running = True
        print(f"RL Optimizer Service started. Optimizing every {self.optimization_interval} seconds.")

        while self.running:
            try:
                # Trigger the RL agent to select and apply a new set of strategy parameters.
                print("Running RL optimization step...")
                new_config = self.integration.optimize_strategy_parameters()
                print(f"Applied new strategy configuration via RL: {new_config}")

                # Wait for the duration of a "learning episode" to gather performance data.
                await asyncio.sleep(self.optimization_interval)

                # Use the performance data from the last period to update the agent's policy.
                print("Updating RL agent policy from recent performance...")
                self.integration.update_from_performance()

                # Publish a heartbeat with RL-specific metrics.
                self.publisher.publish('service_heartbeat', {
                    'service': 'rl_optimizer',
                    'status': 'healthy',
                    'episodes': self.integration.optimizer.steps,
                    'epsilon': self.integration.optimizer.epsilon,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            except Exception as e:
                print(f"An error occurred in RL optimization loop: {e}")
                self.publisher.publish('system_events', {
                    'service': 'rl_optimizer',
                    'event_type': 'error',
                    'message': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                await asyncio.sleep(300)

    def stop(self):
        """Initiates a graceful shutdown and saves the RL model state."""
        print("RL Optimizer Service stopping...")
        self.running = False
        try:
            # Ensure the latest state of the learned model is saved.
            self.integration.optimizer.save_model()
            print("RL model state saved successfully.")
        except Exception as e:
            print(f"Error saving RL model state on shutdown: {e}")


# =====================================================
# ORDER BOOK STRATEGY SERVICE
# =====================================================

class OrderBookService:
    """A service for analyzing order book data to generate trading signals.

    This service focuses on market microstructure. It continuously processes
    order book data (if available) to calculate metrics like order imbalance
    and VPIN. Based on these metrics, it generates its own 'buy', 'sell', or
    'hold' signals and publishes them to the `raw_signals` stream for the
    aggregator to consume.
    """

    def __init__(self):
        """Initializes the OrderBookService."""
        self.integration = OrderBookStrategyIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.update_interval = 5  # The interval in seconds for generating signals.

    async def run(self):
        """The main asynchronous event loop for the service."""
        self.running = True
        print("Order Book Strategy Service started.")

        from config.manager import get_config
        config = get_config()
        symbols = config.get('ingestion', {}).get('symbols', ['BTC/USDT'])

        while self.running:
            try:
                for symbol in symbols:
                    # Generate a signal based on the latest order book snapshot.
                    direction, confidence = self.integration.generate_signal(symbol)

                    if direction != 'hold' and confidence > 0.5:
                        # Publish the generated signal for the aggregator to use.
                        self.publisher.publish('raw_signals', {
                            'strategy': 'orderbook',
                            'symbol': symbol,
                            'direction': direction,
                            'confidence': confidence,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        print(f"Published order book signal: {symbol} {direction} ({confidence:.2f})")

                # Send a heartbeat once per minute.
                if int(time.time()) % 60 == 0:
                    self.publisher.publish('service_heartbeat', {
                        'service': 'orderbook_strategy',
                        'status': 'healthy',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                print(f"An error occurred in order book analysis loop: {e}")
                await asyncio.sleep(30)

    def stop(self):
        """Initiates a graceful shutdown of the service."""
        print("Order Book Service stopping...")
        self.running = False


# =====================================================
# ARBITRAGE SCANNER SERVICE
# =====================================================

class ArbitrageService:
    """A service for continuously scanning for arbitrage opportunities.

    This service connects to multiple exchanges simultaneously, monitors their
    order books, and identifies price discrepancies for the same asset across
    different venues. When a profitable arbitrage opportunity is detected, it
    is logged and can be acted upon.
    """

    def __init__(self):
        """Initializes the ArbitrageService."""
        self.integration = ArbitrageIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', 1000)) / 1000  # ms to seconds

    async def run(self):
        """The main asynchronous event loop for the service."""
        self.running = True
        print("Arbitrage Scanner Service started.")

        # The integration's run_scanner method contains its own infinite loop.
        await self.integration.run_scanner()

    def stop(self):
        """Initiates a graceful shutdown of the service."""
        print("Arbitrage Scanner Service stopping...")
        self.running = False


# =====================================================
# SERVICE LAUNCHERS
# =====================================================

def run_risk_monitor():
    """Launch risk monitoring service"""
    service = RiskMonitorService()
    
    def signal_handler(sig, frame):
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(service.run())


def run_rl_optimizer():
    """Launch RL optimizer service"""
    service = RLOptimizerService()
    
    def signal_handler(sig, frame):
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(service.run())


def run_orderbook_service():
    """Launch order book strategy service"""
    service = OrderBookService()
    
    def signal_handler(sig, frame):
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(service.run())


def run_arbitrage_service():
    """Launch arbitrage scanner service"""
    service = ArbitrageService()
    
    def signal_handler(sig, frame):
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(service.run())


# =====================================================
# MAIN ENTRY POINTS
# =====================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_services.py [risk|rl|orderbook|arbitrage]")
        sys.exit(1)
    
    service_type = sys.argv[1]
    
    services = {
        'risk': run_risk_monitor,
        'rl': run_rl_optimizer,
        'orderbook': run_orderbook_service,
        'arbitrage': run_arbitrage_service
    }
    
    if service_type not in services:
        print(f"Unknown service type: {service_type}")
        print(f"Available: {', '.join(services.keys())}")
        sys.exit(1)
    
    print(f"Starting {service_type} service...")
    services[service_type]()
