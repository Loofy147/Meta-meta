"""
Advanced Feature Service Wrappers

These services run as independent microservices and integrate advanced
features into the main trading system.
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
import signal

# Add parent directory
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
    """
    Continuous risk monitoring service.
    Publishes risk reports and alerts.
    """
    
    def __init__(self):
        self.integration = AdvancedRiskIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.check_interval = int(os.getenv('RISK_CHECK_INTERVAL', 300))
    
    async def run(self):
        """Main service loop"""
        self.running = True
        print(f"Risk Monitor Service started (interval: {self.check_interval}s)")
        
        while self.running:
            try:
                # Publish comprehensive risk report
                self.integration.publish_risk_report('default')
                
                # Heartbeat
                self.publisher.publish('service_heartbeat', {
                    'service': 'risk_monitor',
                    'status': 'healthy',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                print(f"Error in risk monitoring: {e}")
                self.publisher.publish('system_events', {
                    'service': 'risk_monitor',
                    'event_type': 'error',
                    'message': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                await asyncio.sleep(60)
    
    def stop(self):
        """Graceful shutdown"""
        print("Risk Monitor Service stopping...")
        self.running = False


# =====================================================
# RL OPTIMIZER SERVICE
# =====================================================

class RLOptimizerService:
    """
    Reinforcement learning strategy optimization service.
    Periodically adjusts strategy parameters based on learned policy.
    """
    
    def __init__(self):
        training_mode = os.getenv('TRAINING_MODE', 'false').lower() == 'true'
        self.integration = RLOptimizerIntegration(training_mode=training_mode)
        self.publisher = EventPublisher()
        self.running = False
        self.optimization_interval = int(os.getenv('OPTIMIZATION_INTERVAL', 3600))
    
    async def run(self):
        """Main service loop"""
        self.running = True
        print(f"RL Optimizer Service started (interval: {self.optimization_interval}s)")
        
        while self.running:
            try:
                # Optimize strategy parameters
                print("Running RL optimization...")
                new_config = self.integration.optimize_strategy_parameters()
                
                print(f"Applied configuration: {new_config}")
                
                # Wait for episode to complete
                await asyncio.sleep(self.optimization_interval)
                
                # Update from performance and train
                print("Updating RL agent from performance...")
                self.integration.update_from_performance()
                
                # Heartbeat
                self.publisher.publish('service_heartbeat', {
                    'service': 'rl_optimizer',
                    'status': 'healthy',
                    'episodes': self.integration.optimizer.steps,
                    'epsilon': self.integration.optimizer.epsilon,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                print(f"Error in RL optimization: {e}")
                self.publisher.publish('system_events', {
                    'service': 'rl_optimizer',
                    'event_type': 'error',
                    'message': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                await asyncio.sleep(300)
    
    def stop(self):
        """Graceful shutdown"""
        print("RL Optimizer Service stopping...")
        self.running = False
        # Save model before exit
        try:
            self.integration.optimizer.save_model()
            print("RL model saved successfully")
        except Exception as e:
            print(f"Error saving RL model: {e}")


# =====================================================
# ORDER BOOK STRATEGY SERVICE
# =====================================================

class OrderBookService:
    """
    Order book analytics service.
    Generates signals from market microstructure.
    """
    
    def __init__(self):
        self.integration = OrderBookStrategyIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.update_interval = 5  # 5 seconds
    
    async def run(self):
        """Main service loop"""
        self.running = True
        print("Order Book Strategy Service started")
        
        # Get symbols from config
        from config.manager import get_config
        config = get_config()
        symbols = config.get('ingestion', {}).get('symbols', ['BTC/USDT'])
        
        while self.running:
            try:
                for symbol in symbols:
                    # Generate signal from order book
                    direction, confidence = self.integration.generate_signal(symbol)
                    
                    if direction != 'hold' and confidence > 0.5:
                        # Publish as raw signal
                        self.publisher.publish('raw_signals', {
                            'strategy': 'orderbook',
                            'symbol': symbol,
                            'direction': direction,
                            'confidence': confidence,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        print(f"Orderbook signal: {symbol} {direction} ({confidence:.2f})")
                
                # Heartbeat
                if int(time.time()) % 60 == 0:  # Every minute
                    self.publisher.publish('service_heartbeat', {
                        'service': 'orderbook_strategy',
                        'status': 'healthy',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in order book analysis: {e}")
                await asyncio.sleep(30)
    
    def stop(self):
        """Graceful shutdown"""
        print("Order Book Service stopping...")
        self.running = False


# =====================================================
# ARBITRAGE SCANNER SERVICE
# =====================================================

class ArbitrageService:
    """
    Continuous arbitrage opportunity scanner.
    Monitors multiple exchanges for price differentials.
    """
    
    def __init__(self):
        self.integration = ArbitrageIntegration()
        self.publisher = EventPublisher()
        self.running = False
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', 1000)) / 1000  # ms to seconds
    
    async def run(self):
        """Main service loop"""
        self.running = True
        print("Arbitrage Scanner Service started")
        
        # Run the scanner
        await self.integration.run_scanner()
    
    def stop(self):
        """Graceful shutdown"""
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
