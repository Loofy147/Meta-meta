"""Provides the integration layer for all advanced system components.

This module acts as a "glue layer," providing a clean and unified interface to
connect the advanced feature modules (like enterprise risk management, RL,
order book analysis, and arbitrage) with the core trading system infrastructure.

It defines a set of integration classes, each responsible for adapting one
advanced module to the existing system's data formats, function signatures,
and event flows. This approach keeps the advanced logic decoupled from the core,
making the system more modular and maintainable.
"""

import sys
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import psycopg2
import redis
from dotenv import load_dotenv

# Add parent directory to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core system imports
from config.manager import get_config, update_config
from event_bus.publisher import EventPublisher
from portfolio.manager import get_portfolio_status

# Dynamically import advanced modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced'))

load_dotenv()


class AdvancedRiskIntegration:
    """Integrates the Enterprise Risk Management system into the trade lifecycle.

    This class enhances the basic `risk/validator.py` by providing methods that
    incorporate advanced metrics like VaR, CVaR, and stress testing into the
    pre-trade validation process. It also includes functionality for periodic,
    system-wide risk reporting.
    """

    def __init__(self):
        """Initializes the AdvancedRiskIntegration layer."""
        from advanced_risk_system import PortfolioRiskAnalyzer, StressTester, KellyOptimizer, RealTimeRiskMonitor
        
        self.risk_analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
        self.stress_tester = StressTester()
        self.kelly_optimizer = KellyOptimizer()
        self.monitor = RealTimeRiskMonitor()
        self.db_conn = self._get_db_connection()
        self.publisher = EventPublisher()

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Establishes and returns a a database connection."""
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )

    def enhanced_risk_check(self, portfolio_name: str, symbol: str, quantity: float, trade_value: float) -> Dict[str, Any]:
        """Performs an enhanced pre-trade risk check using advanced metrics.

        This method is intended to be called by the orchestrator before executing a
        trade. It provides a much more robust validation than simple rule checks.

        Args:
            portfolio_name: The name of the portfolio for the trade.
            symbol: The symbol being traded.
            quantity: The quantity of the asset to be traded.
            trade_value: The total monetary value of the proposed trade.

        Returns:
            A dictionary containing the approval status ('approved': bool) and
            a detailed breakdown of the risk assessment.
        """
        portfolio_status = get_portfolio_status(portfolio_name)
        positions = {p['symbol']: p['quantity'] * p['current_price'] for p in portfolio_status.get('positions', [])}
        returns_df = self._fetch_returns_history(list(positions.keys()))
        
        # 1. Run the real-time monitor's comprehensive check.
        risk_result = self.monitor.check_risk_limits(portfolio_status, returns_df)
        
        # 2. Simulate the new position to check its impact on VaR.
        simulated_positions = positions.copy()
        simulated_positions[symbol] = simulated_positions.get(symbol, 0) + trade_value
        var_result = self.risk_analyzer.calculate_portfolio_var(simulated_positions, returns_df)
        
        # 3. Run a stress test on the simulated portfolio.
        current_prices = {p['symbol']: p['current_price'] for p in portfolio_status.get('positions', [])}
        current_prices[symbol] = trade_value / quantity if quantity > 0 else 0
        stress_results = self.stress_tester.run_stress_test(simulated_positions, current_prices, 'flash_crash')

        # Determine final approval based on a combination of risk metrics.
        approved = (
            risk_result['risk_status'] != 'breach' and
            stress_results['pnl_percentage'] > -15 and  # Max 15% loss in flash crash
            var_result.get('portfolio_var', 0) < portfolio_status.get('portfolio_value', 0) * 0.10
        )
        
        return {
            'approved': approved,
            'risk_status': risk_result.get('risk_status'),
            'portfolio_var': var_result.get('portfolio_var'),
            'stress_test_pnl_pct': stress_results.get('pnl_percentage'),
            'violations': risk_result.get('violations', []),
            'warnings': risk_result.get('warnings', [])
        }

    def _fetch_returns_history(self, symbols: List[str], days: int = 90) -> pd.DataFrame:
        """Fetches and computes historical daily returns for a list of symbols."""
        if not symbols:
            return pd.DataFrame()
        try:
            query = "SELECT time, symbol, close FROM candles_1h WHERE symbol = ANY(%s) AND time >= NOW() - INTERVAL '%s days' ORDER BY time;"
            df = pd.read_sql(query, self.db_conn, params=(symbols, days), parse_dates=['time'])
            pivot = df.pivot(index='time', columns='symbol', values='close')
            return pivot.pct_change().dropna()
        except Exception as e:
            print(f"Error fetching historical returns: {e}")
            return pd.DataFrame()

    def publish_risk_report(self, portfolio_name: str = 'default'):
        """Generates and publishes a comprehensive risk report to the event bus."""
        portfolio_status = get_portfolio_status(portfolio_name)
        positions = {p['symbol']: p['quantity'] * p['current_price'] for p in portfolio_status.get('positions', [])}
        returns_df = self._fetch_returns_history(list(positions.keys()))
        risk_result = self.monitor.check_risk_limits(portfolio_status, returns_df)
        
        self.publisher.publish('risk_reports', {
            'portfolio': portfolio_name,
            'risk_status': risk_result.get('risk_status'),
            'violations': risk_result.get('violations'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


class OrderBookStrategyIntegration:
    """Integrates the Order Book Analytics engine as a new signal source.

    This class provides a `generate_signal` method with a signature that matches
    the existing strategies in `signals/strategies/`. This allows it to be
    seamlessly plugged into the main `SignalEngine`.
    """

    def __init__(self):
        """Initializes the OrderBookStrategyIntegration."""
        from orderbook_analytics import OrderBookAnalyzer, OrderBookSnapshot
        self.analyzer = OrderBookAnalyzer()
        self.OrderBookSnapshot = OrderBookSnapshot
        self.db_conn = self._get_db_connection()
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")), db=0)
    
    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Establishes and returns a a database connection."""
        return psycopg2.connect(host=os.getenv("DB_HOST"), database=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"))

    def generate_signal(self, symbol: str) -> Tuple[str, float]:
        """Generates a trading signal from order book analysis.

        Args:
            symbol: The trading symbol to generate a signal for.

        Returns:
            A tuple containing the signal direction ('buy', 'sell', 'hold')
            and a confidence score.
        """
        snapshot_data = self._fetch_latest_orderbook(symbol)
        if not snapshot_data:
            return 'hold', 0.0
        
        snapshot = self.OrderBookSnapshot(**snapshot_data)
        metrics = self.analyzer.process_snapshot(snapshot)
        self._log_orderbook_metrics(symbol, metrics)
        
        return self.analyzer.generate_alpha_signal(metrics)

    def _fetch_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetches the latest order book snapshot from a Redis stream."""
        try:
            messages = self.redis_client.xrevrange(f'orderbook_{symbol}', count=1)
            if not messages: return None
            
            data = messages[0][1]
            return {
                'timestamp': pd.Timestamp.now(),
                'bids': json.loads(data[b'bids'].decode()),
                'asks': json.loads(data[b'asks'].decode()),
                'mid_price': float(data[b'mid_price'].decode()),
                'spread': float(data[b'spread'].decode())
            }
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None

    def _log_orderbook_metrics(self, symbol: str, metrics: Any):
        """Logs the calculated order flow metrics to the database."""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO orderbook_metrics (time, symbol, order_imbalance, vpin, pressure_index, liquidity_score, toxicity_score)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s);
                """, (symbol, metrics.order_imbalance, metrics.vpin, metrics.pressure_index, metrics.liquidity_score, metrics.toxicity_score))
            self.db_conn.commit()
        except Exception as e:
            print(f"Error logging order book metrics: {e}")
            self.db_conn.rollback()


class RLOptimizerIntegration:
    """Integrates the RL Strategy Optimizer for dynamic system parameter tuning.

    This class orchestrates the interaction between the RL agent and the live
    trading system. It handles collecting the current market state, asking the
    agent for the optimal action (i.e., strategy configuration), applying that
    configuration, and later providing performance feedback to the agent for
    learning.
    """

    def __init__(self, training_mode: bool = False):
        """Initializes the RLOptimizerIntegration."""
        from rl_strategy_adapter import RLStrategyOptimizer, MarketStateEncoder
        
        self.optimizer = RLStrategyOptimizer()
        self.encoder = MarketStateEncoder()
        self.training_mode = training_mode
        self.optimizer.load_model()
        self.db_conn = self._get_db_connection()
        self.publisher = EventPublisher()
        self.current_state: Optional[np.ndarray] = None
        self.current_action: Optional[int] = None
        self.episode_start_value: Optional[float] = None

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Establishes and returns a a database connection."""
        return psycopg2.connect(host=os.getenv("DB_HOST"), database=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"))

    def optimize_strategy_parameters(self) -> Dict[str, Any]:
        """Uses the RL agent to select and apply an optimal strategy configuration.

        This method should be called periodically (e.g., every hour) to allow
        the system to adapt to changing market conditions.

        Returns:
            The new configuration dictionary that was applied.
        """
        market_data = self._collect_market_state()
        state = self.encoder.encode(market_data)
        action = self.optimizer.select_action(state, exploit=not self.training_mode)
        new_config = self.optimizer.get_strategy_config(action)
        self._apply_configuration(new_config)
        
        self.current_state = state
        self.current_action = action
        self.episode_start_value = self._get_portfolio_value()
        
        self.publisher.publish('rl_decisions', {'action': action, 'config_name': new_config.get('name')})
        return new_config

    def update_from_performance(self):
        """Calculates a reward based on recent performance and trains the agent.

        This method should be called after a set period (an "episode") to
        complete the feedback loop for the RL agent.
        """
        if self.current_state is None or not self.training_mode:
            return

        current_value = self._get_portfolio_value()
        if self.episode_start_value is None: return

        performance_metrics = self._calculate_performance_metrics(self.episode_start_value, current_value)
        reward = self.optimizer.calculate_reward(performance_metrics)
        next_state = self.encoder.encode(self._collect_market_state())
        
        self.optimizer.store_transition(self.current_state, self.current_action, reward, next_state, False)
        
        if len(self.optimizer.memory) >= self.optimizer.batch_size:
            loss = self.optimizer.train_step()
            self.optimizer.update_target_network()
            print(f"RL Agent trained. Loss: {loss:.4f}, Reward: {reward:.3f}")

    def _collect_market_state(self) -> Dict[str, float]:
        """Collects a wide range of current market metrics for state encoding."""
        # This is a simplified implementation. A production version would be more robust.
        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'drawdown': self._calculate_current_drawdown(),
        }

    def _apply_configuration(self, config: Dict[str, Any]):
        """Applies the RL-selected configuration to the live system."""
        current_config = get_config()
        current_config['risk_management'].update({
            'max_position_size_pct': config.get('max_position_size_pct'),
            'max_portfolio_exposure_pct': config.get('max_portfolio_exposure_pct',
                                                     current_config['risk_management']['max_portfolio_exposure_pct'])
        })
        update_config(current_config)
        
        try:
            with self.db_conn.cursor() as cursor:
                for strategy, weight in config.get('strategy_weights', {}).items():
                    cursor.execute("""
                        INSERT INTO strategy_weights (strategy_name, weight, updated_at) VALUES (%s, %s, NOW())
                        ON CONFLICT (strategy_name) DO UPDATE SET weight = EXCLUDED.weight, updated_at = NOW();
                    """, (strategy, weight))
            self.db_conn.commit()
        except Exception as e:
            print(f"Error updating strategy weights in DB: {e}")
            self.db_conn.rollback()

    def _get_portfolio_value(self) -> float:
        """Gets the current total portfolio value."""
        return get_portfolio_status('default').get('portfolio_value', 0)

    def _calculate_performance_metrics(self, start_value: float, end_value: float) -> Dict[str, float]:
        """Calculates key performance metrics for the reward function."""
        return {
            'pnl': end_value - start_value,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'drawdown': self._calculate_current_drawdown(),
            'hit_rate': 0.5 # Placeholder
        }

    def _calculate_sharpe_ratio(self) -> float:
        # Placeholder implementation
        return 1.0
    
    def _calculate_current_drawdown(self) -> float:
        # Placeholder implementation
        return 0.05


class ArbitrageIntegration:
    """Integrates the Arbitrage Detection engine into the system.

    This class provides the logic for running the arbitrage scanner as a
    continuous background service that publishes identified opportunities to
    the event bus for a separate execution module to handle.
    """

    def __init__(self):
        """Initializes the ArbitrageIntegration."""
        from arbitrage_detector import ArbitrageOrchestrator, ExchangeData
        
        self.orchestrator = ArbitrageOrchestrator()
        self.ExchangeData = ExchangeData
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")), db=0)

    async def run_scanner(self):
        """Continuously scans for arbitrage opportunities.

        This method runs an infinite loop, collecting the latest order book data
        from various exchanges, running it through the arbitrage detection
        logic, and publishing any high-confidence opportunities.
        """
        print("Starting continuous arbitrage scanner...")
        while True:
            try:
                exchange_data = await self._collect_exchange_data()
                opportunities = self.orchestrator.scan_all_opportunities(exchange_data)
                
                for opp in opportunities:
                    if opp.confidence > 0.75 and opp.profit_pct > 0.4:
                        self.orchestrator.publish_opportunity(opp)
                        print(f"Published arbitrage opportunity: {opp.symbol} | Profit: {opp.profit_pct:.3f}%")
                
                await asyncio.sleep(1) # Arbitrage scanning must be high-frequency.
            except Exception as e:
                print(f"Error in arbitrage scanner loop: {e}")
                await asyncio.sleep(10)

    async def _collect_exchange_data(self) -> List[Any]:
        """Collects real-time order book data from all monitored exchanges.

        NOTE: This is a placeholder. A production implementation would use
        asynchronous clients (e.g., via ccxt.pro) to subscribe to real-time
        WebSocket feeds for order book data from multiple exchanges.
        """
        return []


class AdvancedSystemIntegrator:
    """A main orchestrator class for coordinating all advanced features.
    
    This class is intended to be used in a main application entry point to
    launch and manage the various background services for the advanced features.
    """

    def __init__(self):
        """Initializes the AdvancedSystemIntegrator."""
        print("Initializing Advanced Trading System Integration Layer...")
        self.risk_integration = AdvancedRiskIntegration()
        self.orderbook_integration = OrderBookStrategyIntegration()
        self.rl_integration = RLOptimizerIntegration(training_mode=True) # Set training mode
        self.arbitrage_integration = ArbitrageIntegration()
        print("All advanced modules loaded and integrated successfully.")

    def start_services(self):
        """Starts all advanced features as concurrent background services."""
        print("\nStarting all advanced background services...")
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._risk_monitoring_loop())
            loop.create_task(self._rl_optimization_loop())
            loop.create_task(self.arbitrage_integration.run_scanner())
            print("All services have been scheduled to run.")
        except Exception as e:
            print(f"Error starting services: {e}")

    async def _risk_monitoring_loop(self):
        """The main loop for the periodic risk monitoring service."""
        while True:
            try:
                print("Risk monitor running...")
                self.risk_integration.publish_risk_report('default')
                await asyncio.sleep(300)  # Runs every 5 minutes
            except Exception as e:
                print(f"Error in risk monitoring service loop: {e}")
                await asyncio.sleep(60)

    async def _rl_optimization_loop(self):
        """The main loop for the periodic RL-based strategy optimization."""
        while True:
            try:
                print("RL optimizer running optimization step...")
                self.rl_integration.optimize_strategy_parameters()
                
                # Wait for the learning episode duration.
                await asyncio.sleep(3600)

                print("RL optimizer updating from performance...")
                self.rl_integration.update_from_performance()
                
            except Exception as e:
                print(f"Error in RL optimization service loop: {e}")
                await asyncio.sleep(300)


if __name__ == '__main__':
    print("=" * 60)
    print("ADVANCED TRADING SYSTEM INTEGRATION")
    print("=" * 60)
    
    # Initialize integrator
    integrator = AdvancedSystemIntegrator()
    
    # Start all services
    integrator.start_services()
    
    print("\nIntegration complete. System running with advanced features.")
    print("\nPress Ctrl+C to stop.")
    
    # Keep running
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
