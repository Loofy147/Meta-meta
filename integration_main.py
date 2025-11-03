"""
Advanced Features Integration Layer

This module integrates all advanced components into the existing trading system:
- Order Book Analytics
- Reinforcement Learning Strategy Optimizer
- Enterprise Risk Management
- Arbitrage Detection

It provides clean interfaces and orchestrates the flow of data between
the new advanced modules and the existing signal engine, portfolio manager,
and orchestrator.
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

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Existing system imports
from config.manager import get_config, update_config
from event_bus.publisher import EventPublisher
from portfolio.manager import get_portfolio_status

# Advanced module imports (assuming they're in advanced/ directory)
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced'))

load_dotenv()


class AdvancedRiskIntegration:
    """
    Integrates Enterprise Risk Management into the existing risk validator.
    
    Enhances risk/validator.py with VaR, CVaR, and stress testing.
    """
    
    def __init__(self):
        from advanced_risk_system import (
            PortfolioRiskAnalyzer,
            StressTester,
            KellyOptimizer,
            RealTimeRiskMonitor
        )
        
        self.risk_analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
        self.stress_tester = StressTester()
        self.kelly_optimizer = KellyOptimizer()
        self.monitor = RealTimeRiskMonitor()
        
        self.db_conn = self._get_db_connection()
        self.publisher = EventPublisher()
    
    def _get_db_connection(self):
        """Establishes database connection"""
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
    
    def enhanced_risk_check(
        self,
        portfolio_name: str,
        symbol: str,
        quantity: float,
        trade_value: float
    ) -> Dict[str, Any]:
        """
        Enhanced pre-trade risk validation with VaR and stress testing.
        
        Returns:
            Dict with risk assessment and approval status
        """
        # Get current portfolio status
        portfolio_status = get_portfolio_status(portfolio_name)
        
        # Get historical returns for VaR calculation
        returns_df = self._fetch_returns_history(
            [p['symbol'] for p in portfolio_status.get('positions', [])]
        )
        
        # 1. Run comprehensive risk check
        risk_result = self.monitor.check_risk_limits(
            portfolio_status,
            returns_df
        )
        
        # 2. Calculate VaR impact of new trade
        positions = {p['symbol']: p['quantity'] * p['current_price'] 
                    for p in portfolio_status.get('positions', [])}
        
        # Add proposed position
        positions[symbol] = positions.get(symbol, 0) + trade_value
        
        var_result = self.risk_analyzer.calculate_portfolio_var(
            positions,
            returns_df,
            method='historical'
        )
        
        # 3. Stress test with new position
        current_prices = {p['symbol']: p['current_price'] 
                         for p in portfolio_status.get('positions', [])}
        current_prices[symbol] = trade_value / quantity if quantity > 0 else 0
        
        stress_results = self.stress_tester.run_stress_test(
            positions,
            current_prices,
            scenario_name='flash_crash'
        )
        
        # 4. Kelly position sizing recommendation
        strategy_metrics = self._fetch_strategy_metrics()
        kelly_allocations = self.kelly_optimizer.optimize_portfolio_allocation(
            strategy_metrics,
            portfolio_status.get('cash_balance', 100000)
        )
        
        # Determine approval
        approved = (
            risk_result['risk_status'] != 'breach' and
            stress_results['pnl_percentage'] > -10 and  # Max 10% loss in flash crash
            var_result['portfolio_var'] < portfolio_status['portfolio_value'] * 0.15
        )
        
        return {
            'approved': approved,
            'risk_status': risk_result['risk_status'],
            'portfolio_var': var_result['portfolio_var'],
            'portfolio_cvar': var_result['portfolio_cvar'],
            'stress_test_pnl': stress_results['pnl_percentage'],
            'kelly_recommendation': kelly_allocations,
            'warnings': risk_result['warnings'],
            'recommendations': risk_result['recommendations']
        }
    
    def _fetch_returns_history(self, symbols: List[str], days: int = 90) -> pd.DataFrame:
        """Fetches historical returns for symbols"""
        if not symbols:
            return pd.DataFrame()
        
        try:
            # Query last 90 days of 1h candles
            query = """
                SELECT time, symbol, close
                FROM candles_1h
                WHERE symbol = ANY(%s)
                    AND time >= NOW() - INTERVAL '%s days'
                ORDER BY time;
            """
            
            df = pd.read_sql(
                query,
                self.db_conn,
                params=(symbols, days),
                parse_dates=['time']
            )
            
            # Pivot and calculate returns
            pivot = df.pivot(index='time', columns='symbol', values='close')
            returns = pivot.pct_change().dropna()
            
            return returns
        except Exception as e:
            print(f"Error fetching returns: {e}")
            return pd.DataFrame()
    
    def _fetch_strategy_metrics(self) -> Dict[str, Dict]:
        """Fetches strategy performance metrics for Kelly calculation"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT strategy_name, hit_rate, total_pnl, trade_count
                FROM strategy_performance;
            """)
            
            metrics = {}
            for row in cursor.fetchall():
                strategy_name, hit_rate, total_pnl, trade_count = row
                
                if trade_count > 0:
                    avg_trade = total_pnl / trade_count
                    # Estimate avg win/loss from hit rate and avg
                    avg_win = abs(avg_trade) * 2 if avg_trade > 0 else 100
                    avg_loss = abs(avg_trade) * 2 if avg_trade < 0 else 100
                else:
                    avg_win, avg_loss = 100, 100
                
                metrics[strategy_name] = {
                    'hit_rate': hit_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                }
            
            return metrics
        except Exception as e:
            print(f"Error fetching strategy metrics: {e}")
            return {}
    
    def publish_risk_report(self, portfolio_name: str = 'default'):
        """Publishes comprehensive risk report to event bus"""
        portfolio_status = get_portfolio_status(portfolio_name)
        returns_df = self._fetch_returns_history(
            [p['symbol'] for p in portfolio_status.get('positions', [])]
        )
        
        risk_result = self.monitor.check_risk_limits(portfolio_status, returns_df)
        
        # Publish to event bus for monitoring dashboard
        self.publisher.publish('risk_reports', {
            'portfolio': portfolio_name,
            'risk_status': risk_result['risk_status'],
            'violations': risk_result['violations'],
            'warnings': risk_result['warnings'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


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


class RLOptimizerIntegration:
    """
    Integrates RL Strategy Optimizer for dynamic parameter tuning.
    
    Periodically updates system configuration based on learned policy.
    """
    
    def __init__(self, training_mode: bool = False):
        from rl_strategy_adapter import RLStrategyOptimizer, MarketStateEncoder
        
        self.optimizer = RLStrategyOptimizer()
        self.encoder = MarketStateEncoder()
        self.training_mode = training_mode
        
        # Try to load existing model
        self.optimizer.load_model()
        
        self.db_conn = self._get_db_connection()
        self.publisher = EventPublisher()
        
        # Track current episode
        self.current_state = None
        self.current_action = None
        self.episode_start_value = None
    
    def _get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )
    
    def optimize_strategy_parameters(self) -> Dict[str, Any]:
        """
        Uses RL agent to select optimal strategy configuration.
        
        Should be called periodically (e.g., every hour) to adapt to market changes.
        """
        # 1. Encode current market state
        market_data = self._collect_market_state()
        state = self.encoder.encode(market_data)
        
        # 2. Select action (strategy configuration)
        action = self.optimizer.select_action(
            state,
            exploit=not self.training_mode
        )
        
        # 3. Get configuration for selected action
        new_config = self.optimizer.get_strategy_config(action)
        
        # 4. Apply configuration to system
        self._apply_configuration(new_config)
        
        # 5. Store for reward calculation later
        self.current_state = state
        self.current_action = action
        self.episode_start_value = self._get_portfolio_value()
        
        # 6. Log decision
        self.publisher.publish('rl_decisions', {
            'action': action,
            'config': json.dumps(new_config),
            'state_summary': {
                'sharpe': market_data.get('sharpe_ratio', 0),
                'volatility': market_data.get('volatility', 0),
                'trend': market_data.get('trend_strength', 0)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return new_config
    
    def update_from_performance(self):
        """
        Calculates reward and trains the RL agent based on recent performance.
        
        Should be called after an episode (e.g., after 1 hour of trading).
        """
        if self.current_state is None or not self.training_mode:
            return
        
        # Calculate performance metrics
        current_value = self._get_portfolio_value()
        performance_metrics = self._calculate_performance_metrics(
            self.episode_start_value,
            current_value
        )
        
        # Calculate reward
        reward = self.optimizer.calculate_reward(performance_metrics)
        
        # Get next state
        next_market_data = self._collect_market_state()
        next_state = self.encoder.encode(next_market_data)
        
        # Store transition
        self.optimizer.store_transition(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            done=False
        )
        
        # Train the agent
        if len(self.optimizer.memory) >= self.optimizer.batch_size:
            loss = self.optimizer.train_step()
            self.optimizer.update_target_network()
            
            print(f"RL Agent trained. Loss: {loss:.4f}, Reward: {reward:.3f}")
            
            # Periodically save model
            if self.optimizer.steps % 1000 == 0:
                self.optimizer.save_model()
    
    def _collect_market_state(self) -> Dict[str, float]:
        """Collects current market metrics for state encoding"""
        try:
            cursor = self.db_conn.cursor()
            
            state_data = {}
            
            # Get momentum indicators
            for tf in ['1m', '5m', '15m', '1h', '4h']:
                cursor.execute(f"""
                    SELECT (close - LAG(close, 20) OVER (ORDER BY time)) / LAG(close, 20) OVER (ORDER BY time)
                    FROM candles_{tf}
                    WHERE symbol = 'BTC/USDT'
                    ORDER BY time DESC
                    LIMIT 1;
                """)
                result = cursor.fetchone()
                state_data[f'momentum_{tf}'] = result[0] if result and result[0] else 0.0
            
            # Get volatility
            cursor.execute("""
                SELECT STDDEV(close) / AVG(close)
                FROM candles_1h
                WHERE symbol = 'BTC/USDT'
                    AND time >= NOW() - INTERVAL '24 hours';
            """)
            result = cursor.fetchone()
            state_data['volatility'] = result[0] if result and result[0] else 0.02
            
            # Get strategy performance
            cursor.execute("SELECT strategy_name, hit_rate FROM strategy_performance;")
            for row in cursor.fetchall():
                state_data[f'{row[0]}_hit_rate'] = row[1]
            
            # Get portfolio metrics
            portfolio = get_portfolio_status('default')
            state_data['sharpe_ratio'] = self._calculate_sharpe_ratio()
            state_data['drawdown'] = self._calculate_current_drawdown()
            
            return state_data
        except Exception as e:
            print(f"Error collecting market state: {e}")
            return {}
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Applies RL-selected configuration to the system"""
        current_config = get_config()
        
        # Update risk management parameters
        current_config['risk_management'].update({
            'max_position_size_pct': config['max_position_size_pct'],
            'max_portfolio_exposure_pct': config['max_portfolio_exposure_pct']
        })
        
        # Update strategy weights (for aggregator)
        # This would be stored in a separate table for the aggregator to read
        try:
            cursor = self.db_conn.cursor()
            for strategy, weight in config['strategy_weights'].items():
                cursor.execute("""
                    INSERT INTO strategy_weights (strategy_name, weight, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (strategy_name) DO UPDATE
                    SET weight = EXCLUDED.weight, updated_at = NOW();
                """, (strategy, weight))
            self.db_conn.commit()
        except Exception as e:
            print(f"Error updating strategy weights: {e}")
            self.db_conn.rollback()
        
        # Update system config
        update_config(current_config)
        
        print(f"Applied RL configuration: Confidence threshold = {config['confidence_threshold']}")
    
    def _get_portfolio_value(self) -> float:
        """Gets current total portfolio value"""
        portfolio = get_portfolio_status('default')
        return portfolio.get('portfolio_value', 0)
    
    def _calculate_performance_metrics(
        self,
        start_value: float,
        end_value: float
    ) -> Dict[str, float]:
        """Calculates performance metrics for reward"""
        pnl = end_value - start_value
        
        return {
            'pnl': pnl,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'drawdown': self._calculate_current_drawdown(),
            'hit_rate': self._calculate_recent_hit_rate()
        }
    
    def _calculate_sharpe_ratio(self, window_days: int = 30) -> float:
        """Calculates Sharpe ratio over recent period"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT 
                    AVG(daily_return) / NULLIF(STDDEV(daily_return), 0) * SQRT(365) as sharpe
                FROM (
                    SELECT 
                        (SUM(quantity * current_price) - LAG(SUM(quantity * current_price)) OVER (ORDER BY date))
                        / LAG(SUM(quantity * current_price)) OVER (ORDER BY date) as daily_return
                    FROM portfolio_snapshots
                    WHERE portfolio_name = 'default'
                        AND date >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY date
                ) returns;
            """, (window_days,))
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.0
        except:
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculates current drawdown from peak"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                WITH portfolio_values AS (
                    SELECT date, SUM(quantity * current_price) as value
                    FROM portfolio_snapshots
                    WHERE portfolio_name = 'default'
                        AND date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY date
                )
                SELECT 
                    (MAX(value) - (SELECT value FROM portfolio_values ORDER BY date DESC LIMIT 1))
                    / MAX(value) as drawdown
                FROM portfolio_values;
            """)
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.0
        except:
            return 0.0
    
    def _calculate_recent_hit_rate(self, window_trades: int = 50) -> float:
        """Calculates hit rate over recent trades"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as hit_rate
                FROM (
                    SELECT pnl
                    FROM closed_trades
                    WHERE portfolio_name = 'default'
                    ORDER BY closed_at DESC
                    LIMIT %s
                ) recent;
            """, (window_trades,))
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.5
        except:
            return 0.5


class ArbitrageIntegration:
    """
    Integrates Arbitrage Detection into the system.
    
    Runs as a separate service that publishes arbitrage opportunities.
    """
    
    def __init__(self):
        from arbitrage_detector import (
            ArbitrageOrchestrator,
            ExchangeData
        )
        
        self.orchestrator = ArbitrageOrchestrator()
        self.ExchangeData = ExchangeData
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379, db=0
        )
    
    async def run_scanner(self):
        """
        Continuously scans for arbitrage opportunities.
        
        Runs as an async service monitoring multiple exchange feeds.
        """
        print("Starting arbitrage scanner...")
        
        while True:
            try:
                # Collect latest orderbook data from all exchanges
                exchange_data = await self._collect_exchange_data()
                
                # Scan for opportunities
                opportunities = self.orchestrator.scan_all_opportunities(exchange_data)
                
                # Publish high-confidence opportunities
                for opp in opportunities:
                    if opp.confidence > 0.7 and opp.profit_pct > 0.5:
                        self.orchestrator.publish_opportunity(opp)
                        print(f"Arbitrage opportunity: {opp.symbol} | "
                              f"{opp.profit_pct:.2f}% | "
                              f"Confidence: {opp.confidence:.2f}")
                
                # Wait before next scan (arbitrage requires high frequency)
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in arbitrage scanner: {e}")
                await asyncio.sleep(5)
    
    async def _collect_exchange_data(self) -> List:
        """Collects orderbook data from all monitored exchanges"""
        # In production, this would connect to multiple exchange WebSockets
        # For now, return empty list
        # This would be populated from real-time exchange feeds
        return []


# Main integration orchestrator
class AdvancedSystemIntegrator:
    """
    Main orchestrator that coordinates all advanced features.
    """
    
    def __init__(self):
        print("Initializing Advanced Trading System Integration...")
        
        self.risk_integration = AdvancedRiskIntegration()
        self.orderbook_integration = OrderBookStrategyIntegration()
        self.rl_integration = RLOptimizerIntegration(training_mode=False)
        self.arbitrage_integration = ArbitrageIntegration()
        
        print("All advanced modules loaded successfully.")
    
    def start_services(self):
        """Starts all background services"""
        print("\nStarting advanced services...")
        
        # Service 1: Risk monitoring (every 5 minutes)
        asyncio.create_task(self._risk_monitoring_loop())
        
        # Service 2: RL optimization (every hour)
        asyncio.create_task(self._rl_optimization_loop())
        
        # Service 3: Arbitrage scanning (continuous)
        asyncio.create_task(self.arbitrage_integration.run_scanner())
        
        print("All services started.")
    
    async def _risk_monitoring_loop(self):
        """Periodic risk monitoring and reporting"""
        while True:
            try:
                self.risk_integration.publish_risk_report('default')
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                print(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _rl_optimization_loop(self):
        """Periodic RL-based strategy optimization"""
        while True:
            try:
                # Optimize parameters
                new_config = self.rl_integration.optimize_strategy_parameters()
                print(f"RL Optimizer applied new configuration: {new_config}")
                
                # Wait one hour, then update from performance
                await asyncio.sleep(3600)
                self.rl_integration.update_from_performance()
                
            except Exception as e:
                print(f"Error in RL optimization: {e}")
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
