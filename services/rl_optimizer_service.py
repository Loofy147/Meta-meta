"""
Reinforcement Learning Optimizer Service

This service runs in the background to periodically update the system's
strategy parameters using the reinforcement learning optimizer.
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timezone
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_strategy_adapter import RLStrategyOptimizer, MarketStateEncoder
from config.manager import get_config, update_config
from event_bus.publisher import EventPublisher
from portfolio.manager import get_portfolio_status


load_dotenv()

class RLOptimizerIntegration:
    """
    Integrates RL Strategy Optimizer for dynamic parameter tuning.

    Periodically updates system configuration based on learned policy.
    """

    def __init__(self, training_mode: bool = False):

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

            cursor.execute("""
                SELECT STDDEV(close) / AVG(close)
                FROM candles_1h
                WHERE symbol = 'BTC/USDT'
                    AND time >= NOW() - INTERVAL '24 hours';
            """)
            result = cursor.fetchone()
            state_data['volatility'] = result[0] if result and result[0] else 0.02

            cursor.execute("SELECT strategy_name, hit_rate FROM strategy_performance;")
            for row in cursor.fetchall():
                state_data[f'{row[0]}_hit_rate'] = row[1]

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

        current_config['risk_management'].update({
            'max_position_size_pct': config['max_position_size_pct'],
            'max_portfolio_exposure_pct': config['max_portfolio_exposure_pct']
        })

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
            """,)
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


async def run_rl_optimization_loop():
    """
    Periodically runs the RL-based strategy optimization.
    """
    rl_integration = RLOptimizerIntegration(training_mode=True)

    while True:
        try:
            new_config = rl_integration.optimize_strategy_parameters()
            print(f"RL Optimizer applied new configuration: {new_config}")

            await asyncio.sleep(3600)
            rl_integration.update_from_performance()

        except Exception as e:
            print(f"Error in RL optimization: {e}")
            await asyncio.sleep(300)

if __name__ == '__main__':
    print("Starting RL Optimizer Service...")
    asyncio.run(run_rl_optimization_loop())
