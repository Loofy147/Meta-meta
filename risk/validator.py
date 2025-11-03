"""
Pre-Execution Risk Validator

This module provides a crucial safety layer by validating every potential trade
against a set of configurable risk rules before it is sent for execution. It
acts as a pre-trade guardrail to prevent catastrophic losses due to bugs,
unexpected market conditions, or flawed strategy logic.
"""

import os
import sys
from typing import Dict, Any, List
from psycopg2.extensions import connection
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from scipy import stats

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import get_config

load_dotenv()

class PortfolioRiskAnalyzer:
    """
    Comprehensive risk analysis for multi-asset portfolios.
    """

    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1):
        """
        Args:
            confidence_level: Confidence level for VaR/CVaR (e.g., 0.95 = 95%)
            horizon_days: Risk horizon in days
        """
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        self.alpha = 1 - confidence_level

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        returns_df: pd.DataFrame,
        method: str = 'historical'
    ) -> Dict[str, float]:
        """
        Portfolio VaR considering correlations between assets.

        Args:
            positions: Dict of {symbol: dollar_value}
            returns_df: DataFrame with returns for each asset
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            Dict with VaR metrics
        """
        if returns_df.empty:
            return {
            'portfolio_var': 0.0,
            'portfolio_cvar': 0.0,
            'marginal_var': {},
            'diversification_benefit': 0.0
        }
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * self.horizon_days

        # Position weights
        total_value = sum(positions.values())
        weights = np.array([positions.get(col, 0) / total_value for col in returns_df.columns])

        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)

        # Portfolio VaR
        z_score = stats.norm.ppf(self.alpha)
        portfolio_return_var = z_score * portfolio_std
        portfolio_var_dollar = abs(portfolio_return_var * total_value)

        # Marginal VaR (contribution of each asset)
        marginal_var = {}
        for i, symbol in enumerate(returns_df.columns):
            if positions.get(symbol, 0) > 0:
                marginal_contrib = (cov_matrix.iloc[:, i] @ weights) / portfolio_std
                marginal_var[symbol] = marginal_contrib * positions[symbol] * abs(z_score)

        return {
            'portfolio_var': portfolio_var_dollar,
            'portfolio_cvar': portfolio_var_dollar * 1.3,  # Approximation
            'marginal_var': marginal_var,
            'diversification_benefit': sum(marginal_var.values()) - portfolio_var_dollar
        }


class StressTester:
    """
    Scenario analysis and stress testing framework.
    """

    def __init__(self):
        self.scenarios = {
            'flash_crash': {
                'price_shock': -0.20,  # 20% drop
                'volatility_spike': 3.0,  # 3x normal vol
                'correlation_increase': 0.3  # All correlations -> 1.0
            }
        }

    def run_stress_test(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        scenario_name: str = 'flash_crash'
    ) -> Dict[str, float]:
        """
        Simulates portfolio performance under stress scenario.

        Returns:
            Dict with stressed P&L and risk metrics
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]

        # Calculate stressed portfolio value
        total_pnl = 0.0
        position_pnls = {}

        for symbol, position_value in positions.items():
            current_price = current_prices.get(symbol, 0)
            if current_price == 0:
                continue

            # Apply price shock
            shocked_price = current_price * (1 + scenario['price_shock'])
            quantity = position_value / current_price

            # Calculate P&L
            pnl = (shocked_price - current_price) * quantity
            position_pnls[symbol] = pnl
            total_pnl += pnl

        total_value = sum(positions.values())
        pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0

        return {
            'scenario': scenario_name,
            'total_pnl': total_pnl,
            'pnl_percentage': pnl_pct,
            'position_pnls': position_pnls,
            'worst_position': min(position_pnls.items(), key=lambda x: x[1])[0] if position_pnls else None
        }


class KellyOptimizer:
    """
    Implements Kelly Criterion for optimal position sizing.
    """

    def optimize_portfolio_allocation(
        self,
        strategy_metrics: Dict[str, Dict],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Optimizes capital allocation across strategies using Kelly.
        """
        allocations = {}

        for strategy_name, metrics in strategy_metrics.items():
            kelly_frac = self.calculate_kelly_fraction(
                win_rate=metrics.get('hit_rate', 0.5),
                avg_win=metrics.get('avg_win', 100),
                avg_loss=metrics.get('avg_loss', 100)
            )
            allocations[strategy_name] = total_capital * kelly_frac

        return allocations

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Calculates optimal position size using Kelly Criterion.
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b

        kelly_fraction = max(0, min(kelly_fraction, max_kelly))

        return kelly_fraction * 0.5


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring and circuit breaker system.
    """

    def __init__(self):
        self.risk_analyzer = PortfolioRiskAnalyzer()
        self.stress_tester = StressTester()
        self.kelly_optimizer = KellyOptimizer()

        self.thresholds = {
            'max_daily_loss_pct': 5.0,
            'max_portfolio_var_pct': 10.0,
            'max_position_concentration': 0.30,
            'min_liquidity_score': 0.3
        }

    def check_risk_limits(
        self,
        portfolio_status: Dict,
        returns_history: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Comprehensive risk check across all metrics.
        """
        violations = []
        warnings = []

        current_pnl_pct = portfolio_status.get('daily_pnl_pct', 0)
        if current_pnl_pct < -self.thresholds['max_daily_loss_pct']:
            violations.append(f"Daily loss limit breached: {current_pnl_pct:.2f}%")

        positions = {p['symbol']: p['quantity'] * p['current_price']
                    for p in portfolio_status.get('positions', [])}

        if positions and not returns_history.empty:
            var_result = self.risk_analyzer.calculate_portfolio_var(
                positions, returns_history
            )
            portfolio_value = sum(positions.values())
            var_pct = (var_result['portfolio_var'] / portfolio_value * 100) if portfolio_value > 0 else 0

            if var_pct > self.thresholds['max_portfolio_var_pct']:
                violations.append(f"Portfolio VaR exceeds limit: {var_pct:.2f}%")

        if positions:
            total_value = sum(positions.values())
            max_position = max(positions.values()) / total_value if total_value > 0 else 0

            if max_position > self.thresholds['max_position_concentration']:
                warnings.append(f"High position concentration: {max_position*100:.1f}%")

        liquidity_score = portfolio_status.get('liquidity_score', 1.0)
        if liquidity_score < self.thresholds['min_liquidity_score']:
            warnings.append(f"Low liquidity score: {liquidity_score:.2f}")

        if violations:
            risk_status = 'breach'
        elif warnings:
            risk_status = 'warning'
        else:
            risk_status = 'safe'

        recommendations = self._generate_recommendations(violations, warnings)

        return {
            'risk_status': risk_status,
            'violations': violations,
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _generate_recommendations(
        self,
        violations: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generates actionable recommendations based on risk status"""
        recommendations = []

        if violations:
            recommendations.append("IMMEDIATE: Halt new position opening")
            recommendations.append("IMMEDIATE: Review and reduce exposure")

        if any('concentration' in w.lower() for w in warnings):
            recommendations.append("Consider rebalancing to reduce concentration")

        if any('liquidity' in w.lower() for w in warnings):
            recommendations.append("Reduce position sizes in illiquid assets")

        if any('var' in str(violations + warnings).lower()):
            recommendations.append("Implement tighter stop-losses")

        return recommendations


class RiskValidator:
    """
    Validates trades against a set of dynamic, database-driven risk rules.
    """
    def __init__(self, db_conn: connection):
        """
        Initializes the RiskValidator.

        Args:
            db_conn (psycopg2.extensions.connection): An active database connection
                to be used for fetching portfolio status.
        """
        self.config = get_config()['risk_management']
        self.db_conn = db_conn
        self.risk_analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
        self.stress_tester = StressTester()
        self.kelly_optimizer = KellyOptimizer()
        self.monitor = RealTimeRiskMonitor()


    def enhanced_risk_check(
        self,
        portfolio_status: Dict[str, Any],
        symbol: str,
        quantity: float,
        trade_value: float
    ) -> Dict[str, Any]:
        """
        Enhanced pre-trade risk validation with VaR and stress testing.

        Returns:
            Dict with risk assessment and approval status
        """
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
            stress_results['pnl_percentage'] > -10 and
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

if __name__ == '__main__':
    from portfolio.manager import get_db_connection, get_portfolio_status

    print("--- Risk Validator Example ---")
    conn = get_db_connection()
    try:
        validator = RiskValidator(conn)
        portfolio_status = get_portfolio_status('default')

        print("\nValidating a hypothetical $1,000 trade...")
        risk_assessment = validator.enhanced_risk_check(portfolio_status, 'BTC/USDT', 0.02, 1000.0)
        print(f"Is the trade safe? {risk_assessment['approved']}")
        print(f"Risk Assessment: {risk_assessment}")


        print("\nValidating a hypothetical $50,000 trade...")
        risk_assessment_large = validator.enhanced_risk_check(portfolio_status, 'BTC/USDT', 1.0, 50000.0)
        print(f"Is the large trade safe? {risk_assessment_large['approved']}")
        print(f"Risk Assessment: {risk_assessment_large}")

    except Exception as e:
        print(f"\nCould not run risk validator example (this is expected if the database is not seeded): {e}")
    finally:
        conn.close()
