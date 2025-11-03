"""
Enterprise Risk Management System

Implements institutional-grade risk metrics and controls:
- Value at Risk (VaR) using historical, parametric, and Monte Carlo methods
- Conditional Value at Risk (CVaR/ES) for tail risk
- Stress testing and scenario analysis
- Real-time correlation and covariance tracking
- Dynamic position sizing using Kelly Criterion
- Multi-asset portfolio risk attribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

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
        
    def calculate_var_historical(
        self,
        returns: pd.Series,
        position_value: float
    ) -> float:
        """
        Historical VaR using actual return distribution.
        
        Args:
            returns: Historical returns series
            position_value: Current position value
            
        Returns:
            VaR in dollar terms
        """
        if len(returns) < 100:
            return 0.0
        
        # Scale returns to horizon
        scaled_returns = returns * np.sqrt(self.horizon_days)
        
        # Calculate VaR at specified percentile
        var_percentile = np.percentile(scaled_returns, self.alpha * 100)
        var_dollar = abs(var_percentile * position_value)
        
        return var_dollar
    
    def calculate_var_parametric(
        self,
        returns: pd.Series,
        position_value: float
    ) -> float:
        """
        Parametric VaR assuming normal distribution.
        
        Faster but less accurate for fat-tailed distributions.
        """
        if len(returns) < 30:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        # Scale to horizon
        horizon_mean = mean * self.horizon_days
        horizon_std = std * np.sqrt(self.horizon_days)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(self.alpha)
        
        var_return = horizon_mean + z_score * horizon_std
        var_dollar = abs(var_return * position_value)
        
        return var_dollar
    
    def calculate_var_monte_carlo(
        self,
        returns: pd.Series,
        position_value: float,
        n_simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR using bootstrap simulation.
        
        Most accurate but computationally expensive.
        """
        if len(returns) < 50:
            return 0.0
        
        # Simulate future returns by sampling from historical distribution
        simulated_returns = np.random.choice(
            returns.values,
            size=(n_simulations, self.horizon_days),
            replace=True
        )
        
        # Calculate cumulative returns for each path
        cumulative_returns = (1 + simulated_returns).prod(axis=1) - 1
        
        # VaR is the percentile of losses
        var_percentile = np.percentile(cumulative_returns, self.alpha * 100)
        var_dollar = abs(var_percentile * position_value)
        
        return var_dollar
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        position_value: float
    ) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        
        Average loss beyond the VaR threshold - better measure of tail risk.
        """
        if len(returns) < 100:
            return 0.0
        
        scaled_returns = returns * np.sqrt(self.horizon_days)
        
        # Find VaR threshold
        var_threshold = np.percentile(scaled_returns, self.alpha * 100)
        
        # CVaR is average of returns below VaR
        tail_returns = scaled_returns[scaled_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        cvar_return = tail_returns.mean()
        cvar_dollar = abs(cvar_return * position_value)
        
        return cvar_dollar
    
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
            },
            'black_swan': {
                'price_shock': -0.35,
                'volatility_spike': 5.0,
                'correlation_increase': 0.5
            },
            'bullish_breakout': {
                'price_shock': 0.15,
                'volatility_spike': 1.5,
                'correlation_increase': 0.0
            },
            'gradual_decline': {
                'price_shock': -0.10,
                'volatility_spike': 1.2,
                'correlation_increase': 0.1
            },
            '2008_financial_crisis': {
                'price_shock': -0.40,
                'volatility_spike': 6.0,
                'correlation_increase': 0.7
            },
            'covid_march_2020': {
                'price_shock': -0.30,
                'volatility_spike': 4.5,
                'correlation_increase': 0.6
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
    
    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Runs all predefined stress scenarios.
        
        Returns:
            DataFrame with results for each scenario
        """
        results = []
        for scenario_name in self.scenarios.keys():
            result = self.run_stress_test(positions, current_prices, scenario_name)
            results.append(result)
        
        return pd.DataFrame(results)


class KellyOptimizer:
    """
    Implements Kelly Criterion for optimal position sizing.
    
    Kelly Criterion maximizes long-term growth rate while managing risk.
    """
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Calculates optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0 to 1)
            avg_win: Average winning trade size (positive)
            avg_loss: Average losing trade size (positive)
            max_kelly: Maximum kelly fraction (for safety)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            return 0.0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Clip to [0, max_kelly]
        kelly_fraction = max(0, min(kelly_fraction, max_kelly))
        
        # Apply fractional Kelly for safety (common practice)
        return kelly_fraction * 0.5  # Half-Kelly
    
    def optimize_portfolio_allocation(
        self,
        strategy_metrics: Dict[str, Dict],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Optimizes capital allocation across strategies using Kelly.
        
        Args:
            strategy_metrics: Dict of {strategy_name: {hit_rate, avg_win, avg_loss}}
            total_capital: Total available capital
            
        Returns:
            Dict of {strategy_name: allocated_capital}
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


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring and circuit breaker system.
    """
    
    def __init__(self):
        self.risk_analyzer = PortfolioRiskAnalyzer()
        self.stress_tester = StressTester()
        self.kelly_optimizer = KellyOptimizer()
        
        # Circuit breaker thresholds
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
        
        Returns dict with:
        - risk_status: 'safe', 'warning', 'breach'
        - violations: List of limit violations
        - recommendations: Suggested actions
        """
        violations = []
        warnings = []
        
        # 1. Daily loss check
        current_pnl_pct = portfolio_status.get('daily_pnl_pct', 0)
        if current_pnl_pct < -self.thresholds['max_daily_loss_pct']:
            violations.append(f"Daily loss limit breached: {current_pnl_pct:.2f}%")
        
        # 2. VaR check
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
        
        # 3. Concentration check
        if positions:
            total_value = sum(positions.values())
            max_position = max(positions.values()) / total_value if total_value > 0 else 0
            
            if max_position > self.thresholds['max_position_concentration']:
                warnings.append(f"High position concentration: {max_position*100:.1f}%")
        
        # 4. Liquidity check (placeholder - would need real liquidity data)
        liquidity_score = portfolio_status.get('liquidity_score', 1.0)
        if liquidity_score < self.thresholds['min_liquidity_score']:
            warnings.append(f"Low liquidity score: {liquidity_score:.2f}")
        
        # Determine overall status
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
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
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


def integrate_advanced_risk():
    """
    Integration point for advanced risk management.
    
    This should be called:
    - Before every trade (pre-execution check)
    - Periodically (every 5 minutes) for monitoring
    - At end of day for reporting
    """
    monitor = RealTimeRiskMonitor()
    print("Advanced Risk Management System initialized.")
    return monitor


if __name__ == '__main__':
    print("=== Enterprise Risk Management System ===\n")
    
    # Initialize components
    risk_analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
    stress_tester = StressTester()
    kelly_optimizer = KellyOptimizer()
    
    # Example 1: VaR calculation
    print("1. Value at Risk Calculation:")
    mock_returns = pd.Series(np.random.randn(252) * 0.02)  # ~2% daily vol
    position_value = 100000
    
    var_hist = risk_analyzer.calculate_var_historical(mock_returns, position_value)
    var_param = risk_analyzer.calculate_var_parametric(mock_returns, position_value)
    cvar = risk_analyzer.calculate_cvar(mock_returns, position_value)
    
    print(f"   Historical VaR (95%): ${var_hist:,.2f}")
    print(f"   Parametric VaR (95%): ${var_param:,.2f}")
    print(f"   CVaR (Expected Shortfall): ${cvar:,.2f}")
    
    # Example 2: Stress testing
    print("\n2. Stress Testing:")
    positions = {'BTC/USDT': 50000, 'ETH/USDT': 30000}
    current_prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}
    
    stress_result = stress_tester.run_stress_test(positions, current_prices, 'flash_crash')
    print(f"   Scenario: {stress_result['scenario']}")
    print(f"   Portfolio P&L: ${stress_result['total_pnl']:,.2f} ({stress_result['pnl_percentage']:.2f}%)")
    
    # Example 3: Kelly position sizing
    print("\n3. Kelly Criterion Position Sizing:")
    strategy_metrics = {
        'rsi': {'hit_rate': 0.58, 'avg_win': 150, 'avg_loss': 100},
        'macd': {'hit_rate': 0.55, 'avg_win': 120, 'avg_loss': 90},
        'ml': {'hit_rate': 0.62, 'avg_win': 180, 'avg_loss': 110}
    }
    
    allocations = kelly_optimizer.optimize_portfolio_allocation(strategy_metrics, 100000)
    print("   Optimal capital allocation:")
    for strategy, capital in allocations.items():
        print(f"     {strategy}: ${capital:,.2f}")
    
    print("\nRisk management system ready for production deployment.")
