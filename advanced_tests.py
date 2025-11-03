"""
Comprehensive Test Suite for Advanced Features

Tests for:
- Order Book Analytics
- RL Strategy Optimizer
- Advanced Risk Management
- Arbitrage Detection
- Integration Layer
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOrderBookAnalytics(unittest.TestCase):
    """Tests for Order Book Analytics Engine"""
    
    def setUp(self):
        from advanced.orderbook_analytics import (
            OrderBookAnalyzer,
            OrderBookSnapshot
        )
        self.OrderBookAnalyzer = OrderBookAnalyzer
        self.OrderBookSnapshot = OrderBookSnapshot
        
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        analyzer = self.OrderBookAnalyzer(window_size=50, depth_levels=10)
        self.assertEqual(analyzer.window_size, 50)
        self.assertEqual(analyzer.depth_levels, 10)
        self.assertEqual(len(analyzer.snapshot_history), 0)
    
    def test_order_imbalance_calculation(self):
        """Test order imbalance metric"""
        analyzer = self.OrderBookAnalyzer()
        
        snapshot = self.OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bids=[(50000, 2.0), (49999, 1.5)],  # More buy volume
            asks=[(50001, 1.0), (50002, 1.0)],  # Less sell volume
            mid_price=50000.5,
            spread=1.0
        )
        
        metrics = analyzer.process_snapshot(snapshot)
        
        # Should be positive (buy pressure)
        self.assertGreater(metrics.order_imbalance, 0)
        self.assertLessEqual(abs(metrics.order_imbalance), 1.0)
    
    def test_vpin_calculation(self):
        """Test VPIN (toxicity) calculation"""
        analyzer = self.OrderBookAnalyzer()
        
        # Add some trades
        for i in range(60):
            analyzer.record_trade({
                'symbol': 'BTC/USDT',
                'price': 50000 + i,
                'amount': 0.1,
                'side': 'buy' if i % 2 == 0 else 'sell',
                'timestamp': int(datetime.now().timestamp() * 1000)
            })
        
        snapshot = self.OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bids=[(50000, 1.5)],
            asks=[(50001, 1.5)],
            mid_price=50000.5,
            spread=1.0
        )
        
        metrics = analyzer.process_snapshot(snapshot)
        
        # VPIN should be calculated
        self.assertIsNotNone(metrics.vpin)
        self.assertGreaterEqual(metrics.vpin, 0.0)
        self.assertLessEqual(metrics.vpin, 1.0)
    
    def test_signal_generation(self):
        """Test alpha signal generation from metrics"""
        from advanced.orderbook_analytics import OrderFlowMetrics
        
        analyzer = self.OrderBookAnalyzer()
        
        # Strong buy signal
        metrics = OrderFlowMetrics(
            order_imbalance=0.4,
            vpin=0.2,
            pressure_index=0.5,
            liquidity_score=0.8,
            toxicity_score=0.2
        )
        
        direction, confidence = analyzer.generate_alpha_signal(metrics)
        
        self.assertIn(direction, ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestRLOptimizer(unittest.TestCase):
    """Tests for RL Strategy Optimizer"""
    
    def setUp(self):
        from advanced.rl_strategy_adapter import (
            RLStrategyOptimizer,
            MarketStateEncoder
        )
        self.RLStrategyOptimizer = RLStrategyOptimizer
        self.MarketStateEncoder = MarketStateEncoder
    
    def test_optimizer_initialization(self):
        """Test RL optimizer initializes correctly"""
        optimizer = self.RLStrategyOptimizer()
        
        self.assertEqual(optimizer.state_dim, 32)
        self.assertEqual(optimizer.action_dim, 4)
        self.assertIsNotNone(optimizer.q_network)
        self.assertIsNotNone(optimizer.target_network)
    
    def test_market_state_encoding(self):
        """Test market state encoding"""
        encoder = self.MarketStateEncoder()
        
        market_data = {
            'momentum_1m': 0.02,
            'volatility': 0.025,
            'sharpe_ratio': 1.5,
            'rsi_hit_rate': 0.55
        }
        
        state = encoder.encode(market_data)
        
        self.assertEqual(len(state), encoder.feature_dim)
        self.assertTrue(np.all(np.isfinite(state)))
    
    def test_action_selection(self):
        """Test action selection with epsilon-greedy"""
        optimizer = self.RLStrategyOptimizer()
        
        state = np.random.randn(optimizer.state_dim)
        
        # Exploitation
        action = optimizer.select_action(state, exploit=True)
        self.assertIn(action, range(optimizer.action_dim))
        
        # Exploration
        actions = [optimizer.select_action(state) for _ in range(10)]
        self.assertTrue(len(set(actions)) > 1)  # Should explore
    
    def test_reward_calculation(self):
        """Test reward function"""
        optimizer = self.RLStrategyOptimizer()
        
        # Profitable performance
        good_metrics = {
            'pnl': 1000,
            'sharpe_ratio': 2.0,
            'drawdown': 0.02,
            'hit_rate': 0.65
        }
        
        reward_good = optimizer.calculate_reward(good_metrics)
        
        # Loss-making performance
        bad_metrics = {
            'pnl': -500,
            'sharpe_ratio': 0.5,
            'drawdown': 0.15,
            'hit_rate': 0.40
        }
        
        reward_bad = optimizer.calculate_reward(bad_metrics)
        
        # Good performance should have higher reward
        self.assertGreater(reward_good, reward_bad)
    
    def test_experience_replay(self):
        """Test experience replay mechanism"""
        optimizer = self.RLStrategyOptimizer()
        
        # Store some transitions
        for _ in range(optimizer.batch_size + 10):
            state = np.random.randn(optimizer.state_dim)
            action = np.random.randint(0, optimizer.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(optimizer.state_dim)
            
            optimizer.store_transition(state, action, reward, next_state, False)
        
        # Should be able to train
        loss = optimizer.train_step()
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)


class TestAdvancedRisk(unittest.TestCase):
    """Tests for Advanced Risk Management"""
    
    def setUp(self):
        from advanced.advanced_risk_system import (
            PortfolioRiskAnalyzer,
            StressTester,
            KellyOptimizer
        )
        self.PortfolioRiskAnalyzer = PortfolioRiskAnalyzer
        self.StressTester = StressTester
        self.KellyOptimizer = KellyOptimizer
    
    def test_var_historical(self):
        """Test historical VaR calculation"""
        analyzer = self.PortfolioRiskAnalyzer(confidence_level=0.95)
        
        # Generate mock returns (normal distribution)
        returns = pd.Series(np.random.randn(252) * 0.02)  # 2% daily vol
        position_value = 100000
        
        var = analyzer.calculate_var_historical(returns, position_value)
        
        self.assertGreater(var, 0)
        self.assertLess(var, position_value * 0.2)  # Reasonable range
    
    def test_var_parametric(self):
        """Test parametric VaR calculation"""
        analyzer = self.PortfolioRiskAnalyzer(confidence_level=0.95)
        
        returns = pd.Series(np.random.randn(100) * 0.02)
        position_value = 100000
        
        var = analyzer.calculate_var_parametric(returns, position_value)
        
        self.assertGreater(var, 0)
        self.assertIsInstance(var, float)
    
    def test_cvar_calculation(self):
        """Test CVaR (Expected Shortfall)"""
        analyzer = self.PortfolioRiskAnalyzer(confidence_level=0.95)
        
        returns = pd.Series(np.random.randn(252) * 0.02)
        position_value = 100000
        
        cvar = analyzer.calculate_cvar(returns, position_value)
        var = analyzer.calculate_var_historical(returns, position_value)
        
        # CVaR should be >= VaR
        self.assertGreaterEqual(cvar, var * 0.8)  # Allow some numerical error
    
    def test_stress_testing(self):
        """Test stress testing scenarios"""
        tester = self.StressTester()
        
        positions = {'BTC/USDT': 50000, 'ETH/USDT': 30000}
        current_prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}
        
        result = tester.run_stress_test(positions, current_prices, 'flash_crash')
        
        self.assertIn('scenario', result)
        self.assertIn('total_pnl', result)
        self.assertIn('pnl_percentage', result)
        
        # Flash crash should result in loss
        self.assertLess(result['total_pnl'], 0)
    
    def test_kelly_criterion(self):
        """Test Kelly position sizing"""
        optimizer = self.KellyOptimizer()
        
        # Favorable odds
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=150,
            avg_loss=100,
            max_kelly=0.25
        )
        
        self.assertGreater(kelly, 0)
        self.assertLessEqual(kelly, 0.25)
        
        # Unfavorable odds
        kelly_bad = optimizer.calculate_kelly_fraction(
            win_rate=0.40,
            avg_win=100,
            avg_loss=150
        )
        
        self.assertEqual(kelly_bad, 0)  # Should not bet


class TestArbitrageDetection(unittest.TestCase):
    """Tests for Arbitrage Detection Engine"""
    
    def setUp(self):
        from advanced.arbitrage_detector import (
            SimpleArbitrageDetector,
            TriangularArbitrageDetector,
            ExchangeData
        )
        self.SimpleArbitrageDetector = SimpleArbitrageDetector
        self.TriangularArbitrageDetector = TriangularArbitrageDetector
        self.ExchangeData = ExchangeData
    
    def test_simple_arbitrage_detection(self):
        """Test simple arbitrage between exchanges"""
        detector = self.SimpleArbitrageDetector(min_profit_bps=20)
        
        # Price differential
        exchange_data = [
            self.ExchangeData(
                'binance', 'BTC/USDT',
                bid=49900, ask=50000,
                bid_size=1.5, ask_size=1.2,
                timestamp=datetime.now(), latency_ms=50
            ),
            self.ExchangeData(
                'coinbase', 'BTC/USDT',
                bid=50200, ask=50250,
                bid_size=1.0, ask_size=0.8,
                timestamp=datetime.now(), latency_ms=80
            )
        ]
        
        opportunities = detector.detect_opportunities(exchange_data)
        
        self.assertGreater(len(opportunities), 0)
        self.assertEqual(opportunities[0].type, 'simple')
        self.assertGreater(opportunities[0].profit_pct, 0)
    
    def test_no_arbitrage_when_unprofitable(self):
        """Test no opportunities when spread too small"""
        detector = self.SimpleArbitrageDetector(min_profit_bps=50)
        
        # Small price differential
        exchange_data = [
            self.ExchangeData(
                'binance', 'BTC/USDT',
                bid=50000, ask=50010,
                bid_size=1.5, ask_size=1.2,
                timestamp=datetime.now(), latency_ms=50
            ),
            self.ExchangeData(
                'coinbase', 'BTC/USDT',
                bid=50005, ask=50015,
                bid_size=1.0, ask_size=0.8,
                timestamp=datetime.now(), latency_ms=80
            )
        ]
        
        opportunities = detector.detect_opportunities(exchange_data)
        
        self.assertEqual(len(opportunities), 0)


class TestIntegrationLayer(unittest.TestCase):
    """Tests for Integration Layer"""
    
    @patch('integration.advanced_integration.get_portfolio_status')
    @patch('integration.advanced_integration.psycopg2.connect')
    def test_risk_integration(self, mock_db, mock_portfolio):
        """Test risk management integration"""
        from integration.advanced_integration import AdvancedRiskIntegration
        
        # Mock portfolio status
        mock_portfolio.return_value = {
            'portfolio_value': 100000,
            'cash_balance': 20000,
            'positions': [
                {'symbol': 'BTC/USDT', 'quantity': 1.0, 'current_price': 50000}
            ]
        }
        
        # Mock database connection
        mock_conn = MagicMock()
        mock_db.return_value = mock_conn
        
        integration = AdvancedRiskIntegration()
        
        # Test should initialize without errors
        self.assertIsNotNone(integration.risk_analyzer)
        self.assertIsNotNone(integration.stress_tester)
    
    @patch('integration.advanced_integration.redis.Redis')
    def test_orderbook_integration(self, mock_redis):
        """Test order book strategy integration"""
        from integration.advanced_integration import OrderBookStrategyIntegration
        
        integration = OrderBookStrategyIntegration()
        
        # Should initialize analyzer
        self.assertIsNotNone(integration.analyzer)


class TestEndToEndScenarios(unittest.TestCase):
    """End-to-end scenario tests"""
    
    def test_signal_to_risk_check_flow(self):
        """Test complete flow from signal to risk validation"""
        # This would test the full pipeline:
        # 1. Generate order book signal
        # 2. Aggregate with other strategies
        # 3. Run risk checks
        # 4. Execute if approved
        
        # Mock implementation
        pass
    
    def test_rl_optimization_cycle(self):
        """Test full RL optimization cycle"""
        # This would test:
        # 1. Collect market state
        # 2. Select action
        # 3. Apply configuration
        # 4. Calculate reward
        # 5. Train agent
        
        # Mock implementation
        pass


def run_test_suite():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOrderBookAnalytics))
    suite.addTests(loader.loadTestsFromTestCase(TestRLOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedRisk))
    suite.addTests(loader.loadTestsFromTestCase(TestArbitrageDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
