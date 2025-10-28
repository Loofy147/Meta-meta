import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals

class TestAggregator(unittest.TestCase):

    @patch('aggregator.main.get_strategy_weights')
    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_buy(self, mock_generate_signals, mock_get_weights):
        # Mock the output of the signal engine and weights
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'buy', 'confidence': 0.6}
        ]
        mock_get_weights.return_value = {'rsi': 0.6, 'macd': 0.4}

        signal = aggregate_signals('BTC/USDT', mock_get_weights.return_value)

        self.assertEqual(signal['direction'], 'buy')
        # Expected confidence: (0.7 * 0.6 + 0.6 * 0.4) / (0.6 + 0.4) = 0.66
        self.assertAlmostEqual(signal['confidence'], 0.66)

    @patch('aggregator.main.get_strategy_weights')
    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_sell(self, mock_generate_signals, mock_get_weights):
        # Mock the output of the signal engine and weights
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'sell', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6}
        ]
        mock_get_weights.return_value = {'rsi': 0.6, 'macd': 0.4}

        signal = aggregate_signals('BTC/USDT', mock_get_weights.return_value)

        self.assertEqual(signal['direction'], 'sell')
        self.assertAlmostEqual(signal['confidence'], 0.66)

    @patch('aggregator.main.get_strategy_weights')
    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_conflict(self, mock_generate_signals, mock_get_weights):
        # Mock conflicting signals
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6}
        ]
        mock_get_weights.return_value = {'rsi': 0.6, 'macd': 0.4}

        signal = aggregate_signals('BTC/USDT', mock_get_weights.return_value)

        self.assertEqual(signal['direction'], 'hold')
        self.assertEqual(signal['confidence'], 0.0)

if __name__ == '__main__':
    unittest.main()
