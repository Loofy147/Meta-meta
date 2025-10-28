import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals_for_symbol

class TestAggregator(unittest.TestCase):

    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_buy(self, mock_generate_signals):
        # Mock the output of the signal engine
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'buy', 'confidence': 0.6}
        ]

        signal = aggregate_signals_for_symbol('BTC/USDT')

        self.assertEqual(signal['direction'], 'buy')
        self.assertAlmostEqual(signal['confidence'], 0.65)

    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_sell(self, mock_generate_signals):
        # Mock the output of the signal engine
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'sell', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6}
        ]

        signal = aggregate_signals_for_symbol('BTC/USDT')

        self.assertEqual(signal['direction'], 'sell')
        self.assertAlmostEqual(signal['confidence'], 0.65)

    @patch('aggregator.main.generate_signals')
    def test_aggregate_signal_conflict(self, mock_generate_signals):
        # Mock conflicting signals
        mock_generate_signals.return_value = [
            {'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7},
            {'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6}
        ]

        signal = aggregate_signals_for_symbol('BTC/USDT')

        self.assertEqual(signal['direction'], 'hold')
        self.assertEqual(signal['confidence'], 0.0)

if __name__ == '__main__':
    unittest.main()
