import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signal

class TestAggregator(unittest.TestCase):

    @patch('aggregator.main.generate_signal')
    def test_aggregate_signal_buy(self, mock_generate_signal):
        mock_generate_signal.return_value = 'buy'
        signal = aggregate_signal()
        self.assertEqual(signal['direction'], 'buy')
        self.assertEqual(signal['confidence'], 0.5)

    @patch('aggregator.main.generate_signal')
    def test_aggregate_signal_sell(self, mock_generate_signal):
        mock_generate_signal.return_value = 'sell'
        signal = aggregate_signal()
        self.assertEqual(signal['direction'], 'sell')
        self.assertEqual(signal['confidence'], 0.5)

    @patch('aggregator.main.generate_signal')
    def test_aggregate_signal_hold(self, mock_generate_signal):
        mock_generate_signal.return_value = 'hold'
        signal = aggregate_signal()
        self.assertEqual(signal['direction'], 'hold')
        self.assertEqual(signal['confidence'], 0.0)

if __name__ == '__main__':
    unittest.main()
