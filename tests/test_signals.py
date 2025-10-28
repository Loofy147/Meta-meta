import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.engine import generate_signal

class TestSignalEngine(unittest.TestCase):

    @patch('signals.engine.get_trade_data')
    def test_generate_signal_buy(self, mock_get_trade_data):
        # Create a sample dataframe where the short MA is greater than the long MA
        data = {'price': [i for i in range(100, 200)]}
        df = pd.DataFrame(data)
        mock_get_trade_data.return_value = df

        signal = generate_signal()
        self.assertEqual(signal, 'buy')

    @patch('signals.engine.get_trade_data')
    def test_generate_signal_sell(self, mock_get_trade_data):
        # Create a sample dataframe where the short MA is less than the long MA
        data = {'price': [i for i in range(200, 100, -1)]}
        df = pd.DataFrame(data)
        mock_get_trade_data.return_value = df

        signal = generate_signal()
        self.assertEqual(signal, 'sell')

    @patch('signals.engine.get_trade_data')
    def test_generate_signal_hold(self, mock_get_trade_data):
        # Create a sample dataframe with not enough data
        data = {'price': [100, 101, 102]}
        df = pd.DataFrame(data)
        mock_get_trade_data.return_value = df

        signal = generate_signal()
        self.assertEqual(signal, 'hold')

if __name__ == '__main__':
    unittest.main()
