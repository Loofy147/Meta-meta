import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.engine import generate_signals

class TestSignalEngine(unittest.TestCase):

    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    def test_generate_signals_buy(self, mock_macd_signal, mock_rsi_signal):
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('buy', 0.7)
        mock_macd_signal.return_value = ('buy', 0.6)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 2)
        self.assertIn({'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7, 'symbol': 'BTC/USDT'}, signals)
        self.assertIn({'strategy': 'macd', 'direction': 'buy', 'confidence': 0.6, 'symbol': 'BTC/USDT'}, signals)

    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    def test_generate_signals_sell(self, mock_macd_signal, mock_rsi_signal):
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('sell', 0.7)
        mock_macd_signal.return_value = ('sell', 0.6)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 2)
        self.assertIn({'strategy': 'rsi', 'direction': 'sell', 'confidence': 0.7, 'symbol': 'BTC/USDT'}, signals)
        self.assertIn({'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6, 'symbol': 'BTC/USDT'}, signals)

    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    def test_generate_signals_hold(self, mock_macd_signal, mock_rsi_signal):
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('hold', 0.0)
        mock_macd_signal.return_value = ('hold', 0.0)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 0)

if __name__ == '__main__':
    unittest.main()
