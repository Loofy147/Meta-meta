import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from signals.engine import generate_signals

class TestSignalEngine(unittest.TestCase):

    @patch('signals.engine.get_config')
    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    @patch('signals.strategies.sentiment.generate_signal')
    def test_generate_signals_buy(self, mock_sentiment, mock_macd_signal, mock_rsi_signal, mock_get_config):
        # Mock config to ensure strategies are enabled for the test
        mock_get_config.return_value = {
            'strategies': {
                'rsi': {'enabled': True},
                'macd': {'enabled': True},
                'sentiment': {'enabled': True}
            }
        }
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('buy', 0.7)
        mock_macd_signal.return_value = ('buy', 0.6)
        mock_sentiment.return_value = ('hold', 0.0)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 2)
        self.assertIn({'strategy': 'rsi', 'direction': 'buy', 'confidence': 0.7, 'symbol': 'BTC/USDT'}, signals)
        self.assertIn({'strategy': 'macd', 'direction': 'buy', 'confidence': 0.6, 'symbol': 'BTC/USDT'}, signals)

    @patch('signals.engine.get_config')
    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    @patch('signals.strategies.sentiment.generate_signal')
    def test_generate_signals_sell(self, mock_sentiment, mock_macd_signal, mock_rsi_signal, mock_get_config):
        # Mock config
        mock_get_config.return_value = {
            'strategies': {
                'rsi': {'enabled': True},
                'macd': {'enabled': True},
                'sentiment': {'enabled': True}
            }
        }
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('sell', 0.7)
        mock_macd_signal.return_value = ('sell', 0.6)
        mock_sentiment.return_value = ('hold', 0.0)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 2)
        self.assertIn({'strategy': 'rsi', 'direction': 'sell', 'confidence': 0.7, 'symbol': 'BTC/USDT'}, signals)
        self.assertIn({'strategy': 'macd', 'direction': 'sell', 'confidence': 0.6, 'symbol': 'BTC/USDT'}, signals)

    @patch('signals.engine.get_config')
    @patch('signals.strategies.rsi.generate_signal')
    @patch('signals.strategies.macd.generate_signal')
    @patch('signals.strategies.sentiment.generate_signal')
    def test_generate_signals_hold(self, mock_sentiment, mock_macd_signal, mock_rsi_signal, mock_get_config):
        # Mock config
        mock_get_config.return_value = {
            'strategies': {
                'rsi': {'enabled': True},
                'macd': {'enabled': True},
                'sentiment': {'enabled': True}
            }
        }
        # Mock strategy outputs
        mock_rsi_signal.return_value = ('hold', 0.0)
        mock_macd_signal.return_value = ('hold', 0.0)
        mock_sentiment.return_value = ('hold', 0.0)

        signals = generate_signals('BTC/USDT')

        self.assertEqual(len(signals), 0)

if __name__ == '__main__':
    unittest.main()
