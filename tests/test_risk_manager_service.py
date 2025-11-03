"""Unit tests for the Risk Manager Service."""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent directory to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.manager_service import flatten_portfolio

class TestRiskManagerService(unittest.TestCase):
    """Tests the core logic of the automated de-risking service."""

    @patch('risk.manager_service.get_portfolio_status')
    @patch('risk.manager_service.execute_trade')
    def test_flatten_portfolio_closes_open_long_position(
        self,
        mock_execute_trade: MagicMock,
        mock_get_portfolio_status: MagicMock
    ):
        """
        Verify that the flatten_portfolio function correctly closes a single
        open long position.
        """
        # Arrange: Mock the portfolio to have one open long position.
        mock_portfolio = {
            'positions': [
                {'symbol': 'BTC/USDT', 'quantity': 0.5}
            ]
        }
        mock_get_portfolio_status.return_value = mock_portfolio

        # Act: Call the function that should trigger the de-risking.
        flatten_portfolio('default')

        # Assert: Verify that execute_trade was called once with the correct parameters
        # to sell the long position.
        mock_execute_trade.assert_called_once_with(
            portfolio_name='default',
            symbol='BTC/USDT',
            side='sell',
            quantity=0.5
        )

    @patch('risk.manager_service.get_portfolio_status')
    @patch('risk.manager_service.execute_trade')
    def test_flatten_portfolio_closes_multiple_positions(
        self,
        mock_execute_trade: MagicMock,
        mock_get_portfolio_status: MagicMock
    ):
        """
        Verify that the flatten_portfolio function correctly closes multiple
        open positions, including both long and short positions.
        """
        # Arrange: Mock a portfolio with multiple positions.
        mock_portfolio = {
            'positions': [
                {'symbol': 'BTC/USDT', 'quantity': 0.5},
                {'symbol': 'ETH/USDT', 'quantity': -10.0} # A short position
            ]
        }
        mock_get_portfolio_status.return_value = mock_portfolio

        # Act: Call the de-risking function.
        flatten_portfolio('default')

        # Assert: Verify that execute_trade was called for each position with
        # the correct opposing action.
        self.assertEqual(mock_execute_trade.call_count, 2)

        # Check the call for the BTC long position
        mock_execute_trade.assert_any_call(
            portfolio_name='default',
            symbol='BTC/USDT',
            side='sell',
            quantity=0.5
        )
        # Check the call for the ETH short position
        mock_execute_trade.assert_any_call(
            portfolio_name='default',
            symbol='ETH/USDT',
            side='buy',
            quantity=10.0
        )

    @patch('risk.manager_service.get_portfolio_status')
    @patch('risk.manager_service.execute_trade')
    def test_flatten_portfolio_handles_no_open_positions(
        self,
        mock_execute_trade: MagicMock,
        mock_get_portfolio_status: MagicMock
    ):
        """
        Verify that the service does not attempt to execute trades if the
        portfolio has no open positions.
        """
        # Arrange: Mock an empty portfolio.
        mock_get_portfolio_status.return_value = {'positions': []}

        # Act: Call the de-risking function.
        flatten_portfolio('default')

        # Assert: Verify that execute_trade was not called.
        mock_execute_trade.assert_not_called()

if __name__ == '__main__':
    unittest.main()
