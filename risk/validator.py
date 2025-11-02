"""
Pre-Execution Risk Validator

This module provides a crucial safety layer by validating every potential trade
against a set of configurable risk rules before it is sent for execution. It
acts as a pre-trade guardrail to prevent catastrophic losses due to bugs,
unexpected market conditions, or flawed strategy logic.
"""

import os
import sys
from typing import Dict, Any
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import get_config

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

    def get_portfolio_status(self, portfolio_name: str) -> Dict[str, Any]:
        """
        Retrieves the current status of a portfolio.

        Note: This is a simplified version for risk validation purposes.
        """
        with self.db_conn.cursor() as cursor:
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
            portfolio_id, cash_balance = cursor.fetchone()

            cursor.execute("SELECT symbol, quantity, average_entry_price FROM positions WHERE portfolio_id = %s;", (portfolio_id,))
            positions = cursor.fetchall()

            total_position_value = 0.0
            for symbol, quantity, avg_price in positions:
                cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
                current_price = cursor.fetchone()[0]
                total_position_value += current_price * quantity

            return {
                "cash_balance": cash_balance,
                "total_position_value": total_position_value,
                "total_portfolio_value": cash_balance + total_position_value,
            }

    def is_trade_safe(self, portfolio_name: str, symbol: str, quantity: float, trade_value: float) -> bool:
        """
        Validates a proposed trade against all configured risk rules.

        Args:
            portfolio_name (str): The name of the portfolio for the trade.
            symbol (str): The symbol being traded.
            quantity (float): The quantity of the asset to be traded.
            trade_value (float): The total monetary value of the proposed trade.

        Returns:
            bool: True if the trade passes all risk checks, False otherwise.
        """
        portfolio_status = self.get_portfolio_status(portfolio_name)
        total_portfolio_value = portfolio_status['total_portfolio_value']

        # Rule 1: Max Position Size
        # Ensures a single trade does not represent an overly large portion of the portfolio.
        max_position_value = total_portfolio_value * self.config['max_position_size_pct']
        if trade_value > max_position_value:
            print(f"RISK CHECK FAILED: Trade value (${trade_value:.2f}) exceeds max position size (${max_position_value:.2f}).")
            return False

        # Rule 2: Max Portfolio Exposure
        # Ensures the total value of all positions does not exceed a percentage of the portfolio.
        current_exposure = portfolio_status['total_position_value']
        max_exposure_value = total_portfolio_value * self.config['max_portfolio_exposure_pct']
        if current_exposure + trade_value > max_exposure_value:
            print(f"RISK CHECK FAILED: Trade would exceed max portfolio exposure (${max_exposure_value:.2f}).")
            return False

        # Add other checks here, e.g., daily drawdown limits, symbol concentration limits, etc.

        return True

if __name__ == '__main__':
    # Example Usage:
    # Note: This requires a running database with portfolio data.
    from portfolio.manager import get_db_connection

    print("--- Risk Validator Example ---")
    conn = get_db_connection()
    try:
        validator = RiskValidator(conn)

        # Validate a hypothetical $1,000 trade in the 'default' portfolio
        print("\nValidating a hypothetical $1,000 trade...")
        is_safe = validator.is_trade_safe('default', 'BTC/USDT', 0.02, 1000.0)
        print(f"Is the trade safe? {is_safe}")

        # Validate a larger, potentially unsafe trade
        print("\nValidating a hypothetical $50,000 trade...")
        is_safe_large = validator.is_trade_safe('default', 'BTC/USDT', 1.0, 50000.0)
        print(f"Is the large trade safe? {is_safe_large}")

    except Exception as e:
        print(f"\nCould not run risk validator example (this is expected if the database is not seeded): {e}")
    finally:
        conn.close()
