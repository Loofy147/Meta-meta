"""Provides a pre-execution risk validator for trade management.

This module acts as a crucial safety layer by validating every potential trade
against a set of configurable risk rules before it is sent for execution. It
serves as a pre-trade guardrail to prevent catastrophic losses that could result
from software bugs, unexpected market conditions, or flawed strategy logic.

The risk rules are dynamically loaded from the central system configuration,
allowing for adjustments without requiring a service restart.
"""

import os
import sys
from typing import Dict, Any, Optional
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import get_config

class RiskValidator:
    """Validates trades against a set of dynamic, database-driven risk rules.

    This class encapsulates the logic for fetching portfolio status and applying
    a series of risk checks to a proposed trade.

    Attributes:
        config: A dictionary containing the risk management rules loaded from
            the central configuration.
        db_conn: An active database connection passed during initialization.
    """
    def __init__(self, db_conn: Optional[connection] = None):
        """Initializes the RiskValidator.

        Args:
            db_conn: An active psycopg2 database connection. Passing an existing
                connection is recommended to allow the validator to participate
                in larger database transactions.
        """
        self.config = get_config().get('risk_management', {})
        self.db_conn = db_conn

    def get_portfolio_status(self, portfolio_name: str) -> Dict[str, Any]:
        """Retrieves a simplified, real-time status of a portfolio for risk checks.

        This method queries the database to calculate the total current value of
        all positions and the total value of the portfolio (cash + positions).

        Args:
            portfolio_name: The name of the portfolio to check.

        Returns:
            A dictionary containing the cash balance, total position value,
            and total portfolio value.
        """
        if not self.db_conn:
            raise ConnectionError("Database connection not provided to RiskValidator.")

        with self.db_conn.cursor() as cursor:
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
            portfolio_id, cash_balance = cursor.fetchone()

            cursor.execute("SELECT symbol, quantity FROM positions WHERE portfolio_id = %s;", (portfolio_id,))
            positions = cursor.fetchall()

            total_position_value = 0.0
            for symbol, quantity in positions:
                cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
                price_result = cursor.fetchone()
                if price_result:
                    total_position_value += price_result[0] * quantity

            return {
                "cash_balance": cash_balance,
                "total_position_value": total_position_value,
                "total_portfolio_value": cash_balance + total_position_value,
            }

    def is_trade_safe(self, portfolio_name: str, symbol: str, quantity: float, trade_value: float) -> bool:
        """Validates a proposed trade against all configured risk rules.

        This is the main method of the class. It executes a series of checks:

        1.  **Max Position Size**: Ensures the value of this single trade does
            not exceed a configured percentage of the total portfolio value.
        2.  **Max Portfolio Exposure**: Ensures that after this trade, the total
            value of all assets held does not exceed a configured percentage of
            the total portfolio value.

        Args:
            portfolio_name: The name of the portfolio for the trade.
            symbol: The symbol being traded.
            quantity: The quantity of the asset to be traded.
            trade_value: The total monetary value of the proposed trade.

        Returns:
            True if the trade passes all risk checks, False otherwise.
        """
        portfolio_status = self.get_portfolio_status(portfolio_name)
        total_portfolio_value = portfolio_status.get('total_portfolio_value', 0)

        if total_portfolio_value == 0:
            print("RISK CHECK FAILED: Cannot determine portfolio value.")
            return False

        # Rule 1: Max Position Size Check
        max_pos_pct = self.config.get('max_position_size_pct', 0.1) # Default 10%
        max_position_value = total_portfolio_value * max_pos_pct
        if trade_value > max_position_value:
            print(f"RISK CHECK FAILED: Trade value (${trade_value:,.2f}) exceeds max position size (${max_position_value:,.2f}).")
            return False

        # Rule 2: Max Total Portfolio Exposure Check
        max_exp_pct = self.config.get('max_portfolio_exposure_pct', 0.8) # Default 80%
        current_exposure = portfolio_status.get('total_position_value', 0)
        max_exposure_value = total_portfolio_value * max_exp_pct
        if current_exposure + trade_value > max_exposure_value:
            print(f"RISK CHECK FAILED: Trade would exceed max portfolio exposure (${max_exposure_value:,.2f}).")
            return False

        # Future checks could be added here, e.g., daily drawdown limits,
        # symbol concentration limits, etc.

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
