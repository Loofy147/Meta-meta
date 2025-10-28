import json
import os

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio.manager import get_portfolio_status

class RiskValidator:
    """
    A class for validating trades against a set of risk rules.
    """
    def __init__(self):
        with open('config/main.json', 'r') as f:
            self.config = json.load(f)['risk_management']

    def validate_trade(self, portfolio_name, symbol, trade_value):
        """
        Validates a proposed trade against the configured risk rules.
        """
        portfolio_status = get_portfolio_status(portfolio_name)

        # 1. Check max position size
        max_position_size = portfolio_status['cash_balance'] * self.config['max_position_size_pct']
        if trade_value > max_position_size:
            print(f"Risk validation failed: Trade value ({trade_value}) exceeds max position size ({max_position_size}).")
            return False

        # 2. Check max portfolio exposure
        total_position_value = sum(p['quantity'] * p['current_price'] for p in portfolio_status['positions'])
        total_portfolio_value = portfolio_status['cash_balance'] + total_position_value
        max_exposure = total_portfolio_value * self.config['max_portfolio_exposure_pct']
        if total_position_value + trade_value > max_exposure:
            print(f"Risk validation failed: Trade would exceed max portfolio exposure ({max_exposure}).")
            return False

        return True

if __name__ == '__main__':
    # Example Usage
    validator = RiskValidator()

    # This is a placeholder for a real trade validation
    # This will likely fail if the portfolio has not been seeded with data
    try:
        is_valid = validator.validate_trade('default', 'BTC/USDT', 1000) # Validate a $1000 trade
        print(f"Is the trade valid? {is_valid}")
    except Exception as e:
        print(f"Could not validate trade (this is expected if the database is empty): {e}")
