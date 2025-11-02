"""
Alpaca Broker Integration

This module provides a client for interacting with the Alpaca trading API. It
encapsulates the logic for submitting orders, checking their status, and listing
current positions, supporting both live and paper trading environments.
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from typing import Optional, List
from alpaca_trade_api.entity import Order, Position

load_dotenv()

class AlpacaBroker:
    """
    A broker class for submitting trades and managing positions via the Alpaca API.

    This class handles the authentication and provides simplified methods for common
    trading operations. It requires ALPACA_API_KEY and ALPACA_SECRET_KEY to be
    set in the environment.
    """
    def __init__(self):
        """
        Initializes the broker and establishes a connection to the Alpaca API.

        The API keys are read from environment variables. The connection defaults to
        the paper trading endpoint unless overridden by the ALPACA_BASE_URL env var.
        """
        try:
            self.api = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            )
            self.api.get_account() # Verify connection and credentials
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'gtc') -> Optional[Order]:
        """
        Submits a trade order to the broker.

        Args:
            symbol (str): The symbol for the asset to trade (e.g., 'BTC/USD').
            qty (float): The quantity of the asset to trade.
            side (str): The order side, either 'buy' or 'sell'.
            order_type (str): The type of order ('market', 'limit', etc.).
            time_in_force (str): The time-in-force for the order ('gtc', 'day', etc.).

        Returns:
            Optional[Order]: An Alpaca Order entity if the order was submitted
                             successfully, otherwise None.
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Submitted {side} order for {qty} {symbol}.")
            return order
        except Exception as e:
            print(f"An error occurred while submitting the order for {symbol}: {e}")
            return None

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Retrieves the status of a specific order by its ID.

        Args:
            order_id (str): The unique ID of the order to check.

        Returns:
            Optional[Order]: An Alpaca Order entity containing the latest status,
                             or None if the order is not found.
        """
        try:
            order = self.api.get_order(order_id)
            return order
        except Exception as e:
            print(f"An error occurred while retrieving status for order {order_id}: {e}")
            return None

    def list_positions(self) -> List[Position]:
        """
        Retrieves a list of all open positions in the account.

        Returns:
            List[Position]: A list of Alpaca Position entities. Returns an empty
                            list if there are no positions or an error occurs.
        """
        try:
            positions = self.api.list_positions()
            return positions
        except Exception as e:
            print(f"An error occurred while listing positions: {e}")
            return []

if __name__ == '__main__':
    # Example Usage:
    # This block demonstrates how to use the AlpacaBroker class.
    # It requires valid ALPACA_API_KEY and ALPACA_SECRET_KEY in the .env file.

    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        print("Skipping broker example: Alpaca API keys not found in .env file.")
    else:
        try:
            broker = AlpacaBroker()

            # --- List Positions ---
            print("Current Positions:")
            positions = broker.list_positions()
            if positions:
                for position in positions:
                    print(f"  - {position.symbol}: Qty={position.qty}, Avg Entry Price=${position.avg_entry_price}")
            else:
                print("  No open positions.")

            # --- Submit a Test Order (use with caution) ---
            # Uncomment the following lines to submit a real paper trading order.
            # print("\nSubmitting a test order...")
            # test_order = broker.submit_order(symbol='BTC/USD', qty=0.01, side='buy')
            # if test_order:
            #     print(f"Test order submitted with ID: {test_order.id}")
            #
            #     # --- Check Order Status ---
            #     import time
            #     time.sleep(2) # Give the order time to fill
            #     status = broker.get_order_status(test_order.id)
            #     if status:
            #         print(f"\nOrder Status for {test_order.id}: {status.status}")

        except ConnectionError as e:
            print(e)
