"""Provides a client for interacting with the Alpaca trading API.

This module encapsulates the logic for connecting to the Alpaca API, submitting
trade orders, and querying account information like order status and current
positions. It is designed to work with both live and paper trading environments,
configurable via environment variables.

This class serves as the direct interface to the external broker, abstracting away
the specifics of the `alpaca-trade-api` library from the rest of the application.
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from typing import Optional, List
from alpaca_trade_api.entity import Order, Position

load_dotenv()

class AlpacaBroker:
    """A broker client for placing trades and managing positions via Alpaca.

    This class handles API authentication and provides simplified, robust methods
    for common trading operations. It requires `ALPACA_API_KEY` and
    `ALPACA_SECRET_KEY` to be set in the environment.

    Attributes:
        api: An instance of the `alpaca_trade_api.REST` client.
    """
    def __init__(self):
        """Initializes the broker and connects to the Alpaca API.

        The API keys are read from environment variables. The connection defaults
        to the paper trading endpoint but can be overridden by setting the
        `ALPACA_BASE_URL` environment variable to the live trading URL.

        Raises:
            ConnectionError: If the API keys are missing or invalid, or if the
                connection to the Alpaca API fails.
        """
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ConnectionError("Alpaca API key/secret not found in environment variables.")

        try:
            self.api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            )
            self.api.get_account()  # Verify connection and credentials
            print("Successfully connected to Alpaca API.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'gtc') -> Optional[Order]:
        """Submits a trade order to the broker.

        Args:
            symbol: The symbol for the asset to trade (e.g., 'BTC/USD').
            qty: The quantity of the asset to trade. For crypto, this is the
                base currency amount.
            side: The order side, either 'buy' or 'sell'.
            order_type: The type of order, such as 'market' or 'limit'.
            time_in_force: The time-in-force for the order, e.g., 'gtc' (good
                'til canceled) or 'day'.

        Returns:
            An Alpaca `Order` entity if the order was submitted successfully,
            otherwise None.
        """
        try:
            # Note: Alpaca API expects crypto symbols with a '/', not concatenated.
            order = self.api.submit_order(
                symbol=symbol,
                qty=str(qty), # API expects quantity as a string
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Successfully submitted {side} order for {qty} {symbol}.")
            return order
        except Exception as e:
            print(f"An error occurred while submitting order for {symbol}: {e}")
            return None

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Retrieves the status of a specific order by its ID.

        Args:
            order_id: The unique ID of the order to check.

        Returns:
            An Alpaca `Order` entity containing the latest status, or None if
            the order is not found or an error occurs.
        """
        try:
            order = self.api.get_order(order_id)
            return order
        except Exception as e:
            print(f"An error occurred while retrieving status for order {order_id}: {e}")
            return None

    def list_positions(self) -> List[Position]:
        """Retrieves a list of all open positions in the Alpaca account.

        Returns:
            A list of Alpaca `Position` entities. Returns an empty list if
            there are no positions or if an error occurs.
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
