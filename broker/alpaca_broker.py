import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

class AlpacaBroker:
    """
    A broker class for interacting with the Alpaca API.
    """
    def __init__(self):
        # Note: This requires ALPACA_API_KEY and ALPACA_SECRET_KEY to be set in the .env file
        # For paper trading, the base_url should be the paper trading URL
        self.api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper trading
        )

    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        """
        Submits an order to the broker.
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
            print(f"An error occurred while submitting the order: {e}")
            return None

    def get_order_status(self, order_id):
        """
        Retrieves the status of a specific order.
        """
        try:
            order = self.api.get_order(order_id)
            return order
        except Exception as e:
            print(f"An error occurred while retrieving the order status: {e}")
            return None

    def list_positions(self):
        """
        Lists all open positions.
        """
        try:
            positions = self.api.list_positions()
            return positions
        except Exception as e:
            print(f"An error occurred while listing positions: {e}")
            return []

if __name__ == '__main__':
    # Example Usage (requires API keys in .env)
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        print("Skipping broker example: Alpaca API keys not found in .env file.")
    else:
        broker = AlpacaBroker()

        # --- List Positions ---
        print("Current Positions:")
        positions = broker.list_positions()
        if positions:
            for position in positions:
                print(f"  - {position.symbol}: {position.qty}")
        else:
            print("  No open positions.")

        # --- Submit an Order ---
        # Note: This will submit a real paper trading order if keys are configured.
        # print("\\nSubmitting a test order...")
        # test_order = broker.submit_order(symbol='BTC/USD', qty=0.01, side='buy') # Alpaca uses BTC/USD for crypto
        # if test_order:
        #     print(f"Test order submitted with ID: {test_order.id}")

        #     # --- Check Order Status ---
        #     import time
        #     time.sleep(2) # Give the order time to process
        #     status = broker.get_order_status(test_order.id)
        #     if status:
        #         print(f"\\nOrder Status for {test_order.id}: {status.status}")
