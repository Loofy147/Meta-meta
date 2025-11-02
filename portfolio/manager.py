"""
Portfolio Manager

This module is responsible for managing the state of the trading portfolios. It
handles the core logic of executing trades, updating positions and cash balances,
and tracking the overall value and performance of a portfolio.

All state changes are persisted to the database, ensuring that the system is
robust and can recover its state after a restart.
"""

import psycopg2
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk.validator import RiskValidator
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection() -> connection:
    """
    Establishes and returns a connection to the PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: A database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_portfolio_status(portfolio_name: str = 'default') -> Dict[str, Any]:
    """
    Retrieves the complete real-time status of a specified portfolio.

    This includes the current cash balance, a list of all open positions with
    their unrealized P&L, and the total P&L for the portfolio.

    Args:
        portfolio_name (str): The name of the portfolio to retrieve.

    Returns:
        Dict[str, Any]: A dictionary containing the portfolio's status.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Get portfolio ID and cash
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
            portfolio_id, cash_balance = cursor.fetchone()

            # Get positions
            cursor.execute("SELECT symbol, quantity, average_entry_price FROM positions WHERE portfolio_id = %s;", (portfolio_id,))
            positions_data = cursor.fetchall()

            positions: List[Dict[str, Any]] = []
            total_pnl = 0.0

            for symbol, quantity, avg_price in positions_data:
                # Get the latest price to calculate unrealized P&L
                cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
                current_price_result = cursor.fetchone()
                current_price = current_price_result[0] if current_price_result else avg_price

                pnl = (current_price - avg_price) * quantity
                total_pnl += pnl
                positions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'average_entry_price': avg_price,
                    'current_price': current_price,
                    'unrealized_pnl': pnl
                })

            return {
                'portfolio_name': portfolio_name,
                'cash_balance': cash_balance,
                'total_unrealized_pnl': total_pnl,
                'portfolio_value': cash_balance + sum(p['quantity'] * p['current_price'] for p in positions),
                'positions': positions
            }
    finally:
        conn.close()

def execute_trade(portfolio_name: str, symbol: str, side: str, quantity: float, signal_id: str) -> None:
    """
    Executes a trade, updating the portfolio state in the database.

    This function is the final step in the autonomous loop. It handles updating
    cash balances and positions, records the executed trade, and publishes the
    trade to the event bus for performance tracking.

    Args:
        portfolio_name (str): The name of the portfolio to trade in.
        symbol (str): The symbol of the asset to trade.
        side (str): The direction of the trade ('buy' or 'sell').
        quantity (float): The amount of the asset to trade.
        signal_id (str): The ID of the signal that initiated this trade.
    """
    conn = get_db_connection()
    publisher = EventPublisher()

    try:
        with conn.cursor() as cursor:
            # Get the latest market price for execution
            cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
            price_result = cursor.fetchone()
            if not price_result:
                print(f"Could not get current price for {symbol}. Aborting trade.")
                return
            price = price_result[0]

            trade_value = price * quantity

            # --- Pre-Execution Risk Validation ---
            risk_validator = RiskValidator(conn)
            if not risk_validator.is_trade_safe(portfolio_name, symbol, quantity, trade_value):
                print(f"Trade for {symbol} failed risk validation. Aborting.")
                return

            # Get portfolio ID and current state
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
            portfolio_id, cash_balance = cursor.fetchone()

            if side == 'buy':
                if cash_balance < trade_value:
                    print("Insufficient funds to execute buy.")
                    return
                # Atomically update cash and position
                cursor.execute("UPDATE portfolios SET cash_balance = cash_balance - %s WHERE id = %s;", (trade_value, portfolio_id))
                cursor.execute("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, average_entry_price)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (portfolio_id, symbol) DO UPDATE
                    SET average_entry_price = (positions.average_entry_price * positions.quantity + %s * %s) / (positions.quantity + %s),
                        quantity = positions.quantity + EXCLUDED.quantity;
                """, (portfolio_id, symbol, quantity, price, price, quantity, quantity))

            elif side == 'sell':
                cursor.execute("SELECT quantity FROM positions WHERE portfolio_id = %s AND symbol = %s;", (portfolio_id, symbol))
                current_quantity = cursor.fetchone()
                if not current_quantity or current_quantity[0] < quantity:
                    print(f"Insufficient position in {symbol} to execute sell. Have {current_quantity[0] if current_quantity else 0}, need {quantity}.")
                    return
                cursor.execute("UPDATE portfolios SET cash_balance = cash_balance + %s WHERE id = %s;", (trade_value, portfolio_id))
                cursor.execute("UPDATE positions SET quantity = quantity - %s WHERE portfolio_id = %s AND symbol = %s;", (quantity, portfolio_id, symbol))
                # Clean up position if fully sold
                cursor.execute("DELETE FROM positions WHERE portfolio_id = %s AND symbol = %s AND quantity = 0;", (portfolio_id, symbol))

            # Publish the executed trade for the performance tracker
            publisher.publish('executed_trades', {
                'signal_id': signal_id, 'symbol': symbol, 'side': side,
                'quantity': quantity, 'price': price,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            conn.commit()
            print(f"Successfully executed {side} of {quantity} {symbol} at ${price:.2f}.")

    except Exception as e:
        conn.rollback()
        print(f"An error occurred during trade execution: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    print("--- Portfolio Manager Example ---")
    print("\nInitial Portfolio Status:")
    print(json.dumps(get_portfolio_status(), indent=2))

    # # Example of executing a trade (uncomment to test)
    # print("\nExecuting a test buy order...")
    # execute_trade(
    #     portfolio_name='default',
    #     symbol='BTC/USDT',
    #     side='buy',
    #     quantity=0.01,
    #     signal_id=str(uuid.uuid4())
    # )
    # print("\nPortfolio Status After Buy:")
    # print(json.dumps(get_portfolio_status(), indent=2))
