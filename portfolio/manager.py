"""Manages the state and execution of trades for all trading portfolios.

This module provides the core logic for portfolio management. It is responsible
for executing trades, updating cash balances and positions, and retrieving the
real-time status of any portfolio.

All state-changing operations are performed within database transactions to
ensure data integrity and atomicity. The manager persists all state to the
database, allowing for a robust system that can recover its state after a
restart. It is the sole gateway for trade execution, abstracting away the
details of broker interactions.
"""

import psycopg2
import os
import sys
import json
import uuid
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from psycopg2.extensions import connection

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk.validator import RiskValidator
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    Uses credentials from environment variables (DB_HOST, DB_NAME, etc.).

    Returns:
        A psycopg2 database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_portfolio_status(portfolio_name: str = 'default') -> Dict[str, Any]:
    """Retrieves the complete real-time status of a specified portfolio.

    This function queries the database to get the current cash balance and all
    open positions. It then calculates the unrealized Profit and Loss (P&L) for
    each position based on the latest available market price and computes the
    total portfolio value.

    Args:
        portfolio_name: The name of the portfolio to retrieve.

    Returns:
        A dictionary summarizing the portfolio's status, including cash
        balance, total P&L, total value, and a detailed list of all
        open positions with their individual P&L.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Retrieve portfolio ID and cash balance.
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found.")
            portfolio_id, cash_balance = result

            # Retrieve all open positions for the portfolio.
            cursor.execute("SELECT symbol, quantity, average_entry_price FROM positions WHERE portfolio_id = %s;", (portfolio_id,))
            positions_data = cursor.fetchall()

            positions: List[Dict[str, Any]] = []
            total_unrealized_pnl = 0.0
            total_position_value = 0.0

            for symbol, quantity, avg_price in positions_data:
                # Fetch the latest price to calculate unrealized P&L.
                cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
                current_price_result = cursor.fetchone()
                # Fallback to entry price if no recent candle is found.
                current_price = current_price_result[0] if current_price_result else avg_price

                pnl = (current_price - avg_price) * quantity
                total_unrealized_pnl += pnl
                total_position_value += quantity * current_price
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
                'total_unrealized_pnl': total_unrealized_pnl,
                'portfolio_value': cash_balance + total_position_value,
                'positions': positions
            }
    finally:
        conn.close()

def execute_trade(portfolio_name: str, symbol: str, side: str, quantity: float, signal_id: Optional[str] = None) -> None:
    """Executes a trade and updates the portfolio state within a transaction.

    This is a critical, state-changing function that handles the logic for
    buying and selling assets. It operates within a database transaction to
    ensure that all related updates (cash, positions) either succeed or fail
    together, preventing inconsistent state.

    The logic includes:
    - Fetching the latest price for execution.
    - Final pre-execution risk validation.
    - For 'buy' trades: Decreasing cash and creating or increasing a position,
      recalculating the average entry price.
    - For 'sell' trades: Increasing cash and decreasing a position, removing the
      position if fully sold.
    - Publishing the confirmed executed trade to the event bus for downstream
      performance tracking.

    Args:
        portfolio_name: The name of the portfolio to execute the trade in.
        symbol: The symbol of the asset to trade (e.g., 'BTC/USDT').
        side: The direction of the trade ('buy' or 'sell').
        quantity: The amount of the asset to trade.
        signal_id: The unique ID of the signal that initiated this trade,
            used for performance attribution.
    """
    conn = get_db_connection()
    publisher = EventPublisher()

    try:
        with conn.cursor() as cursor:
            # Fetch the latest market price to execute the trade at.
            cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
            price_result = cursor.fetchone()
            if not price_result:
                print(f"Could not get current price for {symbol}. Aborting trade.")
                return
            price = price_result[0]

            trade_value = price * quantity

            # This is a final, authoritative risk check just before execution.
            risk_validator = RiskValidator(conn)
            if not risk_validator.is_trade_safe(portfolio_name, symbol, quantity, trade_value):
                print(f"Trade for {symbol} failed final risk validation. Aborting.")
                return

            # Retrieve portfolio details needed for the transaction.
            cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s FOR UPDATE;", (portfolio_name,))
            portfolio_id, cash_balance = cursor.fetchone()

            if side == 'buy':
                if cash_balance < trade_value:
                    print("Insufficient funds to execute buy.")
                    return
                # Atomically update cash and create/update the position.
                cursor.execute("UPDATE portfolios SET cash_balance = cash_balance - %s WHERE id = %s;", (trade_value, portfolio_id))
                # This complex query handles both new and existing positions in one step.
                cursor.execute("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, average_entry_price)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (portfolio_id, symbol) DO UPDATE
                    SET average_entry_price = (positions.average_entry_price * positions.quantity + %s * %s) / (positions.quantity + %s),
                        quantity = positions.quantity + EXCLUDED.quantity;
                """, (portfolio_id, symbol, quantity, price, price, quantity, quantity))

            elif side == 'sell':
                cursor.execute("SELECT quantity FROM positions WHERE portfolio_id = %s AND symbol = %s FOR UPDATE;", (portfolio_id, symbol))
                pos_result = cursor.fetchone()
                if not pos_result or pos_result[0] < quantity:
                    print(f"Insufficient position in {symbol} to sell. Have {pos_result[0] if pos_result else 0}, need {quantity}.")
                    return
                # Atomically update cash and the position quantity.
                cursor.execute("UPDATE portfolios SET cash_balance = cash_balance + %s WHERE id = %s;", (trade_value, portfolio_id))
                cursor.execute("UPDATE positions SET quantity = quantity - %s WHERE portfolio_id = %s AND symbol = %s;", (quantity, portfolio_id, symbol))
                # Clean up the position record if it has been fully sold.
                cursor.execute("DELETE FROM positions WHERE portfolio_id = %s AND symbol = %s AND quantity = 0;", (portfolio_id, symbol))

            # If all DB operations succeed, publish the event for the performance tracker.
            trade_signal_id = signal_id if signal_id is not None else str(uuid.uuid4())
            publisher.publish('executed_trades', {
                'signal_id': trade_signal_id, 'symbol': symbol, 'side': side,
                'quantity': quantity, 'price': price,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            conn.commit()
            print(f"Successfully executed {side} of {quantity} {symbol} at ${price:.2f}.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"An error occurred during trade execution, transaction rolled back: {e}")
    finally:
        if conn:
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
