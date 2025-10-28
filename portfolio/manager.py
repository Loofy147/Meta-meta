import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timezone

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk.validator import RiskValidator
from event_bus.publisher import EventPublisher

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def get_portfolio_status(portfolio_name='default'):
    """Retrieves the status of a portfolio, including cash, positions, and P&L."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get portfolio ID and cash
    cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
    portfolio_id, cash_balance = cursor.fetchone()

    # Get positions
    cursor.execute("SELECT symbol, quantity, average_entry_price FROM positions WHERE portfolio_id = %s;", (portfolio_id,))
    positions = []
    total_pnl = 0
    for symbol, quantity, avg_price in cursor.fetchall():
        # Get current price to calculate P&L
        cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
        current_price_result = cursor.fetchone()
        if current_price_result:
            current_price = current_price_result[0]
            pnl = (current_price - avg_price) * quantity
            total_pnl += pnl
            positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'average_entry_price': avg_price,
                'current_price': current_price,
                'pnl': pnl
            })

    conn.close()
    return {
        'portfolio_name': portfolio_name,
        'cash_balance': cash_balance,
        'total_pnl': total_pnl,
        'positions': positions
    }

def execute_trade(portfolio_name, symbol, side, quantity, signal_id):
    """Executes a trade and updates the portfolio in the database after passing risk validation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    publisher = EventPublisher()

    # Get current price
    cursor.execute("SELECT close FROM candles_1m WHERE symbol = %s ORDER BY time DESC LIMIT 1;", (symbol,))
    price_result = cursor.fetchone()
    if not price_result:
        print(f"Could not get current price for {symbol}. Aborting trade.")
        return
    price = price_result[0]

    trade_value = price * quantity

    # --- Pre-Execution Guard ---
    risk_validator = RiskValidator()
    if not risk_validator.validate_trade(portfolio_name, symbol, trade_value):
        print(f"Trade for {symbol} failed risk validation. Aborting.")
        return

    # Get portfolio ID and cash
    cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = %s;", (portfolio_name,))
    portfolio_id, cash_balance = cursor.fetchone()

    if side == 'buy':
        if cash_balance < trade_value:
            print("Insufficient funds to execute buy.")
            return

        # Update cash and position
        cursor.execute("UPDATE portfolios SET cash_balance = cash_balance - %s WHERE id = %s;", (trade_value, portfolio_id))
        cursor.execute("""
            INSERT INTO positions (portfolio_id, symbol, quantity, average_entry_price)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (portfolio_id, symbol) DO UPDATE
            SET quantity = positions.quantity + EXCLUDED.quantity,
                average_entry_price = (positions.average_entry_price * positions.quantity + %s * %s) / (positions.quantity + %s);
        """, (portfolio_id, symbol, quantity, price, price, quantity, quantity))

    elif side == 'sell':
        cursor.execute("SELECT quantity FROM positions WHERE portfolio_id = %s AND symbol = %s;", (portfolio_id, symbol))
        current_quantity = cursor.fetchone()
        if not current_quantity or current_quantity[0] < quantity:
            print("Insufficient position to execute sell.")
            return

        # Update cash and position
        cursor.execute("UPDATE portfolios SET cash_balance = cash_balance + %s WHERE id = %s;", (trade_value, portfolio_id))
        cursor.execute("UPDATE positions SET quantity = quantity - %s WHERE portfolio_id = %s AND symbol = %s;", (quantity, portfolio_id, symbol))

    # Record the trade
    cursor.execute("""
        INSERT INTO executed_trades (portfolio_id, symbol, quantity, price, side, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
    """, (portfolio_id, symbol, quantity, price, side, datetime.now(timezone.utc)))
    trade_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    # Publish the executed trade to the event bus
    publisher.publish('executed_trades', {
        'trade_id': trade_id,
        'signal_id': signal_id,
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price
    })

    print(f"Executed {side} of {quantity} {symbol} at {price}.")

if __name__ == '__main__':
    # Example usage
    print("Initial Portfolio Status:")
    print(get_portfolio_status())
