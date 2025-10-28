import ccxt.pro
import asyncio
import psycopg2
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establishes and returns a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

async def watch_and_store_trades(exchange, symbol, conn):
    """Watches trades for a single symbol and stores them in the database."""
    cursor = conn.cursor()
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                cursor.execute(
                    "INSERT INTO trades (time, symbol, price, amount, side) VALUES (%s, %s, %s, %s, %s)",
                    (trade['datetime'], trade['symbol'], trade['price'], trade['amount'], trade['side'])
                )
            conn.commit()
            print(f"[{symbol}] Inserted {len(trades)} trades.")
        except Exception as e:
            print(f"An error occurred while watching {symbol}: {e}")
            break

async def main():
    """Main function to run the data ingestion."""
    with open('config/main.json', 'r') as f:
        config = json.load(f)['ingestion']

    exchange_name = config.get('exchange', 'kucoin')
    symbols = config.get('symbols', ['BTC/USDT'])

    try:
        exchange_class = getattr(ccxt.pro, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' not found.")
        return

    conn = get_db_connection()
    tasks = [watch_and_store_trades(exchange, symbol, conn) for symbol in symbols]

    print(f"Starting data ingestion for {len(symbols)} symbols on {exchange_name}...")
    await asyncio.gather(*tasks)

    await exchange.close()
    conn.close()
    print("Data ingestion stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nData ingestion stopped by user.")
