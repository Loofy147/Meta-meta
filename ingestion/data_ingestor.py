import ccxt.pro
import asyncio
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    exchange = ccxt.pro.kucoin()
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )
    cursor = conn.cursor()

    while True:
        try:
            trades = await exchange.watch_trades('BTC/USDT')
            for trade in trades:
                cursor.execute(
                    "INSERT INTO trades (time, symbol, price, amount, side) VALUES (%s, %s, %s, %s, %s)",
                    (trade['datetime'], trade['symbol'], trade['price'], trade['amount'], trade['side'])
                )
            conn.commit()
            print(f"Inserted {len(trades)} trades.")
        except Exception as e:
            print(f"An error occurred: {e}")
            await exchange.close()
            conn.close()
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Data ingestion stopped.")
