import sys
import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from datetime import date

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals_for_symbol
from portfolio.manager import get_portfolio_status
from llm.chief_analyst import generate_analysis
from backtester.engine import run_backtest

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # A very simple HTML frontend
    today = date.today().isoformat()
    return f"""
    <html>
        <head>
            <title>Trading System Dashboard</title>
        </head>
        <body>
            <h1>Trading System Dashboard</h1>
            <h2><a href="/signal?symbol=BTC/USDT">Latest Signal for BTC/USDT</a></h2>
            <h2><a href="/portfolio">Current Portfolio</a></h2>
            <h2><a href="/analysis?symbol=BTC/USDT">LLM Analysis for BTC/USDT</a></h2>
            <h2><a href="/backtest?symbol=BTC/USDT&start_date={today}&end_date={today}">Run Backtest for BTC/USDT</a></h2>
        </body>
    </html>
    """

@app.get("/signal")
async def get_signal(symbol: str = Query("BTC/USDT")):
    return aggregate_signals_for_symbol(symbol)

@app.get("/portfolio")
async def get_portfolio():
    return get_portfolio_status('default')

@app.get("/analysis")
async def get_llm_analysis(symbol: str = Query("BTC/USDT")):
    signal = aggregate_signals_for_symbol(symbol)
    return generate_analysis(signal)

@app.get("/backtest")
async def run_backtest_endpoint(symbol: str = Query("BTC/USDT"), start_date: str = Query(...), end_date: str = Query(...)):
    start_str = f"{start_date} 00:00:00"
    end_str = f"{end_date} 23:59:59"
    return run_backtest(symbol, start_str, end_str)
