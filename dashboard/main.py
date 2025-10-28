import sys
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signal
from paper_trader.main import get_portfolio

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # A very simple HTML frontend
    return """
    <html>
        <head>
            <title>Trading System Dashboard</title>
        </head>
        <body>
            <h1>Trading System Dashboard</h1>
            <h2><a href="/signal">Latest Signal</a></h2>
            <h2><a href="/portfolio">Current Portfolio</a></h2>
        </body>
    </html>
    """

@app.get("/signal")
async def get_signal():
    return aggregate_signal()

@app.get("/portfolio")
async def get_portfolio_status():
    return get_portfolio()
