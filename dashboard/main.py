import sys
import os
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from datetime import date

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals
from portfolio.manager import get_portfolio_status
from llm.chief_analyst import generate_analysis
from backtester.vectorized_engine import run_backtest
from config.manager import get_config, update_config

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    today = date.today().isoformat()
    return f"""
    <html>
        <head><title>Trading System Dashboard</title></head>
        <body>
            <h1>Trading System Dashboard</h1>
            <h2><a href="/config">System Configuration</a></h2>
            <h2><a href="/signal?symbol=BTC/USDT">Latest Signal for BTC/USDT</a></h2>
            <h2><a href="/portfolio">Current Portfolio</a></h2>
            <h2><a href="/analysis?symbol=BTC/USDT">LLM Analysis for BTC/USDT</a></h2>
            <h2><a href="/backtest?symbol=BTC/USDT&start_date={today}&end_date={today}">Run Backtest for BTC/USDT</a></h2>
        </body>
    </html>
    """

@app.get("/config", response_class=HTMLResponse)
async def get_config_page():
    # A simple UI for viewing and editing the config
    return """
    <html>
        <head><title>System Configuration</title></head>
        <body>
            <h1>System Configuration</h1>
            <textarea id="config" rows="20" cols="80"></textarea><br>
            <button onclick="saveConfig()">Save Configuration</button>
            <script>
                fetch('/api/config')
                    .then(response => response.json())
                    .then(data => document.getElementById('config').value = JSON.stringify(data, null, 2));

                function saveConfig() {
                    fetch('/api/config', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: document.getElementById('config').value
                    }).then(() => alert('Configuration saved!'));
                }
            </script>
        </body>
    </html>
    """

@app.get("/api/config")
async def get_system_config():
    return get_config()

@app.post("/api/config")
async def update_system_config(request: Request):
    new_config = await request.json()
    update_config(new_config)
    return {"status": "success"}

@app.get("/signal")
async def get_signal(symbol: str = Query("BTC/USDT")):
    # This is a placeholder for a more robust way of getting the latest signal
    # In a real system, this would likely read from a cache or the event bus
    strategy_weights = {} # Placeholder
    return aggregate_signals(symbol, strategy_weights)

@app.get("/portfolio")
async def get_portfolio_status_endpoint():
    return get_portfolio_status('default')

@app.get("/analysis")
async def get_llm_analysis(symbol: str = Query("BTC/USDT")):
    strategy_weights = {} # Placeholder
    signal = aggregate_signals(symbol, strategy_weights)
    return generate_analysis(signal)

@app.get("/backtest")
async def run_backtest_endpoint(symbol: str = Query("BTC/USDT"), start_date: str = Query(...), end_date: str = Query(...)):
    start_str = f"{start_date} 00:00:00"
    end_str = f"{end_date} 23:59:59"
    return run_backtest(symbol, start_str, end_str)
