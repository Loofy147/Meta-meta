"""
Trading System Dashboard API

This module provides a web-facing API for interacting with the trading system.
It uses the FastAPI framework to expose endpoints for viewing configuration,
checking signals, running backtests, and more.
"""

import sys
import os
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from datetime import date, datetime, timedelta
from typing import Dict, Any

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals, get_strategy_weights
from portfolio.manager import get_portfolio_status
from llm.chief_analyst import generate_analysis
from backtester.vectorized_engine import run_backtest
from config.manager import get_config, update_config

app = FastAPI(title="Trading System API", description="An API for interacting with the autonomous trading system.")

@app.get("/", response_class=HTMLResponse, summary="Main Dashboard Page")
async def read_root() -> HTMLResponse:
    """
    Serves the main HTML dashboard page.

    This page provides an overview of the available API endpoints and links for
    easy navigation.
    """
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return HTMLResponse(content=f"""
    <html>
        <head><title>Trading System Dashboard</title></head>
        <body>
            <h1>Trading System Dashboard</h1>
            <ul>
                <li><h2><a href="/config">View/Edit System Configuration</a></h2></li>
                <li><h2><a href="/api/signal?symbol=BTC/USDT">Get Latest Signal for BTC/USDT</a></h2></li>
                <li><h2><a href="/api/portfolio">View Current Portfolio</a></h2></li>
                <li><h2><a href="/api/analysis?symbol=BTC/USDT">Get LLM Analysis for BTC/USDT</a></h2></li>
                <li><h2><a href="/api/backtest?symbol=BTC/USDT&start_date={yesterday}&end_date={today}">Run Backtest for BTC/USDT</a></h2></li>
            </ul>
        </body>
    </html>
    """)

@app.get("/config", response_class=HTMLResponse, summary="Configuration Editor Page")
async def get_config_page() -> HTMLResponse:
    """
    Serves a simple HTML page for viewing and editing the system configuration.

    This provides a user-friendly way to dynamically adjust strategy parameters
    and risk settings without redeploying the application.
    """
    return HTMLResponse(content="""
    <html>
        <head><title>System Configuration</title></head>
        <body>
            <h1>System Configuration</h1>
            <textarea id="config" rows="25" cols="100" style="font-family: monospace;"></textarea><br><br>
            <button onclick="saveConfig()">Save Configuration</button>
            <p id="status"></p>
            <script>
                fetch('/api/config')
                    .then(response => response.json())
                    .then(data => document.getElementById('config').value = JSON.stringify(data, null, 2));

                function saveConfig() {
                    document.getElementById('status').innerText = 'Saving...';
                    fetch('/api/config', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: document.getElementById('config').value
                    }).then(response => {
                        if (response.ok) {
                            document.getElementById('status').innerText = 'Configuration saved successfully!';
                        } else {
                            document.getElementById('status').innerText = 'Error saving configuration.';
                        }
                    });
                }
            </script>
        </body>
    </html>
    """)

@app.get("/api/config", summary="Get System Configuration")
async def get_system_config() -> Dict[str, Any]:
    """
    Retrieves the current system configuration from the database.
    """
    return get_config()

@app.post("/api/config", summary="Update System Configuration")
async def update_system_config(request: Request) -> Dict[str, str]:
    """
    Updates the system configuration in the database with the provided JSON.
    """
    try:
        new_config = await request.json()
        update_config(new_config)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

@app.get("/api/signal", summary="Get Latest Aggregated Signal")
async def get_signal(symbol: str = Query("BTC/USDT", description="The symbol to get a signal for, e.g., 'BTC/USDT'")) -> Dict[str, Any]:
    """
    Generates and returns the latest aggregated signal for a given symbol.
    """
    strategy_weights = get_strategy_weights()
    return aggregate_signals(symbol, strategy_weights)

@app.get("/api/portfolio", summary="Get Portfolio Status")
async def get_portfolio_status_endpoint() -> Dict[str, Any]:
    """
    Retrieves the current status of the default trading portfolio.
    """
    return get_portfolio_status('default')

@app.get("/api/analysis", summary="Get LLM-Powered Signal Analysis")
async def get_llm_analysis(symbol: str = Query("BTC/USDT", description="The symbol to analyze, e.g., 'BTC/USDT'")) -> Dict[str, Any]:
    """
    Generates an LLM-powered analysis and playbook for the latest signal.
    """
    strategy_weights = get_strategy_weights()
    signal = aggregate_signals(symbol, strategy_weights)
    return generate_analysis(signal)

@app.get("/api/backtest", summary="Run a Vectorized Backtest")
async def run_backtest_endpoint(
    symbol: str = Query("BTC/USDT", description="The symbol to backtest, e.g., 'BTC/USDT'"),
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD format")
) -> Dict[str, Any]:
    """
    Runs a vectorized backtest for a given symbol and date range.
    """
    return run_backtest(symbol, start_date.isoformat(), end_date.isoformat())
