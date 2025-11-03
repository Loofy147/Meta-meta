"""Provides a FastAPI-based web API and dashboard for the trading system.

This module serves as the primary user-facing interface for interacting with
the trading system. It uses the FastAPI framework to expose a series of RESTful
API endpoints for critical functions, such as:
- Viewing and dynamically updating the system's configuration.
- Triggering on-demand signal generation for a specific asset.
- Viewing the current state of the trading portfolio.
- Requesting an LLM-powered analysis of a trading signal.
- Running historical backtests to evaluate strategy performance.

It also serves simple HTML pages for a basic web-based dashboard and
configuration editor.
"""

import sys
import os
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from datetime import date, datetime, timedelta
from typing import Dict, Any

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregator.main import aggregate_signals, get_strategy_weights
from portfolio.manager import get_portfolio_status
from llm.chief_analyst import generate_analysis
from backtester.vectorized_engine import run_backtest
from config.manager import get_config, update_config

app = FastAPI(
    title="Trading System API",
    description="A web API for monitoring and interacting with the autonomous trading system.",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse, summary="Serves the main dashboard page")
async def read_root() -> HTMLResponse:
    """Serves a simple HTML dashboard page.

    This root endpoint provides a basic landing page with links to the key
    interactive features of the API, such as the configuration editor and
    endpoints for getting signals and running backtests.

    Returns:
        An HTML response containing the dashboard page.
    """
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return HTMLResponse(content=f"""
    <html>
        <head><title>Trading System Dashboard</title></head>
        <body>
            <h1>Trading System API Dashboard</h1>
            <p>Welcome to the trading system's main control panel.</p>
            <ul>
                <li><h2><a href="/config">View & Edit System Configuration</a></h2></li>
                <li><h2><a href="/docs">API Documentation (Swagger UI)</a></h2></li>
            </ul>
            <h3>Quick Actions:</h3>
            <ul>
                <li><a href="/api/portfolio" target="_blank">View Current Portfolio</a></li>
                <li><a href="/api/signal?symbol=BTC/USDT" target="_blank">Get Latest Signal for BTC/USDT</a></li>
                <li><a href="/api/analysis?symbol=BTC/USDT" target="_blank">Get LLM Analysis for BTC/USDT</a></li>
                <li><a href="/api/backtest?symbol=BTC/USDT&start_date={yesterday}&end_date={today}" target="_blank">Run Backtest for BTC/USDT (Yesterday)</a></li>
            </ul>
        </body>
    </html>
    """)

@app.get("/config", response_class=HTMLResponse, summary="Serves the configuration editor page")
async def get_config_page() -> HTMLResponse:
    """Serves a simple HTML page for viewing and editing the system configuration.

    This page includes a textarea pre-filled with the current system
    configuration in JSON format and a button to save changes via a POST
    request to the `/api/config` endpoint.

    Returns:
        An HTML response containing the configuration editor.
    """
    return HTMLResponse(content="""
    <html>
        <head><title>System Configuration Editor</title></head>
        <body>
            <h1>System Configuration Editor</h1>
            <p>Modify the JSON below and click save to update the system's parameters in real-time.</p>
            <textarea id="config" rows="30" cols="120" style="font-family: monospace;"></textarea><br><br>
            <button onclick="saveConfig()">Save Configuration</button>
            <p id="status"></p>
            <script>
                fetch('/api/config')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('config').value = JSON.stringify(data, null, 2);
                    });

                function saveConfig() {
                    let statusEl = document.getElementById('status');
                    statusEl.innerText = 'Saving...';
                    statusEl.style.color = 'black';
                    fetch('/api/config', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: document.getElementById('config').value
                    }).then(async response => {
                        if (response.ok) {
                            statusEl.innerText = 'Configuration saved successfully!';
                            statusEl.style.color = 'green';
                        } else {
                            const error = await response.json();
                            statusEl.innerText = `Error saving configuration: ${error.detail}`;
                            statusEl.style.color = 'red';
                        }
                    });
                }
            </script>
        </body>
    </html>
    """)

@app.get("/api/config", summary="Retrieves the current system configuration")
async def get_system_config() -> Dict[str, Any]:
    """Retrieves the complete system configuration object from the database.

    This configuration is cached in memory for performance but is fetched from
    the source of truth (the database) on the first call.

    Returns:
        A dictionary representing the system's JSON configuration.
    """
    return get_config()

@app.post("/api/config", summary="Updates the system configuration")
async def update_system_config(request: Request) -> Dict[str, str]:
    """Updates the system configuration in the database with the provided JSON.

    This endpoint receives a raw JSON body and uses the `config_manager` to
    update the configuration in the database. This change will be picked up by
    all services on their next configuration fetch.

    Args:
        request: The incoming FastAPI request object containing the JSON body.

    Returns:
        A dictionary with a "status" key indicating success.

    Raises:
        HTTPException: If the request body is not valid JSON.
    """
    try:
        new_config = await request.json()
        update_config(new_config)
        return {"status": "success", "message": "Configuration updated."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

@app.get("/api/signal", summary="Generates and returns the latest aggregated signal")
async def get_signal(symbol: str = Query("BTC/USDT", description="The asset symbol to get a signal for")) -> Dict[str, Any]:
    """Triggers an on-demand signal aggregation for a specified symbol.

    This endpoint runs the full signal generation and aggregation pipeline to
    produce a final, weighted signal, which is then returned.

    Args:
        symbol: The trading symbol (e.g., 'BTC/USDT').

    Returns:
        The aggregated signal dictionary.
    """
    strategy_weights = get_strategy_weights()
    return aggregate_signals(symbol, strategy_weights)

@app.get("/api/portfolio", summary="Retrieves the current portfolio status")
async def get_portfolio_status_endpoint() -> Dict[str, Any]:
    """Fetches the real-time status of the default trading portfolio.

    This includes cash balance, a list of all open positions, and the
    unrealized P&L for each position.

    Returns:
        A dictionary detailing the portfolio's current state.
    """
    return get_portfolio_status('default')

@app.get("/api/analysis", summary="Generates an LLM-powered analysis of the latest signal")
async def get_llm_analysis(symbol: str = Query("BTC/USDT", description="The asset symbol to analyze")) -> Dict[str, Any]:
    """Generates an LLM-powered qualitative analysis and playbook.

    First, it generates the latest aggregated signal for the given symbol.
    Then, it passes this structured signal to the `chief_analyst` module to
    get a human-readable interpretation.

    Args:
        symbol: The trading symbol (e.g., 'BTC/USDT').

    Returns:
        A dictionary containing the 'analysis' and 'playbook' from the LLM.
    """
    strategy_weights = get_strategy_weights()
    signal = aggregate_signals(symbol, strategy_weights)
    return generate_analysis(signal)

@app.get("/api/backtest", summary="Runs a historical vectorized backtest")
async def run_backtest_endpoint(
    symbol: str = Query("BTC/USDT", description="The symbol to backtest"),
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD format")
) -> Dict[str, Any]:
    """Runs a fast, vectorized backtest for a given symbol and date range.

    This endpoint uses the vectorized backtesting engine to quickly simulate
    a strategy's performance over historical data.

    Args:
        symbol: The trading symbol to backtest.
        start_date: The start date for the backtest period.
        end_date: The end date for the backtest period.

    Returns:
        A dictionary of key performance indicators (KPIs) from the backtest.
    """
    return run_backtest(symbol, start_date.isoformat(), end_date.isoformat())
