# Autonomous Trading Signal System

This repository contains a sophisticated, event-driven autonomous trading system designed for signal generation, aggregation, and execution. It is built with a modular, microservice-oriented architecture, leveraging modern technologies like Docker, Redis Streams, TimescaleDB, and FastAPI to create a robust and scalable platform for algorithmic trading.

## Key Features

- **Event-Driven Architecture:** Core services communicate asynchronously via a Redis Streams message bus, ensuring loose coupling and scalability.
- **Modular Signal Engine:** Strategies are implemented as independent modules that can be easily added or modified. The system supports technical indicator-based, machine learning-based, and even LLM-based (sentiment analysis) strategies.
- **Adaptive Signal Aggregation:** Raw signals from various strategies are weighted based on their historical performance (hit rate and P&L) to produce a single, high-confidence aggregated signal.
- **High-Fidelity Backtesting:** Includes both a fast, vectorized backtester for rapid iteration and a high-fidelity, event-driven backtester that replays historical data through the live event bus for realistic simulations.
- **Database-Driven State:** All critical state, including portfolio positions, performance metrics, and system configuration, is persisted in a robust TimescaleDB database.
- **Dynamic Configuration:** System parameters, strategy settings, and risk rules can be updated dynamically via a REST API and dashboard without requiring a service restart.
- **Pre-Execution Risk Management:** A dedicated risk validator acts as a guardrail, checking every potential trade against configurable rules (e.g., max position size, portfolio exposure) before execution.
- **LLM Integration:** A "Chief Analyst" module uses an LLM (e.g., GPT-4) to provide human-readable analysis and tactical playbooks for trading signals, adding a layer of explainability.
- **Containerized Deployment:** The entire system and its dependencies (databases, observability stack) are containerized with Docker and orchestrated with Docker Compose for easy, reproducible deployment.

## System Architecture

The system is composed of several microservices that communicate via Redis Streams:

1.  **Ingestion Service (`ingestion/`):** Connects to real-time market data feeds (e.g., WebSocket APIs from exchanges) and publishes raw trade data to the `raw_trades` stream.
2.  **Data Resampler (`database/resampler.py`):** Consumes raw trades and resamples them into multiple timeframes (1m, 5m, 15m, 1h), creating OHLCV candles. These candles are stored in TimescaleDB and published to dedicated streams (e.g., `candles_1m`).
3.  **Feature Calculator (`features/calculator.py`):** Listens for new candles and calculates a suite of technical indicators (RSI, MACD, etc.), saving them to feature tables in the database.
4.  **Signal Engine (`signals/`):** A collection of strategies that generate raw trading signals ('buy', 'sell', 'hold') based on the latest features.
5.  **Signal Aggregator (`aggregator/`):** Consumes raw signals, weights them using performance data from the `strategy_performance` table, and publishes a final, high-confidence signal to the `aggregated_signals` stream.
6.  **Orchestrator (`orchestrator/`):** The core of the trading loop. It listens for aggregated signals, validates them against risk rules, and instructs the Portfolio Manager to execute trades.
7.  **Portfolio Manager (`portfolio/`):** Manages the state of the trading portfolio (cash, positions), executes trades via a broker API, and publishes executed trades to the `executed_trades` stream.
8.  **Performance Tracker (`performance/`):** Consumes executed trades, calculates the P&L of each closed trade, and updates the performance metrics for the contributing strategies.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- An account with Alpaca (for paper/live trading), Polygon.io (for news data), and OpenAI (for LLM features), with API keys available.

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up environment variables:**
    Copy the `.env.example` file to `.env` and fill in your API keys for Alpaca, Polygon, and OpenAI, as well as your database credentials.
    ```bash
    cp .env.example .env
    # Edit .env with your credentials
    ```

3.  **Build and start all services:**
    The entire system, including the databases and application services, can be started with a single command.
    ```bash
    sudo docker compose up -d --build
    ```
    *Note: `sudo` may be required depending on your Docker installation.*

4.  **Initialize the database schema:**
    Run the schema script to create and seed the necessary tables in the PostgreSQL/TimescaleDB database.
    ```bash
    python database/schema.py
    ```

5.  **Train the ML Model (Optional):**
    To use the machine learning strategy, you first need to train the model. This requires a labeled dataset (`ml/labeled_data.csv`).
    ```bash
    python ml/train.py
    ```

### Usage

- **Web Dashboard:** Once the services are running, you can access the main dashboard in your browser at `http://localhost:8000`. From here, you can view and update the system configuration, check the latest signals, and run backtests.
- **Running a Backtest:** The event-driven backtester can be run directly to simulate a strategy over a historical period.
  ```bash
  python backtester/event_driven_engine.py
  ```
- **Running Tests:** The unit tests can be run to verify the functionality of individual components. A test database (`postgres_test`) is required.
  ```bash
  # Create the test database first
  python -m unittest discover tests
  ```
