# Autonomous Trading Signal System

This repository contains a sophisticated, event-driven autonomous trading system designed for signal generation, aggregation, and execution. It is built with a modular, microservice-oriented architecture, leveraging modern technologies like Docker, Redis Streams, TimescaleDB, and FastAPI to create a robust and scalable platform for algorithmic trading.

The system not only covers the full trade lifecycle from data ingestion to execution but also incorporates a suite of advanced, institutional-grade features for enhanced alpha generation and risk management.

## Key Features

- **Event-Driven Architecture:** Core services communicate asynchronously via a Redis Streams message bus, ensuring loose coupling and scalability.
- **Modular & Adaptive Signal Engine:** The system supports a wide range of alpha sources:
    - **Technical Analysis**: Classic indicators like RSI and MACD.
    - **Machine Learning**: A predictive model trained on historical features.
    - **LLM-Powered Sentiment Analysis**: Ingests real-time news from Polygon.io and uses an LLM to gauge market sentiment.
    - **Order Book Analytics**: Analyzes market microstructure and order flow dynamics (e.g., VPIN, order imbalance) to generate high-frequency signals.
- **Adaptive Signal Aggregation:** Raw signals are intelligently weighted based on the historical performance (hit rate) of their source strategy, ensuring that more successful strategies have a greater influence on the final decision.
- **Reinforcement Learning Optimization**: A Deep Q-Network (DQN) agent continuously learns and adjusts strategy parameters and risk configurations in real-time, adapting the system's behavior to changing market regimes.
- **Enterprise-Grade Risk Management**: Goes beyond simple rule checks to include:
    - **Value at Risk (VaR)** and **Conditional VaR (CVaR)** calculations.
    - **Stress Testing** against predefined market shock scenarios.
    - **Real-time monitoring** and "circuit breaker" capabilities.
- **Cross-Exchange Arbitrage Detection**: A dedicated service monitors multiple exchanges to identify and act on simple, triangular, and funding rate arbitrage opportunities.
- **High-Fidelity Backtesting:** Includes both a fast, vectorized backtester for rapid iteration and a high-fidelity, event-driven backtester that replays historical data through the live event bus for realistic simulations.
- **LLM "Chief Analyst"**: A powerful LLM integration provides human-readable analysis and tactical playbooks for trading signals, adding a layer of explainability.
- **Containerized Deployment:** The entire system and its dependencies (databases, observability stack) are containerized with Docker and orchestrated with Docker Compose for easy, reproducible deployment.

## System Architecture

The system is composed of several microservices that communicate via Redis Streams:

1.  **Ingestion Service (`ingestion/`):** Connects to real-time market data feeds (e.g., WebSocket APIs from exchanges) and publishes raw trade data.
2.  **Data Resampler (`database/resampler.py`):** Consumes raw trades and creates multi-timeframe OHLCV candles.
3.  **Feature Calculator (`features/calculator.py`):** Calculates technical indicators from candles to build a feature store.
4.  **Signal Engine (`signals/`):** A collection of strategies (TA, ML, Sentiment) that generate raw trading signals.
5.  **Signal Aggregator (`aggregator/`):** Weights raw signals by historical performance to produce a single, high-confidence signal.
6.  **Orchestrator (`orchestrator/`):** The core of the trading loop. It listens for aggregated signals, validates them against risk rules, and instructs the Portfolio Manager to execute trades.
7.  **Portfolio Manager (`portfolio/`):** Manages the state of the trading portfolio (cash, positions) and executes trades.
8.  **Performance Tracker (`performance/`):** Attributes the P&L of closed trades back to the contributing strategies to update their performance metrics.

### Advanced Services

In addition to the core pipeline, the following advanced services run concurrently:

9.  **Risk Monitoring Service (`advanced_services.py`):** Periodically calculates and reports advanced risk metrics (VaR, CVaR) and checks for limit breaches.
10. **RL Optimizer Service (`advanced_services.py`):** The Reinforcement Learning agent that observes market conditions and performance, and dynamically adjusts strategy and risk parameters.
11. **Order Book Service (`advanced_services.py`):** Consumes Level 2 market data, calculates microstructure metrics, and publishes them as another source of raw signals.
12. **Arbitrage Service (`advanced_services.py`):** Connects to multiple exchanges to scan for and publish arbitrage opportunities.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- API keys for:
    - A trading broker like **Alpaca** (for paper/live execution)
    - **Polygon.io** (for news data)
    - **OpenAI** (for LLM-based sentiment and analysis)

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up environment variables:**
    Copy the `.env.example` file to `.env` and fill in your API keys and database credentials.
    ```bash
    cp .env.example .env
    # Edit .env with your credentials
    ```

3.  **Build and start all services:**
    The entire system can be started with a single Docker Compose command.
    ```bash
    sudo docker compose up -d --build
    ```
    *Note: `sudo` may be required depending on your Docker installation.*

4.  **Initialize the database schema:**
    Run the schema scripts to create and seed all necessary tables for both the core system and the advanced features.
    ```bash
    python database/schema.py
    python advanced_schema.py
    ```

5.  **Prepare the ML Model (Optional):**
    To use the machine learning strategy, you first need a labeled dataset and then to train the model.
    ```bash
    # This script generates a labeled dataset from your historical data
    python ml/data_labeling.py
    # This script trains the model and saves the artifact
    python ml/train.py
    ```

### Usage

- **Web Dashboard:** Access the main dashboard at `http://localhost:8000`. From here, you can view the API documentation, edit system configuration, and access quick links for other actions.
- **Running a Backtest:** The event-driven backtester can be run to simulate a strategy over a historical period. This requires the other services to be running.
  ```bash
  python backtester/main.py
  ```
- **Running Unit Tests:**
  ```bash
  # Ensure a test database is created first
  python -m unittest discover tests
  ```
