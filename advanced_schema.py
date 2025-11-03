"""Defines and initializes database schema extensions for advanced features.

This script builds upon the core database schema by adding a suite of new
tables, indexes, views, and functions required to support the system's
advanced capabilities. It is designed to be idempotent and can be run safely
on an existing database.

The extensions cover several key areas:
- **Order Book Analytics**: Tables for storing high-frequency order book
  metrics like imbalance, VPIN, and liquidity scores.
- **Reinforcement Learning**: Tables for logging RL agent decisions, tracking
  episode performance, and managing dynamically updated strategy weights.
- **Enterprise Risk Management**: Tables for historical VaR calculations,
  stress test results, and a log of risk alerts and violations.
- **Arbitrage Detection**: Tables for logging cross-exchange arbitrage
  opportunities and monitoring the health of exchange data feeds.
- **Enhanced Performance Tracking**: Detailed tables for closed trades,
  portfolio snapshots, and aggregated performance metrics over various periods.
- **System Monitoring**: Tables for tracking the health of individual
  microservices and logging critical system-wide events.
"""

import psycopg2
import os
from dotenv import load_dotenv
from psycopg2.extensions import connection

load_dotenv()


def get_db_connection() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    Uses credentials from environment variables (DB_HOST, DB_NAME, etc.).

    Returns:
        A psycopg2 database connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )


def create_advanced_schema():
    """Creates all new tables, views, and functions for advanced features.

    This function connects to the database and executes a series of DDL
    (Data Definition Language) commands to create the required schema extensions.
    It is wrapped in a transaction to ensure atomicity.
    """
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            print("Creating advanced schema extensions...")
            
            # Ensure TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # =====================================================
            # 1. ORDER BOOK ANALYTICS TABLES
            # =====================================================
            print("\n1. Creating order book analytics tables...")
            
            # Order flow metrics time series
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_metrics (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    order_imbalance DOUBLE PRECISION,
                    vpin DOUBLE PRECISION,
                    pressure_index DOUBLE PRECISION,
                    liquidity_score DOUBLE PRECISION,
                    toxicity_score DOUBLE PRECISION,
                    bid_depth DOUBLE PRECISION,
                    ask_depth DOUBLE PRECISION,
                    spread_bps DOUBLE PRECISION
                );
            """)
            cursor.execute("""
                SELECT create_hypertable('orderbook_metrics', 'time', 
                    if_not_exists => TRUE);
            """)
            
            # Order book snapshots (for replay and analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    bids JSONB NOT NULL,  -- Array of [price, size]
                    asks JSONB NOT NULL,
                    mid_price DOUBLE PRECISION,
                    spread DOUBLE PRECISION
                );
            """)
            cursor.execute("""
                SELECT create_hypertable('orderbook_snapshots', 'time', 
                    if_not_exists => TRUE);
            """)
            
            # =====================================================
            # 2. REINFORCEMENT LEARNING TABLES
            # =====================================================
            print("2. Creating RL agent tables...")
            
            # RL decisions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_decisions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    action INTEGER NOT NULL,
                    state_vector DOUBLE PRECISION[] NOT NULL,
                    config JSONB NOT NULL,
                    epsilon DOUBLE PRECISION,
                    portfolio_value DOUBLE PRECISION
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_decisions_timestamp 
                ON rl_decisions(timestamp DESC);
            """)
            
            # RL episode results (for tracking learning progress)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_episodes (
                    id SERIAL PRIMARY KEY,
                    episode_number INTEGER NOT NULL,
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ NOT NULL,
                    start_value DOUBLE PRECISION NOT NULL,
                    end_value DOUBLE PRECISION NOT NULL,
                    total_reward DOUBLE PRECISION NOT NULL,
                    sharpe_ratio DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    trades_count INTEGER,
                    win_rate DOUBLE PRECISION
                );
            """)
            
            # Strategy weights (dynamically updated by RL agent)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_weights (
                    strategy_name TEXT PRIMARY KEY,
                    weight DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    updated_by TEXT DEFAULT 'rl_agent'
                );
            """)
            
            # =====================================================
            # 3. ADVANCED RISK MANAGEMENT TABLES
            # =====================================================
            print("3. Creating advanced risk tables...")
            
            # VaR calculations history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS var_calculations (
                    id SERIAL PRIMARY KEY,
                    calculation_time TIMESTAMPTZ NOT NULL,
                    portfolio_name TEXT NOT NULL,
                    confidence_level DOUBLE PRECISION NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    var_historical DOUBLE PRECISION,
                    var_parametric DOUBLE PRECISION,
                    var_monte_carlo DOUBLE PRECISION,
                    cvar DOUBLE PRECISION,
                    portfolio_value DOUBLE PRECISION,
                    var_pct DOUBLE PRECISION
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_var_calc_time 
                ON var_calculations(calculation_time DESC);
            """)
            
            # Stress test results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    id SERIAL PRIMARY KEY,
                    test_time TIMESTAMPTZ NOT NULL,
                    portfolio_name TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    portfolio_value_before DOUBLE PRECISION NOT NULL,
                    expected_pnl DOUBLE PRECISION NOT NULL,
                    expected_pnl_pct DOUBLE PRECISION NOT NULL,
                    worst_position TEXT,
                    worst_position_pnl DOUBLE PRECISION,
                    passed BOOLEAN NOT NULL
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stress_test_time 
                ON stress_test_results(test_time DESC);
            """)
            
            # Risk alerts and violations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_time TIMESTAMPTZ NOT NULL,
                    alert_type TEXT NOT NULL,  -- 'warning', 'breach', 'critical'
                    portfolio_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold_value DOUBLE PRECISION,
                    actual_value DOUBLE PRECISION,
                    severity TEXT NOT NULL,
                    message TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMPTZ,
                    acknowledged_by TEXT
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_alerts_unack 
                ON risk_alerts(alert_time DESC) WHERE NOT acknowledged;
            """)
            
            # Kelly Criterion recommendations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kelly_recommendations (
                    id SERIAL PRIMARY KEY,
                    recommendation_time TIMESTAMPTZ NOT NULL,
                    strategy_name TEXT NOT NULL,
                    win_rate DOUBLE PRECISION NOT NULL,
                    avg_win DOUBLE PRECISION NOT NULL,
                    avg_loss DOUBLE PRECISION NOT NULL,
                    kelly_fraction DOUBLE PRECISION NOT NULL,
                    recommended_allocation DOUBLE PRECISION NOT NULL,
                    applied BOOLEAN DEFAULT FALSE
                );
            """)
            
            # =====================================================
            # 4. ARBITRAGE TABLES
            # =====================================================
            print("4. Creating arbitrage detection tables...")
            
            # Arbitrage opportunities detected
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                    id SERIAL PRIMARY KEY,
                    detected_at TIMESTAMPTZ NOT NULL,
                    opportunity_type TEXT NOT NULL,  -- 'simple', 'triangular', 'funding'
                    symbol TEXT NOT NULL,
                    buy_exchange TEXT NOT NULL,
                    sell_exchange TEXT NOT NULL,
                    buy_price DOUBLE PRECISION NOT NULL,
                    sell_price DOUBLE PRECISION NOT NULL,
                    profit_pct DOUBLE PRECISION NOT NULL,
                    profit_net DOUBLE PRECISION NOT NULL,
                    volume_limit DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    execution_window_ms INTEGER NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    execution_time TIMESTAMPTZ,
                    actual_profit DOUBLE PRECISION
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_arb_detected 
                ON arbitrage_opportunities(detected_at DESC);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_arb_unexecuted 
                ON arbitrage_opportunities(detected_at DESC) WHERE NOT executed;
            """)
            
            # Exchange orderbook feed status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchange_feed_status (
                    exchange TEXT PRIMARY KEY,
                    last_update TIMESTAMPTZ NOT NULL,
                    status TEXT NOT NULL,  -- 'active', 'degraded', 'down'
                    latency_ms DOUBLE PRECISION,
                    message_rate DOUBLE PRECISION,  -- messages per second
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    last_error_time TIMESTAMPTZ
                );
            """)
            
            # =====================================================
            # 5. ENHANCED PERFORMANCE TRACKING
            # =====================================================
            print("5. Creating enhanced performance tables...")
            
            # Daily portfolio snapshots (for historical analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id SERIAL PRIMARY KEY,
                    portfolio_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    current_price DOUBLE PRECISION NOT NULL,
                    position_value DOUBLE PRECISION NOT NULL,
                    unrealized_pnl DOUBLE PRECISION,
                    UNIQUE(portfolio_name, date, symbol)
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date 
                ON portfolio_snapshots(portfolio_name, date DESC);
            """)
            
            # Closed trades (for P&L tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id SERIAL PRIMARY KEY,
                    portfolio_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    opened_at TIMESTAMPTZ NOT NULL,
                    closed_at TIMESTAMPTZ NOT NULL,
                    direction TEXT NOT NULL,  -- 'long', 'short'
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    pnl DOUBLE PRECISION NOT NULL,
                    pnl_pct DOUBLE PRECISION NOT NULL,
                    fees DOUBLE PRECISION NOT NULL,
                    holding_period_hours DOUBLE PRECISION NOT NULL,
                    signal_id UUID,
                    strategy_source TEXT,
                    notes TEXT
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_closed_trades_time 
                ON closed_trades(closed_at DESC);
            """)
            
            # Strategy attribution (which strategies contributed to each trade)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_attribution (
                    trade_id INTEGER REFERENCES closed_trades(id) ON DELETE CASCADE,
                    strategy_name TEXT NOT NULL,
                    contribution_weight DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (trade_id, strategy_name)
                );
            """)
            
            # Performance metrics by period
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_periods (
                    id SERIAL PRIMARY KEY,
                    portfolio_name TEXT NOT NULL,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    period_type TEXT NOT NULL,  -- 'daily', 'weekly', 'monthly'
                    starting_value DOUBLE PRECISION NOT NULL,
                    ending_value DOUBLE PRECISION NOT NULL,
                    net_pnl DOUBLE PRECISION NOT NULL,
                    return_pct DOUBLE PRECISION NOT NULL,
                    sharpe_ratio DOUBLE PRECISION,
                    sortino_ratio DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    win_rate DOUBLE PRECISION,
                    profit_factor DOUBLE PRECISION,
                    total_trades INTEGER NOT NULL,
                    UNIQUE(portfolio_name, period_start, period_type)
                );
            """)
            
            # =====================================================
            # 6. SYSTEM MONITORING TABLES
            # =====================================================
            print("6. Creating system monitoring tables...")
            
            # Service health checks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    service_name TEXT PRIMARY KEY,
                    last_heartbeat TIMESTAMPTZ NOT NULL,
                    status TEXT NOT NULL,  -- 'healthy', 'degraded', 'down'
                    uptime_seconds BIGINT,
                    cpu_percent DOUBLE PRECISION,
                    memory_mb DOUBLE PRECISION,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    version TEXT
                );
            """)
            
            # System events log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id SERIAL PRIMARY KEY,
                    event_time TIMESTAMPTZ NOT NULL,
                    event_type TEXT NOT NULL,  -- 'info', 'warning', 'error', 'critical'
                    service_name TEXT,
                    event_category TEXT,  -- 'trading', 'risk', 'system', 'data'
                    message TEXT NOT NULL,
                    details JSONB,
                    resolved BOOLEAN DEFAULT FALSE
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_events_time 
                ON system_events(event_time DESC);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_events_type 
                ON system_events(event_type, event_time DESC);
            """)
            
            # =====================================================
            # 7. SEED INITIAL DATA
            # =====================================================
            print("7. Seeding initial data...")
            
            # Initialize strategy weights
            default_weights = {
                'rsi': 0.2,
                'macd': 0.2,
                'ml': 0.2,
                'sentiment': 0.2,
                'orderbook': 0.2
            }
            
            for strategy, weight in default_weights.items():
                cursor.execute("""
                    INSERT INTO strategy_weights (strategy_name, weight, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (strategy_name) DO NOTHING;
                """, (strategy, weight))
            
            # Initialize service health for monitoring
            services = [
                'ingestion',
                'resampler',
                'feature_calculator',
                'aggregator',
                'orchestrator',
                'portfolio_manager',
                'risk_monitor',
                'rl_optimizer',
                'arbitrage_scanner'
            ]
            
            for service in services:
                cursor.execute("""
                    INSERT INTO service_health (service_name, last_heartbeat, status)
                    VALUES (%s, NOW(), 'unknown')
                    ON CONFLICT (service_name) DO NOTHING;
                """, (service,))
            
            # =====================================================
            # 8. CREATE USEFUL VIEWS
            # =====================================================
            print("8. Creating analytical views...")
            
            # Latest risk metrics view
            cursor.execute("""
                CREATE OR REPLACE VIEW latest_risk_metrics AS
                SELECT DISTINCT ON (portfolio_name)
                    portfolio_name,
                    calculation_time,
                    var_historical,
                    var_parametric,
                    cvar,
                    var_pct,
                    portfolio_value
                FROM var_calculations
                ORDER BY portfolio_name, calculation_time DESC;
            """)
            
            # Recent performance view
            cursor.execute("""
                CREATE OR REPLACE VIEW recent_performance AS
                SELECT 
                    portfolio_name,
                    period_type,
                    ending_value,
                    return_pct,
                    sharpe_ratio,
                    max_drawdown,
                    win_rate,
                    total_trades
                FROM performance_periods
                WHERE period_end >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY period_end DESC;
            """)
            
            # Active arbitrage opportunities view
            cursor.execute("""
                CREATE OR REPLACE VIEW active_arbitrage_opportunities AS
                SELECT 
                    id,
                    detected_at,
                    opportunity_type,
                    symbol,
                    buy_exchange,
                    sell_exchange,
                    profit_pct,
                    profit_net,
                    confidence,
                    execution_window_ms,
                    AGE(NOW(), detected_at) as age
                FROM arbitrage_opportunities
                WHERE NOT executed
                    AND detected_at >= NOW() - INTERVAL '5 minutes'
                ORDER BY profit_pct * confidence DESC;
            """)
            
            # Strategy performance leaderboard
            cursor.execute("""
                CREATE OR REPLACE VIEW strategy_leaderboard AS
                SELECT 
                    sp.strategy_name,
                    sp.hit_rate,
                    sp.total_pnl,
                    sp.trade_count,
                    sw.weight as current_weight,
                    sw.updated_at as weight_updated
                FROM strategy_performance sp
                LEFT JOIN strategy_weights sw ON sp.strategy_name = sw.strategy_name
                ORDER BY sp.hit_rate * sp.total_pnl DESC;
            """)
            
            # =====================================================
            # 9. CREATE FUNCTIONS FOR ANALYTICS
            # =====================================================
            print("9. Creating analytical functions...")
            
            # Function to calculate rolling Sharpe ratio
            cursor.execute("""
                CREATE OR REPLACE FUNCTION calculate_rolling_sharpe(
                    p_portfolio_name TEXT,
                    p_window_days INTEGER DEFAULT 30
                ) RETURNS DOUBLE PRECISION AS $$
                DECLARE
                    v_sharpe DOUBLE PRECISION;
                BEGIN
                    SELECT 
                        AVG(daily_return) / NULLIF(STDDEV(daily_return), 0) * SQRT(252)
                    INTO v_sharpe
                    FROM (
                        SELECT 
                            date,
                            (ending_value - starting_value) / starting_value as daily_return
                        FROM performance_periods
                        WHERE portfolio_name = p_portfolio_name
                            AND period_type = 'daily'
                            AND date >= CURRENT_DATE - p_window_days
                    ) daily_returns;
                    
                    RETURN COALESCE(v_sharpe, 0.0);
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to get current portfolio risk score
            cursor.execute("""
                CREATE OR REPLACE FUNCTION get_portfolio_risk_score(
                    p_portfolio_name TEXT
                ) RETURNS JSONB AS $$
                DECLARE
                    v_result JSONB;
                BEGIN
                    SELECT jsonb_build_object(
                        'var_pct', var_pct,
                        'max_drawdown', (
                            SELECT max_drawdown 
                            FROM performance_periods 
                            WHERE portfolio_name = p_portfolio_name 
                                AND period_type = 'monthly'
                            ORDER BY period_end DESC 
                            LIMIT 1
                        ),
                        'risk_alerts_count', (
                            SELECT COUNT(*) 
                            FROM risk_alerts 
                            WHERE portfolio_name = p_portfolio_name 
                                AND NOT acknowledged
                        ),
                        'last_stress_test_passed', (
                            SELECT passed 
                            FROM stress_test_results 
                            WHERE portfolio_name = p_portfolio_name 
                            ORDER BY test_time DESC 
                            LIMIT 1
                        )
                    )
                    INTO v_result
                    FROM latest_risk_metrics
                    WHERE portfolio_name = p_portfolio_name;
                    
                    RETURN COALESCE(v_result, '{}'::jsonb);
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            conn.commit()
            print("\n✓ Advanced schema created successfully!")
            
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Error creating schema: {e}")
        raise
    finally:
        conn.close()


def create_indexes():
    """Creates additional performance indexes"""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            print("\nCreating performance indexes...")
            
            # Composite indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_metrics_symbol_time 
                ON orderbook_metrics(symbol, time DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_closed_trades_portfolio_symbol 
                ON closed_trades(portfolio_name, symbol, closed_at DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_arbitrage_type_profit 
                ON arbitrage_opportunities(opportunity_type, profit_pct DESC) 
                WHERE NOT executed;
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_periods_portfolio_type 
                ON performance_periods(portfolio_name, period_type, period_end DESC);
            """)
            
            conn.commit()
            print("✓ Indexes created successfully!")
            
    except Exception as e:
        conn.rollback()
        print(f"✗ Error creating indexes: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    print("=" * 70)
    print("ADVANCED TRADING SYSTEM - DATABASE SCHEMA EXTENSION")
    print("=" * 70)
    
    try:
        create_advanced_schema()
        create_indexes()
        
        print("\n" + "=" * 70)
        print("✓ DATABASE SCHEMA EXTENSION COMPLETE")
        print("=" * 70)
        print("\nNew tables created:")
        print("  • orderbook_metrics, orderbook_snapshots")
        print("  • rl_decisions, rl_episodes, strategy_weights")
        print("  • var_calculations, stress_test_results, risk_alerts")
        print("  • arbitrage_opportunities, exchange_feed_status")
        print("  • portfolio_snapshots, closed_trades, performance_periods")
        print("  • service_health, system_events")
        print("\nViews created:")
        print("  • latest_risk_metrics")
        print("  • recent_performance")
        print("  • active_arbitrage_opportunities")
        print("  • strategy_leaderboard")
        print("\nFunctions created:")
        print("  • calculate_rolling_sharpe()")
        print("  • get_portfolio_risk_score()")
        
    except Exception as e:
        print(f"\n✗ Failed to extend database schema: {e}")
        exit(1)
