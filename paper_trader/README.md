# Paper Trader

This is a simple paper trading module for the trading system MVP.

## State Management

The portfolio is stored in `portfolio.json`. This is a temporary solution for the MVP and is not suitable for production use. It is not thread-safe and could lead to data corruption if accessed by multiple processes simultaneously.

## Future Improvements

For a more robust system, the portfolio should be stored in a transactional database (like PostgreSQL) or a dedicated in-memory store like Redis.
