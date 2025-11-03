"""Provides a centralized, structured logging configuration.

This module uses the `structlog` library to create a standardized JSON-based
logger. Structured logging is essential for modern, observable systems, as it
allows for easy parsing, filtering, and querying of log data in systems like
Loki, Elasticsearch, or Splunk.

The primary function, `get_logger`, configures and returns a logger instance
that automatically includes the service name, log level, and an ISO 8601
timestamp in every log entry, rendered as a JSON object.
"""

import structlog
from typing import Any

def get_logger(service_name: str, **initial_context: Any) -> Any:
    """Configures and returns a structured logger for a given service.

    This function sets up a global `structlog` configuration on its first run
    and then returns a logger instance with bound context, such as the service
    name. The output is a JSON string, making it ideal for containerized
    environments and log aggregation systems.

    Args:
        service_name: The name of the service or component that will be
            using the logger. This will be included in all log entries.
        **initial_context: Any additional key-word arguments to bind to the
            logger as initial context for all log entries from this logger.

    Returns:
        A configured `structlog` logger instance.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(min_level=structlog.INFO),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(service=service_name, **initial_context)

if __name__ == '__main__':
    # Example Usage
    logger = get_logger("example_service")
    logger.info("This is an informational message.", data={"key": "value"})
    logger.error("This is an error message.", error_code=500)
