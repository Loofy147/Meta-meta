import structlog

def get_logger(service_name):
    """
    Configures and returns a structlog logger for a given service.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    return structlog.get_logger(service=service_name)

if __name__ == '__main__':
    # Example Usage
    logger = get_logger("example_service")
    logger.info("This is an informational message.", data={"key": "value"})
    logger.error("This is an error message.", error_code=500)
