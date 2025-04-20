import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging():
    """Set up application logging with rotating file handler"""
    # Reset all existing loggers and handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = False

    # Reset root logger as well
    logging.root.handlers = []

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create handlers
    file_handler = RotatingFileHandler(
        f"{log_dir}/legal_decomposition.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()

    # Configure format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Set up root logger with file handler only (to avoid duplication in console)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Configure specialized loggers with both handlers
    logger_configs = {
        "pipeline": logging.DEBUG,
        "api": logging.INFO,
        "components": logging.DEBUG,
        "document_store": logging.INFO,
        "app": logging.INFO,  # Adding app logger for application-level messages
    }

    for logger_name, level in logger_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False  # Critical to prevent duplication
        logger.handlers = []  # Clear any existing handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Return the application logger for direct use
    app_logger = logging.getLogger("app")
    return app_logger