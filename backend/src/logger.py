"""
Minimal logger setup for Python applications.

Provides console logging with timestamp, level, and logger name.
Call `get_logger(__name__)` from any module to start logging.

note: generated using DeepSeek
"""

import logging
import sys

# Default log level (can be changed via set_log_level)
DEFAULT_LEVEL = logging.INFO

# Simple format: time - level - name - message
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Flag to ensure we only configure the root logger once
_configured = False


def _configure_logging(level=DEFAULT_LEVEL):
    """
    Set up the root logger with a console handler and a clean format.
    Idempotent: subsequent calls do nothing.
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.
    Ensures logging is configured before returning.
    """
    _configure_logging()  # no-op if already configured
    return logging.getLogger(name)
