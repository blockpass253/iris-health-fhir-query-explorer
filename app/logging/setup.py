"""Structured logging configuration via structlog."""

import logging

import structlog

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging once for the process."""
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""
    configure_logging()
    return structlog.get_logger(name)
