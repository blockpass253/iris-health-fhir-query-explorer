"""Structured logging configuration via structlog.

Events fan out through stdlib logging to both the terminal and a log file
(``<IRIS_DEBUG_DIR>/app.log``, default ``debug/app.log``). The file matters
because the Textual TUI takes over the terminal, hiding stream output; the file
stays inspectable (e.g. ``tail -f debug/app.log``) during a TUI session.
"""

import logging

import structlog

from app.debug.dump import debug_dir, dump_enabled

LOG_FILE = "app.log"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging once for the process."""
    global _configured
    if _configured:
        return

    shared_processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    def _formatter(*, colors: bool) -> structlog.stdlib.ProcessorFormatter:
        return structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=colors),
            ],
        )

    root = logging.getLogger()
    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_formatter(colors=True))
    handlers: list[logging.Handler] = [stream_handler]

    if dump_enabled():
        try:
            log_path = debug_dir() / LOG_FILE
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(_formatter(colors=False))
            handlers.append(file_handler)
        except Exception:  # a log file must never block startup
            stream_handler.handle(
                logging.LogRecord(
                    "logging.setup",
                    logging.WARNING,
                    __file__,
                    0,
                    "could not open log file; logging to stream only",
                    None,
                    None,
                )
            )

    root.handlers = handlers
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""
    configure_logging()
    return structlog.get_logger(name)
