"""
Configuración centralizada de logging usando structlog con output JSON.

Escribe a consola (stdout) y a archivo rotativo.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/bot.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """
    Inicializa structlog con:
    - Output JSON a archivo rotativo.
    - Output legible (ConsoleRenderer) en consola.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Evitar duplicar handlers si se llama más de una vez
    if root_logger.handlers:
        root_logger.handlers.clear()

    # --- Handler de archivo con JSON ---
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    # --- Handler de consola legible ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # --- Procesadores compartidos ---
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter JSON para archivo
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # Formatter legible para consola
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=shared_processors,
    )

    file_handler.setFormatter(json_formatter)
    console_handler.setFormatter(console_formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retorna un logger structlog para el módulo dado."""
    return structlog.get_logger(name)
