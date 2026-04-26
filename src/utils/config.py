"""
Carga y validación de config.yaml con merge de variables de entorno.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """
    Carga config.yaml y aplica overrides desde variables de entorno.

    Raises:
        FileNotFoundError: Si config.yaml no existe.
        ValueError: Si los valores no pasan validación.
    """
    load_dotenv()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml no encontrado en: {config_path.resolve()}")

    with config_path.open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    config = _apply_env_overrides(config)
    _validate(config)

    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Sobreescribe valores de config con variables de entorno cuando están presentes."""
    log_level_env = os.getenv("LOG_LEVEL")
    if log_level_env:
        config.setdefault("logging", {})["level"] = log_level_env

    # DRY_RUN se lee en OKXClient directamente desde el entorno,
    # pero lo exponemos también en config para trazabilidad
    dry_run_raw = os.getenv("DRY_RUN", "true").lower()
    config["dry_run"] = dry_run_raw != "false"

    return config


def _validate(config: dict[str, Any]) -> None:
    """
    Valida tipos y rangos críticos del config.

    Raises:
        ValueError: Si alguna validación falla.
    """
    _validate_tokens(config)
    _validate_grid(config)
    _validate_risk(config)
    _validate_loop(config)
    _validate_trailing(config)


def _validate_tokens(config: dict[str, Any]) -> None:
    """Valida la lista de tokens candidatos."""
    tokens = config.get("tokens")
    if not tokens:
        raise ValueError("'tokens' debe ser una lista no vacía de símbolos de trading")
    if not isinstance(tokens, list):
        raise ValueError("'tokens' debe ser una lista")
    if len(tokens) == 0:
        raise ValueError("'tokens' debe contener al menos un símbolo")

    for i, token in enumerate(tokens):
        symbol = token.get("symbol")
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"tokens[{i}].symbol debe ser un string no vacío")

        num_levels = token.get("num_levels", 0)
        if num_levels < 2:
            raise ValueError(
                f"tokens[{i}].num_levels debe ser >= 2, recibido: {num_levels} (symbol={symbol})"
            )

        min_step = token.get("min_step_usdt", 0)
        if min_step <= 0:
            raise ValueError(
                f"tokens[{i}].min_step_usdt debe ser > 0, recibido: {min_step} (symbol={symbol})"
            )


def _validate_grid(config: dict[str, Any]) -> None:
    """Valida la sección grid (parámetros globales de capital y rango)."""
    grid = config.get("grid", {})

    total_capital: float = grid.get("total_capital_usdt", 0)
    max_order: float = grid.get("max_order_usdt", 0)
    range_width_pct: float = grid.get("range_width_pct", 0)

    if total_capital <= 0:
        raise ValueError(f"grid.total_capital_usdt debe ser > 0, recibido: {total_capital}")
    if max_order <= 0:
        raise ValueError(f"grid.max_order_usdt debe ser > 0, recibido: {max_order}")
    if range_width_pct <= 0:
        raise ValueError(f"grid.range_width_pct debe ser > 0, recibido: {range_width_pct}")


def _validate_risk(config: dict[str, Any]) -> None:
    """Valida la sección risk."""
    risk = config.get("risk", {})
    if risk.get("max_daily_loss_usdt", 0) <= 0:
        raise ValueError("risk.max_daily_loss_usdt debe ser > 0")


def _validate_loop(config: dict[str, Any]) -> None:
    """Valida la sección loop."""
    loop = config.get("loop", {})
    if loop.get("interval_seconds", 0) < 5:
        raise ValueError("loop.interval_seconds debe ser >= 5")
    if loop.get("max_consecutive_errors", 0) < 1:
        raise ValueError("loop.max_consecutive_errors debe ser >= 1")


def _validate_trailing(config: dict[str, Any]) -> None:
    """Valida la sección trailing (si está presente)."""
    trailing = config.get("trailing", {})
    if "recenter_every_cycles" in trailing and trailing["recenter_every_cycles"] < 1:
        raise ValueError("trailing.recenter_every_cycles debe ser >= 1")
