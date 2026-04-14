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
    grid = config.get("grid", {})

    price_min: float = grid.get("price_min", 0)
    price_max: float = grid.get("price_max", 0)
    num_levels: int = grid.get("num_levels", 0)
    total_capital: float = grid.get("total_capital_usdt", 0)
    max_order: float = grid.get("max_order_usdt", 0)

    if price_min <= 0:
        raise ValueError(f"grid.price_min debe ser > 0, recibido: {price_min}")
    if price_max <= price_min:
        raise ValueError(
            f"grid.price_max ({price_max}) debe ser mayor que grid.price_min ({price_min})"
        )
    if num_levels < 2:
        raise ValueError(f"grid.num_levels debe ser >= 2, recibido: {num_levels}")
    if total_capital <= 0:
        raise ValueError(f"grid.total_capital_usdt debe ser > 0, recibido: {total_capital}")
    if max_order <= 0:
        raise ValueError(f"grid.max_order_usdt debe ser > 0, recibido: {max_order}")

    risk = config.get("risk", {})
    if risk.get("max_daily_loss_usdt", 0) <= 0:
        raise ValueError("risk.max_daily_loss_usdt debe ser > 0")

    loop = config.get("loop", {})
    if loop.get("interval_seconds", 0) < 5:
        raise ValueError("loop.interval_seconds debe ser >= 5")
    if loop.get("max_consecutive_errors", 0) < 1:
        raise ValueError("loop.max_consecutive_errors debe ser >= 1")

    trailing = config.get("trailing", {})
    if "recenter_every_cycles" in trailing and trailing["recenter_every_cycles"] < 1:
        raise ValueError("trailing.recenter_every_cycles debe ser >= 1")
    if "min_step_usdt" in trailing and trailing["min_step_usdt"] <= 0:
        raise ValueError("trailing.min_step_usdt debe ser > 0")
