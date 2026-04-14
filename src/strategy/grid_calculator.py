"""
Calcula los niveles de precio de la grilla y el tamaño de orden por nivel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Mínimo de BTC aceptado por OKX spot (~$0.85 a $85k)
MIN_ORDER_BASE = 0.00001


@dataclass
class GridLevel:
    """Representa un nivel de precio de la grilla."""

    index: int
    price: float
    order_size_base: float   # Cantidad en BTC (moneda base)
    order_size_quote: float  # Cantidad en USDT (capital asignado)


class GridCalculator:
    """
    Calcula niveles equidistantes entre price_min y price_max (inclusive).

    Divide el rango en (num_levels - 1) intervalos, produciendo exactamente
    num_levels precios desde price_min hasta price_max.
    """

    def __init__(
        self,
        price_min: float,
        price_max: float,
        num_levels: int,
        total_capital_usdt: float,
        max_order_usdt: float,
        fee_rate: float = 0.001,
    ) -> None:
        self._validate_params(price_min, price_max, num_levels, total_capital_usdt, max_order_usdt)

        self.price_min = price_min
        self.price_max = price_max
        self.num_levels = num_levels
        self.total_capital_usdt = total_capital_usdt
        self.max_order_usdt = max_order_usdt
        self.fee_rate = fee_rate

    @staticmethod
    def _validate_params(
        price_min: float,
        price_max: float,
        num_levels: int,
        total_capital_usdt: float,
        max_order_usdt: float,
    ) -> None:
        if price_min <= 0:
            raise ValueError(f"price_min debe ser > 0, recibido: {price_min}")
        if price_max <= price_min:
            raise ValueError(f"price_max ({price_max}) debe ser > price_min ({price_min})")
        if num_levels < 2:
            raise ValueError(f"num_levels debe ser >= 2, recibido: {num_levels}")
        if total_capital_usdt <= 0:
            raise ValueError(f"total_capital_usdt debe ser > 0, recibido: {total_capital_usdt}")
        if max_order_usdt <= 0:
            raise ValueError(f"max_order_usdt debe ser > 0, recibido: {max_order_usdt}")

    def calculate(self) -> list[GridLevel]:
        """
        Calcula los niveles de la grilla.

        Returns:
            Lista de GridLevel ordenados de menor a mayor precio.

        Raises:
            ValueError: Si el capital por nivel es insuficiente para el mínimo del exchange.
        """
        step = (self.price_max - self.price_min) / (self.num_levels - 1)
        capital_per_level = min(
            self.total_capital_usdt / self.num_levels,
            self.max_order_usdt,
        )

        # Advertir si el spread entre niveles no cubre las fees (ida + vuelta)
        step_pct = step / self.price_min
        min_profitable_step_pct = 2 * self.fee_rate
        if step_pct < min_profitable_step_pct:
            logger.warning(
                "spread_insuficiente | step_pct=%.4f min_required=%.4f",
                step_pct,
                min_profitable_step_pct,
            )

        levels: list[GridLevel] = []
        for i in range(self.num_levels):
            price = self.price_min + i * step
            order_size_base = capital_per_level / price

            if order_size_base < MIN_ORDER_BASE:
                raise ValueError(
                    f"Nivel {i} (precio={price:.2f}): order_size_base={order_size_base:.8f} "
                    f"< mínimo del exchange ({MIN_ORDER_BASE}). "
                    "Aumentar capital o reducir num_levels."
                )

            levels.append(
                GridLevel(
                    index=i,
                    price=round(price, 2),
                    order_size_base=round(order_size_base, 8),
                    order_size_quote=round(capital_per_level, 4),
                )
            )

        logger.info(
            "grid_calculada | levels=%d step=%.2f capital_por_nivel=%.4f USDT",
            self.num_levels,
            step,
            capital_per_level,
        )
        return levels
