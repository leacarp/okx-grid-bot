"""
Calcula los niveles de precio de la grilla y el tamaño de orden por nivel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Mínimo de BTC aceptado por OKX spot (~$0.85 a $85k)
MIN_ORDER_BASE = 0.00001

# Regímenes de mercado soportados
REGIME_RANGING = "RANGING"
REGIME_TRENDING_UP = "TRENDING_UP"
REGIME_TRENDING_DOWN = "TRENDING_DOWN"
_VALID_REGIMES = {REGIME_RANGING, REGIME_TRENDING_UP, REGIME_TRENDING_DOWN}

# Proporciones de distribución asimétrica
_TRENDING_BUY_RATIO = 0.7   # 70% BUY en TRENDING_UP
_TRENDING_SELL_RATIO = 0.7  # 70% SELL en TRENDING_DOWN


@dataclass
class GridLevel:
    """Representa un nivel de precio de la grilla."""

    index: int
    price: float
    order_size_base: float         # Cantidad en BTC (moneda base)
    order_size_quote: float        # Cantidad en USDT (capital asignado)
    side: str = field(default="")  # "buy", "sell" o "" (auto: OrderManager decide por precio)


class GridCalculator:
    """
    Calcula niveles equidistantes entre price_min y price_max (inclusive).

    Divide el rango en (num_levels - 1) intervalos, produciendo exactamente
    num_levels precios desde price_min hasta price_max.

    Soporta position sizing dinámico según score de mercado y auto-ajuste
    del rango de la grilla según ATR.
    """

    def __init__(
        self,
        price_min: float,
        price_max: float,
        num_levels: int,
        total_capital_usdt: float,
        max_order_usdt: float,
        fee_rate: float = 0.001,
        min_order_usdt: float = 0.0,
        range_width_pct: float | None = None,
    ) -> None:
        """
        Args:
            price_min: Precio mínimo del rango de la grilla.
            price_max: Precio máximo del rango de la grilla.
            num_levels: Número de niveles de la grilla (>= 2).
            total_capital_usdt: Capital total asignado en USDT.
            max_order_usdt: Capital máximo por orden en USDT.
            fee_rate: Tasa de comisión (ej: 0.001 = 0.1%).
            min_order_usdt: Capital mínimo por nivel (piso para position sizing).
                Si 0.0, no se aplica piso.
            range_width_pct: Ancho de rango configurado como porcentaje del precio
                (ej: 5.0 = 5%). Se usa para auto-ajuste por ATR en calculate().
                Si None, el auto-ajuste por ATR no se aplica.
        """
        self._validate_params(price_min, price_max, num_levels, total_capital_usdt, max_order_usdt)

        self.price_min = price_min
        self.price_max = price_max
        self.num_levels = num_levels
        self.total_capital_usdt = total_capital_usdt
        self.max_order_usdt = max_order_usdt
        self.fee_rate = fee_rate
        self.min_order_usdt = max(0.0, min_order_usdt)
        self.range_width_pct = range_width_pct

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

    @staticmethod
    def _score_multiplier(score: float) -> float:
        """
        Retorna el multiplicador de capital según el score de mercado.

        Args:
            score: Score de utilidad del par (0.0 a 1.0).

        Returns:
            1.0 si score >= 0.7, 0.75 si 0.4 <= score < 0.7, 0.5 si score < 0.4.
        """
        if score >= 0.7:
            return 1.0
        if score >= 0.4:
            return 0.75
        return 0.5

    def _apply_atr_range_adjustment(
        self, atr_pct: float
    ) -> tuple[float, float]:
        """
        Expande el rango precio_min/precio_max si el ATR supera el rango configurado.

        El rango efectivo debe ser al menos 2x el ATR actual. Si range_width_pct
        no está configurado, retorna el rango original sin cambios.

        Args:
            atr_pct: ATR expresado como ratio decimal (ej: 0.03 = 3%).

        Returns:
            Tupla (eff_price_min, eff_price_max) con el rango efectivo ajustado.
        """
        if self.range_width_pct is None:
            return self.price_min, self.price_max

        midpoint = (self.price_min + self.price_max) / 2
        if midpoint <= 0:
            return self.price_min, self.price_max

        current_range_pct = (self.price_max - self.price_min) / midpoint * 100
        required_range_pct = max(self.range_width_pct, atr_pct * 200)

        if required_range_pct <= current_range_pct:
            return self.price_min, self.price_max

        half = midpoint * required_range_pct / 200
        eff_price_min = midpoint - half
        eff_price_max = midpoint + half

        logger.info(
            "rango_ajustado_por_atr | atr_pct=%.2f%% rango_original=%.2f%% "
            "rango_efectivo=%.2f%% nuevo_min=%.2f nuevo_max=%.2f",
            atr_pct * 100,
            current_range_pct,
            required_range_pct,
            eff_price_min,
            eff_price_max,
        )
        return eff_price_min, eff_price_max

    def calculate(
        self,
        regime: str = REGIME_RANGING,
        score: float = 1.0,
        atr_pct: float | None = None,
    ) -> list[GridLevel]:
        """
        Calcula los niveles de la grilla con distribución según el régimen de mercado.

        Args:
            regime: Régimen de mercado para la distribución de órdenes.
                - "RANGING": distribución simétrica (comportamiento original).
                - "TRENDING_UP": 70% de niveles como BUY, 30% como SELL.
                - "TRENDING_DOWN": 30% de niveles como BUY, 70% como SELL.
            score: Score de utilidad del par (0.0 a 1.0). Ajusta el capital por
                nivel multiplicando por un factor:
                - score >= 0.7 → 100% del capital_por_nivel
                - score 0.4–0.7 → 75%
                - score < 0.4 → 50%
                Default 1.0 (sin ajuste) para mantener compatibilidad cuando no
                hay MarketAnalyzer. Nunca baja de min_order_usdt si está configurado.
            atr_pct: ATR expresado como ratio decimal (ej: 0.03 = 3%). Si se
                provee junto con range_width_pct (en __init__), el rango se
                expande para garantizar al menos 2x el ATR actual.

        Returns:
            Lista de GridLevel ordenados de menor a mayor precio.
            En regímenes trending, cada nivel tiene `side` pre-asignado ("buy"/"sell").
            En RANGING, `side` queda vacío y el OrderManager decide por precio actual.

        Raises:
            ValueError: Si el régimen es inválido o el capital por nivel es insuficiente.
        """
        if regime not in _VALID_REGIMES:
            raise ValueError(
                f"regime inválido: '{regime}'. Valores permitidos: {sorted(_VALID_REGIMES)}"
            )

        # Auto-ajuste de rango por ATR (expande si es necesario)
        eff_price_min = self.price_min
        eff_price_max = self.price_max
        if atr_pct is not None:
            eff_price_min, eff_price_max = self._apply_atr_range_adjustment(atr_pct)

        step = (eff_price_max - eff_price_min) / (self.num_levels - 1)
        capital_per_level = min(
            self.total_capital_usdt / self.num_levels,
            self.max_order_usdt,
        )

        # Position sizing dinámico según score
        multiplier = self._score_multiplier(score)
        effective_capital = capital_per_level * multiplier
        if self.min_order_usdt > 0:
            effective_capital = max(effective_capital, self.min_order_usdt)

        # Advertir si el spread entre niveles no cubre las fees (ida + vuelta)
        step_pct = step / eff_price_min if eff_price_min > 0 else 0.0
        min_profitable_step_pct = 2 * self.fee_rate
        if step_pct < min_profitable_step_pct:
            logger.warning(
                "spread_insuficiente | step_pct=%.4f min_required=%.4f",
                step_pct,
                min_profitable_step_pct,
            )

        levels: list[GridLevel] = []
        for i in range(self.num_levels):
            price = eff_price_min + i * step
            order_size_base = effective_capital / price

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
                    order_size_quote=round(effective_capital, 4),
                )
            )

        buy_levels, sell_levels = self._apply_regime_distribution(levels, regime)

        logger.info(
            "grid_calculada | levels=%d step=%.2f capital_base=%.4f score=%.2f "
            "multiplier=%.2f capital_efectivo=%.4f USDT",
            self.num_levels,
            step,
            capital_per_level,
            score,
            multiplier,
            effective_capital,
        )
        return levels

    @staticmethod
    def _apply_regime_distribution(
        levels: list[GridLevel],
        regime: str,
    ) -> tuple[int, int]:
        """
        Asigna el campo `side` de cada GridLevel según el régimen.

        Para RANGING no modifica nada (side = "" → OrderManager decide por precio).
        Para tendencias, los niveles de precio más bajo se etiquetan como "buy"
        y los más alto como "sell", en la proporción correspondiente al régimen.

        Returns:
            Tupla (buy_count, sell_count) con la distribución aplicada.
        """
        n = len(levels)
        if regime == REGIME_RANGING:
            return 0, 0

        if regime == REGIME_TRENDING_UP:
            buy_count = max(1, min(n - 1, round(_TRENDING_BUY_RATIO * n)))
            sell_count = n - buy_count
        else:  # REGIME_TRENDING_DOWN — calcular sell_count directamente para consistencia
            sell_count = max(1, min(n - 1, round(_TRENDING_SELL_RATIO * n)))
            buy_count = n - sell_count

        for i, level in enumerate(levels):
            level.side = "buy" if i < buy_count else "sell"

        logger.info(
            "grid_asimetrica | regime=%s buy_levels=%d sell_levels=%d",
            regime,
            buy_count,
            sell_count,
        )
        return buy_count, sell_count
