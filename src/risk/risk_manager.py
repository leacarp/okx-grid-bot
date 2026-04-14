"""
Validaciones de riesgo para operaciones del bot.

Implementa circuit breaker para detener el bot ante errores consecutivos
o pérdidas que superen los límites configurados.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Valida que cada operación cumpla con los límites de riesgo.

    Si se activa el circuit breaker, todas las validaciones retornan False
    hasta que se reinicie el bot.
    """

    def __init__(
        self,
        risk_config: dict[str, Any],
        loop_config: dict[str, Any],
    ) -> None:
        self._max_daily_loss = risk_config.get("max_daily_loss_usdt", 2.0)
        self._max_open_orders = risk_config.get("max_open_orders", 10)
        self._max_consecutive_errors = loop_config.get("max_consecutive_errors", 5)

        self._consecutive_errors: int = 0
        self._circuit_open: bool = False

    # ------------------------------------------------------------------
    # Validaciones de operación
    # ------------------------------------------------------------------

    def can_place_order(
        self,
        order_size_usdt: float,
        max_order_usdt: float,
    ) -> tuple[bool, str]:
        """Verifica que el tamaño de la orden no exceda el máximo configurado."""
        if self._circuit_open:
            return False, "circuit_breaker_activado"
        if order_size_usdt > max_order_usdt:
            reason = (
                f"orden_excede_maximo | size={order_size_usdt:.4f} max={max_order_usdt:.4f}"
            )
            logger.warning(reason)
            return False, reason
        return True, "ok"

    def check_daily_loss(self, current_loss: float) -> tuple[bool, str]:
        """
        Verifica que la pérdida diaria acumulada no exceda el límite.

        Si se supera, activa el circuit breaker permanentemente.
        """
        if current_loss >= self._max_daily_loss:
            reason = (
                f"perdida_diaria_excedida | loss={current_loss:.4f} "
                f"max={self._max_daily_loss:.4f}"
            )
            logger.critical(reason)
            self._circuit_open = True
            return False, reason
        return True, "ok"

    def check_open_orders(self, count: int) -> tuple[bool, str]:
        """Verifica que no se supere el máximo de órdenes abiertas simultáneas."""
        if count >= self._max_open_orders:
            reason = (
                f"max_ordenes_abiertas_alcanzado | count={count} "
                f"max={self._max_open_orders}"
            )
            logger.warning(reason)
            return False, reason
        return True, "ok"

    def check_balance(self, available: float, required: float) -> tuple[bool, str]:
        """Verifica que el balance disponible sea suficiente para la operación."""
        if available < required:
            reason = (
                f"balance_insuficiente | available={available:.4f} "
                f"required={required:.4f}"
            )
            logger.warning(reason)
            return False, reason
        return True, "ok"

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def register_error(self) -> bool:
        """
        Registra un error consecutivo e incrementa el contador.

        Returns:
            True si se activó el circuit breaker en esta llamada, False si no.
        """
        self._consecutive_errors += 1
        logger.warning(
            "error_registrado | consecutivos=%d max=%d",
            self._consecutive_errors,
            self._max_consecutive_errors,
        )

        if self._consecutive_errors >= self._max_consecutive_errors:
            self._circuit_open = True
            logger.critical(
                "circuit_breaker_activado | errores_consecutivos=%d",
                self._consecutive_errors,
            )
            return True
        return False

    def reset_errors(self) -> None:
        """Resetea el contador de errores consecutivos tras un ciclo exitoso."""
        if self._consecutive_errors > 0:
            logger.debug("errores_reseteados | previos=%d", self._consecutive_errors)
            self._consecutive_errors = 0

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def circuit_open(self) -> bool:
        """True si el circuit breaker está activado (bot debe detenerse)."""
        return self._circuit_open

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors
