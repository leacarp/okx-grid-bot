"""
Validaciones de riesgo para operaciones del bot.

Implementa circuit breaker para detener el bot ante errores consecutivos
o pérdidas que superen los límites configurados.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.core.pnl_tracker import PnLTracker

logger = structlog.get_logger(__name__)


class RiskManager:
    """
    Valida que cada operación cumpla con los límites de riesgo.

    Si se activa el circuit breaker, todas las validaciones retornan False
    hasta que se reinicie el bot (o hasta medianoche UTC si fue por pérdida diaria).
    """

    def __init__(
        self,
        risk_config: dict[str, Any],
        loop_config: dict[str, Any],
        pnl_tracker: "PnLTracker | None" = None,
        max_order_usdt: float | None = None,
    ) -> None:
        self._max_daily_loss = risk_config.get("max_daily_loss_usdt", 2.0)
        self._max_open_orders = risk_config.get("max_open_orders", 10)
        self._max_consecutive_errors = loop_config.get("max_consecutive_errors", 5)
        self._pnl_tracker = pnl_tracker
        self._max_order_usdt: float | None = max_order_usdt or risk_config.get("max_order_usdt")

        self._consecutive_errors: int = 0
        self._circuit_open: bool = False
        self._circuit_reason: str | None = None
        self._last_reset_date: str | None = None

    # ------------------------------------------------------------------
    # Reset diario
    # ------------------------------------------------------------------

    def _check_daily_reset(self) -> None:
        """
        Resetea el circuit breaker de pérdida diaria a las 00:00 UTC.

        Solo resetea si el circuit fue activado por pérdida diaria (no por
        errores consecutivos). Los errores consecutivos requieren reinicio manual.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if self._last_reset_date == today:
            return

        self._last_reset_date = today
        if self._circuit_open and self._circuit_reason == "daily_loss":
            self._circuit_open = False
            self._circuit_reason = None
            logger.info(
                "circuit_breaker_reseteado_por_dia_nuevo",
                fecha=today,
            )

    # ------------------------------------------------------------------
    # Validación unificada (nueva)
    # ------------------------------------------------------------------

    def pre_order_check(
        self,
        order_size_usdt: float,
        available_balance: float | None,
        open_order_count: int,
    ) -> tuple[bool, str]:
        """
        Ejecuta todas las validaciones en secuencia antes de colocar una orden.

        Secuencia:
          1. Circuit breaker (bloqueo duro).
          2. Pérdida diaria via PnLTracker (si está configurado).
          3. Balance disponible vs tamaño de orden (si available_balance no es None).
          4. Máximo de órdenes abiertas simultáneas.
          5. Tamaño de orden vs máximo configurado (si max_order_usdt está configurado).

        Args:
            order_size_usdt: Tamaño de la orden en USDT.
            available_balance: Balance disponible en USDT. None para omitir el check.
            open_order_count: Número de órdenes abiertas actualmente.

        Returns:
            (True, "ok") si todas las validaciones pasan.
            (False, reason) con la primera validación que falla.
        """
        self._check_daily_reset()

        if self._circuit_open:
            return False, "circuit_breaker_activado"

        # 1. Pérdida diaria via PnLTracker
        if self._pnl_tracker is not None:
            daily_pnl = self._pnl_tracker.get_daily_pnl()
            if daily_pnl < 0:
                ok, reason = self.check_daily_loss(abs(daily_pnl))
                if not ok:
                    self._circuit_reason = "daily_loss"
                    return False, reason

        # 2. Balance disponible
        if available_balance is not None:
            ok, reason = self.check_balance(available_balance, order_size_usdt)
            if not ok:
                return False, reason

        # 3. Máximo de órdenes abiertas
        ok, reason = self.check_open_orders(open_order_count)
        if not ok:
            return False, reason

        # 4. Tamaño de orden vs máximo
        if self._max_order_usdt is not None:
            ok, reason = self.can_place_order(order_size_usdt, self._max_order_usdt)
            if not ok:
                return False, reason

        return True, "ok"

    # ------------------------------------------------------------------
    # Validaciones individuales
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
            logger.warning("orden_excede_maximo", size=order_size_usdt, max=max_order_usdt)
            return False, reason
        return True, "ok"

    def check_daily_loss(self, current_loss: float) -> tuple[bool, str]:
        """
        Verifica que la pérdida diaria acumulada no exceda el límite.

        Si se supera, activa el circuit breaker. Se resetea automáticamente
        a las 00:00 UTC vía _check_daily_reset().
        """
        if current_loss >= self._max_daily_loss:
            reason = (
                f"perdida_diaria_excedida | loss={current_loss:.4f} "
                f"max={self._max_daily_loss:.4f}"
            )
            logger.critical(
                "perdida_diaria_excedida",
                loss=current_loss,
                max=self._max_daily_loss,
            )
            self._circuit_open = True
            self._circuit_reason = "daily_loss"
            return False, reason
        return True, "ok"

    def check_open_orders(self, count: int) -> tuple[bool, str]:
        """Verifica que no se supere el máximo de órdenes abiertas simultáneas."""
        if count >= self._max_open_orders:
            reason = (
                f"max_ordenes_abiertas_alcanzado | count={count} "
                f"max={self._max_open_orders}"
            )
            logger.warning(
                "max_ordenes_abiertas_alcanzado",
                count=count,
                max=self._max_open_orders,
            )
            return False, reason
        return True, "ok"

    def check_balance(self, available: float, required: float) -> tuple[bool, str]:
        """Verifica que el balance disponible sea suficiente para la operación."""
        if available < required:
            reason = (
                f"balance_insuficiente | available={available:.4f} "
                f"required={required:.4f}"
            )
            logger.warning(
                "balance_insuficiente",
                available=available,
                required=required,
            )
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
            "error_registrado",
            consecutivos=self._consecutive_errors,
            max=self._max_consecutive_errors,
        )

        if self._consecutive_errors >= self._max_consecutive_errors:
            self._circuit_open = True
            self._circuit_reason = "consecutive_errors"
            logger.critical(
                "circuit_breaker_activado",
                errores_consecutivos=self._consecutive_errors,
            )
            return True
        return False

    def reset_errors(self) -> None:
        """Resetea el contador de errores consecutivos tras un ciclo exitoso."""
        if self._consecutive_errors > 0:
            logger.debug("errores_reseteados", previos=self._consecutive_errors)
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
