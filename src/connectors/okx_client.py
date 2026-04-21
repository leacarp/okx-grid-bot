"""
Wrapper sobre ccxt.okx con retry automático, manejo de errores y modo DRY_RUN.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import ccxt
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    ccxt.RateLimitExceeded,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
)


def _build_retry_decorator():
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class OKXClient:
    """
    Wrapper fino sobre ccxt.okx.

    En modo DRY_RUN los métodos de escritura (create_limit_order, cancel_order)
    registran la acción por log sin enviarla al exchange.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        dry_run: bool = True,
        sandbox: bool = False,
        exchange: ccxt.okx | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OKX_API_KEY no configurada")
        if not api_secret:
            raise ValueError("OKX_API_SECRET no configurada")
        if not passphrase:
            raise ValueError("OKX_API_PASSPHRASE no configurada")

        self.dry_run = dry_run
        self._sandbox = sandbox

        if exchange is not None:
            self._exchange = exchange
        else:
            self._exchange = ccxt.okx(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "password": passphrase,
                    "enableRateLimit": True,
                }
            )
            if sandbox:
                self._exchange.set_sandbox_mode(True)

    # ------------------------------------------------------------------
    # Métodos de lectura
    # ------------------------------------------------------------------

    @_build_retry_decorator()
    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Retorna ticker con bid, ask y last para el símbolo dado."""
        return self._exchange.fetch_ticker(symbol)

    @_build_retry_decorator()
    def fetch_balance(self) -> dict[str, Any]:
        """Retorna balances de la cuenta."""
        return self._exchange.fetch_balance()

    @_build_retry_decorator()
    def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Retorna lista de órdenes abiertas para el símbolo."""
        return self._exchange.fetch_open_orders(symbol)

    @_build_retry_decorator()
    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Retorna el estado de una orden específica."""
        return self._exchange.fetch_order(order_id, symbol)

    # ------------------------------------------------------------------
    # Métodos de escritura (bloqueados en DRY_RUN)
    # ------------------------------------------------------------------

    @_build_retry_decorator()
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict[str, Any] | None:
        """
        Coloca una orden límite.

        En DRY_RUN retorna un dict simulado sin llamar al exchange.
        """
        if self.dry_run:
            simulated = {
                "id": f"dry_run_{side}_{price}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "open",
                "dry_run": True,
            }
            logger.info(
                "orden_simulada | symbol=%s side=%s price=%.2f amount=%.8f",
                symbol,
                side,
                price,
                amount,
            )
            return simulated

        if side not in ("buy", "sell"):
            raise ValueError(f"side inválido: {side!r}. Debe ser 'buy' o 'sell'.")

        try:
            return self._exchange.create_order(symbol, "limit", side, amount, price)
        except ccxt.InsufficientFunds as exc:
            logger.warning(
                "fondos_insuficientes | symbol=%s side=%s price=%.2f amount=%.8f error=%s",
                symbol,
                side,
                price,
                amount,
                str(exc)[:120],
            )
            return None

    @_build_retry_decorator()
    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """
        Cancela una orden por ID.

        En DRY_RUN solo loggea sin llamar al exchange.
        """
        if self.dry_run:
            logger.info("cancelacion_simulada | order_id=%s symbol=%s", order_id, symbol)
            return {"id": order_id, "status": "canceled", "dry_run": True}

        return self._exchange.cancel_order(order_id, symbol)

    # ------------------------------------------------------------------
    # Conexión / validación
    # ------------------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """
        Valida credenciales haciendo fetch_balance().

        Lanza AuthenticationError si las credenciales son incorrectas.
        """
        try:
            balance = self.fetch_balance()
            logger.info(
                "conexion_exitosa | sandbox=%s dry_run=%s",
                self._sandbox,
                self.dry_run,
            )
            return balance
        except ccxt.AuthenticationError:
            logger.critical("error_autenticacion | Credenciales OKX inválidas. Bot detenido.")
            raise

    # ------------------------------------------------------------------
    # Factory desde variables de entorno
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, sandbox: bool = False) -> "OKXClient":
        """Crea un OKXClient leyendo credenciales y DRY_RUN desde el entorno."""
        api_key = os.getenv("OKX_API_KEY", "")
        api_secret = os.getenv("OKX_API_SECRET", "")
        passphrase = os.getenv("OKX_API_PASSPHRASE", "")
        dry_run_raw = os.getenv("DRY_RUN", "true").lower()
        dry_run = dry_run_raw != "false"

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            dry_run=dry_run,
            sandbox=sandbox,
        )
