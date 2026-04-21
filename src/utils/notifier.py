"""
Notificador de Telegram para eventos clave del bot.

Falla silenciosamente: nunca interrumpe el loop principal.
Usa requests directamente, sin dependencia extra de librería de Telegram.
"""
from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger("notifier")

_TELEGRAM_API_TIMEOUT_SECONDS = 5
_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Envía mensajes a un chat de Telegram via Bot API (HTTP POST).

    Si TOKEN o CHAT_ID no están configurados, los métodos retornan
    inmediatamente sin hacer ninguna llamada de red.
    Si la llamada falla (timeout, red caída, token inválido),
    loggea el error y continúa — nunca propaga la excepción.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        # Soportar tanto TELEGRAM_TOKEN como TELEGRAM_BOT_TOKEN (convención alternativa)
        self._token: str | None = (
            token
            or os.getenv("TELEGRAM_TOKEN")
            or os.getenv("TELEGRAM_BOT_TOKEN")
        )
        self._chat_id: str | None = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._enabled: bool = bool(self._token and self._chat_id)

        if self._enabled:
            logger.info(
                "notifier_habilitado",
                chat_id_preview=f"{self._chat_id[:4]}****" if self._chat_id else None,
            )
        else:
            logger.warning(
                "notifier_deshabilitado",
                token_configurado=bool(self._token),
                chat_id_configurado=bool(self._chat_id),
                hint="Configurar TELEGRAM_TOKEN (o TELEGRAM_BOT_TOKEN) y TELEGRAM_CHAT_ID",
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def send(self, message: str) -> bool:
        """
        Envía un mensaje de texto plano a Telegram.

        Args:
            message: Texto a enviar (máximo 4096 caracteres por límite de API).

        Returns:
            True si el mensaje fue enviado correctamente, False en cualquier fallo.
        """
        if not self._enabled:
            return False
        return self._post(message)

    def notify_bot_started(self, symbol: str, dry_run: bool) -> None:
        """Notifica que el bot arrancó correctamente."""
        mode = "DRY RUN" if dry_run else "REAL"
        self.send(f"Bot iniciado\nSímbolo: {symbol}\nModo: {mode}")

    def notify_orders_filled(
        self,
        symbol: str,
        filled_count: int,
        current_price: float,
        cycle: int,
        dry_run: bool,
    ) -> None:
        """Notifica cuando se ejecutaron órdenes en un ciclo."""
        mode = " [DRY RUN]" if dry_run else ""
        self.send(
            f"Órdenes ejecutadas{mode}\n"
            f"Símbolo: {symbol}\n"
            f"Precio actual: ${current_price:,.2f}\n"
            f"Filled: {filled_count}\n"
            f"Ciclo: {cycle}"
        )

    def notify_critical_error(self, error: str, cycle: int) -> None:
        """Notifica un error crítico en el ciclo."""
        self.send(
            f"Error crítico en ciclo {cycle}\n"
            f"Error: {error}"
        )

    def notify_circuit_breaker(self, consecutive_errors: int) -> None:
        """Notifica que el circuit breaker se activó y el bot se detuvo."""
        self.send(
            f"CIRCUIT BREAKER ACTIVADO\n"
            f"El bot se detuvo tras {consecutive_errors} errores consecutivos."
        )

    def notify_grid_recentered(
        self,
        reason: str,
        current_price: float,
        new_min: float,
        new_max: float,
        step: float,
    ) -> None:
        """Notifica que la grilla fue recentrada (trailing grid)."""
        reason_label = "precio fuera de rango" if reason == "out_of_range" else "ciclos"
        self.send(
            f"\U0001f504 Grilla recentrada\n"
            f"Motivo: {reason_label}\n"
            f"Precio actual: ${current_price:,.2f}\n"
            f"Nuevo rango: ${new_min:,.2f} - ${new_max:,.2f}\n"
            f"Step: ${step:,.2f}"
        )

    def notify_no_usdt_available(
        self,
        usdt_available: float,
        required: float,
    ) -> None:
        """Notifica que no hay USDT suficiente para recentrar la grilla."""
        self.send(
            f"\u26a0\ufe0f Sin USDT disponible - esperando ejecución de SELLs\n"
            f"USDT disponible: ${usdt_available:,.4f}\n"
            f"Mínimo requerido: ${required:,.4f}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post(self, text: str) -> bool:
        """
        Realiza el POST a la Telegram Bot API.

        Falla silenciosamente ante cualquier excepción de red o HTTP.
        """
        url = _TELEGRAM_API_BASE.format(token=self._token)
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text[:4096],
        }
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=_TELEGRAM_API_TIMEOUT_SECONDS,
            )
            if not response.ok:
                logger.warning(
                    "telegram_respuesta_no_ok",
                    status_code=response.status_code,
                    body=response.text[:200],
                )
                return False
            return True

        except requests.exceptions.Timeout:
            logger.warning("telegram_timeout", timeout_s=_TELEGRAM_API_TIMEOUT_SECONDS)
            return False
        except requests.exceptions.ConnectionError as exc:
            logger.warning("telegram_connection_error", error=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_error_inesperado", error=str(exc))
            return False
