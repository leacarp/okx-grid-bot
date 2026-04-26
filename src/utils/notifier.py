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

    def notify_pair_changed(
        self,
        new_symbol: str,
        score: float,
        regime: str,
        atr_pct: float,
    ) -> None:
        """
        Notifica el cambio de par seleccionado por PairSelector.

        Args:
            new_symbol: Nuevo par seleccionado (ej. "ETH/USDT").
            score: Score de utilidad del nuevo par (0.0 a 1.0).
            regime: Régimen de mercado (RANGING, TRENDING_UP, TRENDING_DOWN).
            atr_pct: ATR expresado como porcentaje del precio (ej. 0.032 = 3.2%).
        """
        self.send(
            f"\U0001f504 Cambiando a {new_symbol}\n"
            f"Score: {score:.2f} | R\u00e9gimen: {regime} | ATR: {atr_pct * 100:.1f}%"
        )

    def notify_no_suitable_pair(self, best_symbol: str, best_score: float, min_score: float) -> None:
        """Notifica que ningún par supera el score mínimo para operar."""
        self.send(
            f"\u26a0\ufe0f Ningún par apto para operar\n"
            f"Mejor par: {best_symbol} | Score: {best_score:.2f} | M\u00ednimo: {min_score:.2f}"
        )

    def notify_weekly_summary(
        self,
        trades: int,
        ganancia_neta: float,
        mejor_par: str,
        mejor_par_profit: float,
        peor_par: str,
        peor_par_profit: float,
    ) -> None:
        """
        Envía el resumen semanal de PnL a Telegram (domingos a las 00:00 UTC).

        Args:
            trades: Número total de trades completados en la semana.
            ganancia_neta: Ganancia neta total en USDT.
            mejor_par: Símbolo con mayor ganancia (ej: "BTC/USDT").
            mejor_par_profit: Ganancia neta del mejor par en USDT.
            peor_par: Símbolo con menor ganancia o mayor pérdida (ej: "SOL/USDT").
            peor_par_profit: Ganancia/pérdida neta del peor par en USDT.
        """
        net_sign = "+" if ganancia_neta >= 0 else "-"
        mejor_sign = "+" if mejor_par_profit >= 0 else "-"
        peor_sign = "+" if peor_par_profit >= 0 else "-"

        self.send(
            f"\U0001f4c8 Resumen semanal\n"
            f"Trades: {trades} | Ganancia neta: {net_sign}${abs(ganancia_neta):.2f} USDT\n"
            f"Mejor par: {mejor_par} ({mejor_sign}${abs(mejor_par_profit):.2f})\n"
            f"Peor par: {peor_par} ({peor_sign}${abs(peor_par_profit):.2f})"
        )

    def notify_daily_summary(
        self,
        winning_trades: int,
        losing_trades: int,
        gross_profit: float,
        total_fees: float,
        net_profit: float,
    ) -> None:
        """
        Envía el resumen diario de PnL a Telegram (una vez al día a las 00:00 UTC).

        Usa ⚠️ como encabezado si la ganancia neta es negativa.
        """
        header = "\u26a0\ufe0f" if net_profit < 0 else "\U0001f4ca"
        profit_sign = "+" if net_profit >= 0 else ""
        gross_sign = "+" if gross_profit >= 0 else ""

        self.send(
            f"{header} Resumen del día\n"
            f"\u2705 Trades ganadores: {winning_trades}\n"
            f"\u274c Trades perdedores: {losing_trades}\n"
            f"\U0001f4b0 Profit bruto: {gross_sign}${gross_profit:.2f} USDT\n"
            f"\U0001f4b8 Fees pagadas: -${total_fees:.2f} USDT\n"
            f"\U0001f4c8 Ganancia neta: {profit_sign}${net_profit:.2f} USDT"
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
