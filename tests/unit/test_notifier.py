"""
Tests unitarios para TelegramNotifier.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.utils.notifier import TelegramNotifier

# Variables de entorno que el notifier puede leer
_TELEGRAM_ENV_VARS = ["TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]


@pytest.fixture(autouse=True)
def _clear_telegram_env(monkeypatch):
    """Elimina las variables de entorno de Telegram para aislar cada test."""
    for var in _TELEGRAM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def make_notifier_enabled() -> TelegramNotifier:
    """Crea un TelegramNotifier con token y chat_id explícitos (sin env vars)."""
    return TelegramNotifier(token="test_token_123", chat_id="test_chat_456")


class TestTelegramNotifierConfig:
    def test_deshabilitado_sin_token(self):
        n = TelegramNotifier(token=None, chat_id="123")
        assert n._enabled is False

    def test_deshabilitado_sin_chat_id(self):
        n = TelegramNotifier(token="tok", chat_id=None)
        assert n._enabled is False

    def test_habilitado_con_ambos(self):
        n = make_notifier_enabled()
        assert n._enabled is True

    def test_send_retorna_false_si_deshabilitado(self):
        n = TelegramNotifier(token=None, chat_id=None)
        assert n.send("hola") is False


class TestTelegramNotifierSend:
    def test_send_llama_post(self):
        n = make_notifier_enabled()
        with patch.object(n, "_post", return_value=True) as mock_post:
            result = n.send("mensaje test")
        mock_post.assert_called_once_with("mensaje test")
        assert result is True

    def test_send_retorna_false_si_post_falla(self):
        n = make_notifier_enabled()
        with patch.object(n, "_post", return_value=False):
            result = n.send("mensaje")
        assert result is False

    def test_post_timeout_retorna_false(self):
        import requests
        n = make_notifier_enabled()
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            result = n._post("test")
        assert result is False

    def test_post_connection_error_retorna_false(self):
        import requests
        n = make_notifier_enabled()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            result = n._post("test")
        assert result is False


class TestTelegramNotifierWeeklySummary:
    """Tests del método notify_weekly_summary."""

    def test_envia_mensaje_con_ganancia_positiva(self):
        n = make_notifier_enabled()
        with patch.object(n, "send") as mock_send:
            n.notify_weekly_summary(
                trades=42,
                ganancia_neta=3.75,
                mejor_par="BTC/USDT",
                mejor_par_profit=2.50,
                peor_par="SOL/USDT",
                peor_par_profit=-0.30,
            )
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "Resumen semanal" in msg
        assert "42" in msg
        assert "+$3.75" in msg
        assert "BTC/USDT" in msg
        assert "+$2.50" in msg
        assert "SOL/USDT" in msg
        assert "-$0.30" in msg

    def test_envia_mensaje_con_ganancia_negativa(self):
        n = make_notifier_enabled()
        with patch.object(n, "send") as mock_send:
            n.notify_weekly_summary(
                trades=5,
                ganancia_neta=-1.20,
                mejor_par="ETH/USDT",
                mejor_par_profit=0.10,
                peor_par="BTC/USDT",
                peor_par_profit=-1.30,
            )
        msg = mock_send.call_args[0][0]
        assert "-$1.20" in msg
        assert "ETH/USDT" in msg
        assert "BTC/USDT" in msg

    def test_mejor_par_con_signo_positivo(self):
        n = make_notifier_enabled()
        with patch.object(n, "send") as mock_send:
            n.notify_weekly_summary(
                trades=10,
                ganancia_neta=5.0,
                mejor_par="BTC/USDT",
                mejor_par_profit=5.0,
                peor_par="BTC/USDT",
                peor_par_profit=5.0,
            )
        msg = mock_send.call_args[0][0]
        assert "+$5.00" in msg

    def test_peor_par_con_signo_negativo(self):
        n = make_notifier_enabled()
        with patch.object(n, "send") as mock_send:
            n.notify_weekly_summary(
                trades=3,
                ganancia_neta=-0.50,
                mejor_par="ETH/USDT",
                mejor_par_profit=-0.10,
                peor_par="SOL/USDT",
                peor_par_profit=-0.40,
            )
        msg = mock_send.call_args[0][0]
        assert "-$0.40" in msg

    def test_no_envia_si_deshabilitado(self):
        n = TelegramNotifier(token=None, chat_id=None)
        with patch.object(n, "_post") as mock_post:
            n.notify_weekly_summary(
                trades=1,
                ganancia_neta=0.0,
                mejor_par="BTC/USDT",
                mejor_par_profit=0.0,
                peor_par="BTC/USDT",
                peor_par_profit=0.0,
            )
        mock_post.assert_not_called()

    def test_contiene_trades_count(self):
        n = make_notifier_enabled()
        with patch.object(n, "send") as mock_send:
            n.notify_weekly_summary(
                trades=123,
                ganancia_neta=1.0,
                mejor_par="BTC/USDT",
                mejor_par_profit=1.0,
                peor_par="BTC/USDT",
                peor_par_profit=1.0,
            )
        msg = mock_send.call_args[0][0]
        assert "123" in msg
