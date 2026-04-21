"""
Fixtures compartidos para los tests del OKX Grid Bot.
"""
from __future__ import annotations

import pytest
import ccxt

from src.connectors.okx_client import OKXClient
from src.risk.risk_manager import RiskManager
from src.strategy.grid_state import GridState


# ------------------------------------------------------------------
# Silenciar Telegram en TODOS los tests
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _silence_telegram(mocker):
    """
    Evita que los tests envíen notificaciones reales a Telegram.

    Se aplica automáticamente a cada test. Parchea la clase
    TelegramNotifier en el módulo bot_loop para que cualquier
    instancia creada implícitamente (cuando no se inyecta notifier)
    sea un MagicMock silencioso.

    Los tests que inyectan su propio mock_notifier no se ven
    afectados porque BotLoop usa ese objeto en lugar de instanciar
    la clase parcheada.
    """
    mocker.patch("src.core.bot_loop.TelegramNotifier")


# ------------------------------------------------------------------
# Configuración de prueba
# ------------------------------------------------------------------

@pytest.fixture
def sample_config() -> dict:
    return {
        "grid": {
            "symbol": "BTC/USDT",
            "price_min": 60000.0,
            "price_max": 75000.0,
            "num_levels": 5,
            "total_capital_usdt": 10.0,
            "max_order_usdt": 2.5,
        },
        "risk": {
            "max_daily_loss_usdt": 2.0,
            "max_open_orders": 10,
            "min_profit_per_trade_pct": 0.3,
        },
        "exchange": {"sandbox": False},
        "loop": {
            "interval_seconds": 5,
            "max_consecutive_errors": 3,
        },
        "logging": {"level": "DEBUG"},
        "dry_run": True,
    }


# ------------------------------------------------------------------
# Mock del exchange (ccxt)
# ------------------------------------------------------------------

@pytest.fixture
def mock_exchange(mocker):
    """Mock de ccxt.okx con respuestas predefinidas para tests."""
    exchange = mocker.MagicMock(spec=ccxt.okx)
    exchange.fetch_ticker.return_value = {
        "bid": 67499.0,
        "ask": 67501.0,
        "last": 67500.0,
    }
    exchange.fetch_balance.return_value = {
        "USDT": {"free": 10.0, "used": 0.0, "total": 10.0},
        "free": {"USDT": 10.0},
    }
    exchange.create_limit_buy_order.return_value = {
        "id": "order_buy_123",
        "status": "open",
    }
    exchange.create_limit_sell_order.return_value = {
        "id": "order_sell_123",
        "status": "open",
    }
    exchange.fetch_open_orders.return_value = []
    exchange.cancel_order.return_value = {"id": "order_buy_123", "status": "canceled"}
    return exchange


@pytest.fixture
def mock_client(mock_exchange) -> OKXClient:
    """OKXClient con exchange mockeado y DRY_RUN=True."""
    return OKXClient(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
        dry_run=True,
        exchange=mock_exchange,
    )


@pytest.fixture
def mock_client_live(mock_exchange) -> OKXClient:
    """OKXClient con exchange mockeado y DRY_RUN=False (para tests de reciclado real)."""
    return OKXClient(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
        dry_run=False,
        exchange=mock_exchange,
    )


# ------------------------------------------------------------------
# RiskManager por defecto
# ------------------------------------------------------------------

@pytest.fixture
def risk_manager(sample_config) -> RiskManager:
    return RiskManager(
        risk_config=sample_config["risk"],
        loop_config=sample_config["loop"],
    )


# ------------------------------------------------------------------
# GridState con directorio temporal
# ------------------------------------------------------------------

@pytest.fixture
def grid_state(tmp_path) -> GridState:
    return GridState(state_file=tmp_path / "grid_state.json")
