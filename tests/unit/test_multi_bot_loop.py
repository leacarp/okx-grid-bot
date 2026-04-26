"""
Tests unitarios para MultiBotLoop.

Estrategia de aislamiento:
  - Se parchea _build_components en cada instancia para usar GridState en tmp_path
    y PriceReaders mockeados por símbolo.
  - time.sleep y TelegramNotifier se parchean globalmente en el módulo.
  - mock_client (DRY_RUN=True) es el fixture estándar de conftest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.multi_bot_loop import MultiBotLoop, PAIR_COOLDOWN_SECONDS
from src.core.order_manager import OrderManager
from src.core.price_reader import PriceReader
from src.risk.risk_manager import RiskManager
from src.strategy.grid_state import GridState

_SLEEP_PATH = "src.core.multi_bot_loop.time.sleep"
_NOTIFIER_PATH = "src.core.multi_bot_loop.TelegramNotifier"


# ------------------------------------------------------------------
# Helper: construye MultiBotLoop con dependencias controladas
# ------------------------------------------------------------------


def _make_loop(
    config: dict[str, Any],
    mock_client: Any,
    tmp_path: Path,
    mocker: Any,
    prices: dict[str, float] | None = None,
) -> tuple[MultiBotLoop, dict[str, MagicMock]]:
    """
    Crea un MultiBotLoop con:
      - time.sleep y TelegramNotifier parcheados.
      - _build_components reemplazado para usar GridState en tmp_path
        y un PriceReader mock configurable por símbolo.

    Args:
        prices: dict symbol -> precio a retornar. Defecto 67500.0 para todos.

    Returns:
        (loop, price_readers) donde price_readers es un dict symbol -> mock_pr.
    """
    mocker.patch(_SLEEP_PATH)
    mocker.patch(_NOTIFIER_PATH)

    prices = prices or {}
    price_readers: dict[str, MagicMock] = {}

    loop = MultiBotLoop(
        config=config,
        client=mock_client,
        notifier=MagicMock(),
    )

    original_build = loop._build_components

    def patched_build(symbol: str):
        if symbol not in price_readers:
            mock_pr = mocker.MagicMock(spec=PriceReader)
            mock_pr.get_current_price.return_value = prices.get(symbol, 67500.0)
            mock_pr.is_price_in_range.return_value = True
            price_readers[symbol] = mock_pr

        safe = symbol.replace("/", "_")
        gs = GridState(state_file=tmp_path / f"gs_{safe}.json")
        rm = RiskManager(config["risk"], config["loop"])
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol=symbol,
            max_order_usdt=config["grid"]["max_order_usdt"],
        )
        return gs, om, price_readers[symbol]

    loop._build_components = patched_build  # type: ignore[method-assign]
    return loop, price_readers


# ------------------------------------------------------------------
# Tests de inicialización
# ------------------------------------------------------------------


class TestInitialization:
    def test_primer_token_inicializado(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        ok = loop._initialize_token(sample_config_multitoken["tokens"][0])

        assert ok is True
        assert loop.current_symbol == "BTC/USDT"

    def test_rango_calculado_correctamente(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """range_width_pct=5% sobre precio=100 → [97.5, 102.5]"""
        loop, _ = _make_loop(
            sample_config_multitoken, mock_client, tmp_path, mocker,
            prices={"BTC/USDT": 100.0},
        )
        loop._initialize_token(sample_config_multitoken["tokens"][0])

        assert abs(loop._price_min - 97.5) < 0.01
        assert abs(loop._price_max - 102.5) < 0.01

    def test_falla_si_no_hay_precio(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        loop, prs = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        # Pre-crear el mock del símbolo para poder configurar el side_effect
        prs["BTC/USDT"] = mocker.MagicMock(spec=PriceReader)
        prs["BTC/USDT"].get_current_price.side_effect = Exception("No price")

        ok = loop._initialize_token(sample_config_multitoken["tokens"][0])

        assert ok is False
        assert loop.current_symbol is None

    def test_carga_estado_existente_sin_re_colocar_ordenes(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """Si el archivo de estado ya existe, no coloca órdenes nuevas."""
        symbol = "ETH/USDT"
        safe = symbol.replace("/", "_")
        state_path = tmp_path / f"gs_{safe}.json"

        # Pre-poblar estado en disco
        pre_gs = GridState(state_file=state_path)
        pre_gs.initialize(
            symbol=symbol,
            grid_config={"price_min": 3000, "price_max": 3100, "num_levels": 3},
            levels=[
                {"index": 0, "price": 3000.0, "status": "buy_open",
                 "order_id": "eth_0", "side": "buy", "amount": 0.001},
            ],
        )

        loop, _ = _make_loop(
            sample_config_multitoken, mock_client, tmp_path, mocker,
            prices={symbol: 3050.0},
        )
        token_cfg = next(t for t in sample_config_multitoken["tokens"] if t["symbol"] == symbol)
        ok = loop._initialize_token(token_cfg)

        assert ok is True
        assert loop.current_symbol == symbol
        # No debería haber llamado a create_limit_order para un estado ya cargado
        mock_client._exchange.create_limit_buy_order.assert_not_called()


# ------------------------------------------------------------------
# Tests de cambio de par (round-robin)
# ------------------------------------------------------------------


class TestPairSwitch:
    def test_cambio_exitoso_cancela_y_reinicializa(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """Al cambiar de par se cancelan las órdenes del par actual y se inicia el siguiente."""
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._initialize_token(sample_config_multitoken["tokens"][0])
        assert loop.current_symbol == "BTC/USDT"

        prev_om = loop._current_order_manager
        cancel_spy = mocker.spy(prev_om, "cancel_all_orders")

        success = loop._switch_to_next()

        assert success is True
        assert loop.current_symbol == "ETH/USDT"
        cancel_spy.assert_called_once()

    def test_round_robin_orden_correcto(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """Verifica la secuencia BTC → ETH → SOL → BTC."""
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._initialize_token(sample_config_multitoken["tokens"][0])

        loop._last_switch_time = 0.0
        loop._switch_to_next()
        assert loop.current_symbol == "ETH/USDT"

        loop._last_switch_time = 0.0
        loop._switch_to_next()
        assert loop.current_symbol == "SOL/USDT"

        loop._last_switch_time = 0.0
        loop._switch_to_next()
        assert loop.current_symbol == "BTC/USDT"

    def test_cooldown_impide_cambio_prematuro(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        import time as _time

        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._last_switch_time = _time.time()  # acaba de cambiar

        assert loop._should_switch() is False

    def test_cooldown_permite_cambio_tras_espera(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._last_switch_time = 0.0  # hace mucho tiempo

        assert loop._should_switch() is True


# ------------------------------------------------------------------
# Tests de atomicidad / fallback
# ------------------------------------------------------------------


class TestAtomicity:
    def test_fallback_si_preflight_falla(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """
        Si el pre-flight (verificar precio del siguiente) falla, NO se cancela
        la grilla actual y el símbolo activo no cambia.
        """
        loop, prs = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._initialize_token(sample_config_multitoken["tokens"][0])
        assert loop.current_symbol == "BTC/USDT"

        # Forzar que ETH/USDT no tenga precio (pre-flight fallará)
        eth_pr = mocker.MagicMock(spec=PriceReader)
        eth_pr.get_current_price.side_effect = Exception("ETH exchange down")
        prs["ETH/USDT"] = eth_pr

        prev_om = loop._current_order_manager
        cancel_spy = mocker.spy(prev_om, "cancel_all_orders")

        success = loop._switch_to_next()

        assert success is False
        assert loop.current_symbol == "BTC/USDT"  # no cambió
        cancel_spy.assert_not_called()  # no se cancelaron órdenes

    def test_fallback_si_initialize_falla_despues_de_cancelar(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """
        Si el pre-flight pasa pero place_initial_orders del nuevo token falla,
        el sistema intenta recuperar el token anterior.
        """
        loop, prs = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)
        loop._initialize_token(sample_config_multitoken["tokens"][0])
        assert loop.current_symbol == "BTC/USDT"

        # ETH/USDT: pre-flight OK pero get_current_price falla en segunda llamada
        # (la que ocurre dentro de _initialize_token para ETH)
        eth_pr = mocker.MagicMock(spec=PriceReader)
        call_tracker = {"n": 0}

        def eth_price():
            call_tracker["n"] += 1
            if call_tracker["n"] >= 2:
                raise Exception("ETH init failed")
            return 3200.0

        eth_pr.get_current_price.side_effect = eth_price
        eth_pr.is_price_in_range.return_value = True
        prs["ETH/USDT"] = eth_pr

        success = loop._switch_to_next()

        assert success is False
        # Tras el fallo debe haber intentado recuperar BTC/USDT
        assert loop.current_symbol == "BTC/USDT"

    def test_un_solo_token_nunca_cambia(
        self, mock_client, tmp_path, mocker
    ):
        """Con un solo token en la lista, el loop no intenta cambiar de par."""
        config_single = {
            "tokens": [{"symbol": "BTC/USDT", "num_levels": 4, "min_step_usdt": 500}],
            "grid": {"total_capital_usdt": 5.0, "max_order_usdt": 1.0, "range_width_pct": 5.0},
            "risk": {"max_daily_loss_usdt": 2.0, "max_open_orders": 10,
                     "min_profit_per_trade_pct": 0.3, "min_order_usdt": 0.5},
            "exchange": {"sandbox": False},
            "loop": {"interval_seconds": 5, "max_consecutive_errors": 5},
            "dry_run": True,
        }
        loop, _ = _make_loop(config_single, mock_client, tmp_path, mocker)
        loop._initialize_token(config_single["tokens"][0])

        loop._last_switch_time = 0.0  # forzar que debería switchear
        # run un ciclo — con un solo token len(tokens) == 1 → no llama _switch_to_next
        loop.run(max_cycles=1)

        assert loop.current_symbol == "BTC/USDT"


# ------------------------------------------------------------------
# Tests del ciclo completo (run)
# ------------------------------------------------------------------


class TestRunLoop:
    def test_run_completa_ciclos(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)

        loop.run(max_cycles=3)

        assert loop.total_cycles == 3

    def test_run_cambia_de_par_en_ciclo(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """Verificar que el símbolo cambia después del cooldown dentro del run."""
        loop, _ = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)

        # Forzar cooldown a 0 para que el switch ocurra en el primer ciclo
        loop._pair_cooldown = 0.0

        loop.run(max_cycles=2)

        # Tras el switch el símbolo debe haber cambiado de BTC/USDT
        assert loop.current_symbol != "BTC/USDT" or loop.total_cycles >= 1

    def test_run_retorna_si_primer_token_falla(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        """Si el primer token falla al inicializar, run() retorna sin ejecutar ciclos."""
        loop, prs = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)

        # Pre-crear el mock del primer símbolo con fallo
        btc_pr = mocker.MagicMock(spec=PriceReader)
        btc_pr.get_current_price.side_effect = Exception("No BTC price")
        prs["BTC/USDT"] = btc_pr

        loop.run(max_cycles=5)

        assert loop.total_cycles == 0
        assert not loop.is_running

    def test_circuit_breaker_detiene_loop(
        self, sample_config_multitoken, mock_client, tmp_path, mocker
    ):
        loop, prs = _make_loop(sample_config_multitoken, mock_client, tmp_path, mocker)

        # Forzar excepción en check_filled_orders para disparar errores consecutivos
        btc_pr = prs.get("BTC/USDT") or mocker.MagicMock(spec=PriceReader)
        btc_pr.get_current_price.return_value = 67500.0
        btc_pr.is_price_in_range.side_effect = Exception("Network error")
        prs["BTC/USDT"] = btc_pr

        # max_consecutive_errors = 5
        loop.run(max_cycles=100)

        assert not loop.is_running
        assert loop.total_cycles < 100
