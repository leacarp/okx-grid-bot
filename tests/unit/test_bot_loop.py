"""
Tests unitarios para BotLoop.

Se inyectan todas las dependencias para evitar llamadas reales a OKX
y se mockea time.sleep para que los tests sean instantáneos.
"""
from __future__ import annotations

import pytest

from src.core.bot_loop import BotLoop
from src.core.order_manager import OrderManager
from src.core.price_reader import PriceReader
from src.risk.risk_manager import RiskManager
from src.strategy.grid_state import GridState


# ------------------------------------------------------------------
# Helper: construye un BotLoop con todas las dependencias mockeadas
# ------------------------------------------------------------------

_SLEEP_PATH = "src.core.bot_loop.time.sleep"


def make_bot_loop(
    config,
    mock_client,
    tmp_path,
    mocker,
    price_side_effect=None,
    preload_state: bool = False,
):
    """
    Crea un BotLoop con dependencias inyectadas y time.sleep mockeado.

    Args:
        price_side_effect: Lista de precios que retornará get_current_price(),
                           o None para usar 67500.0 siempre.
        preload_state: Si True, pre-pobla el estado para saltar _initialize_grid.
    """
    mocker.patch(_SLEEP_PATH)  # sin esperas reales en tests

    gs = GridState(state_file=tmp_path / "grid_state.json")

    if preload_state:
        gs.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "preloaded_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 75000.0, "status": "sell_open",
                 "order_id": "preloaded_1", "side": "sell", "amount": 0.00003},
            ],
        )

    rm = RiskManager(config["risk"], config["loop"])
    pr = mocker.MagicMock(spec=PriceReader)

    if price_side_effect:
        pr.get_current_price.side_effect = price_side_effect
    else:
        pr.get_current_price.return_value = 67500.0

    pr.is_price_in_range.return_value = True

    om = OrderManager(
        client=mock_client,
        risk_manager=rm,
        grid_state=gs,
        symbol=config["grid"]["symbol"],
        max_order_usdt=config["grid"]["max_order_usdt"],
    )

    return BotLoop(
        config=config,
        client=mock_client,
        price_reader=pr,
        risk_manager=rm,
        grid_state=gs,
        order_manager=om,
    )


# ------------------------------------------------------------------
# Tests de inicialización de la grilla
# ------------------------------------------------------------------

class TestBotLoopInicializacion:
    def test_inicializa_grid_si_no_hay_estado(self, sample_config, mock_client, tmp_path, mocker):
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        bot.run(max_cycles=1)

        assert bot.total_cycles == 1
        # El estado debe haberse creado
        assert (tmp_path / "grid_state.json").exists()

    def test_carga_estado_existente_sin_reinicializar(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        # Crear estado previo
        gs = GridState(state_file=tmp_path / "grid_state.json")
        gs.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 5},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "pre_existing", "side": "buy", "amount": 0.00003}],
        )

        mocker.patch("time.sleep")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 67500.0
        pr.is_price_in_range.return_value = True
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol="BTC/USDT",
            max_order_usdt=2.5,
        )

        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        bot.run(max_cycles=1)

        # El estado pre-existente no debe haber cambiado
        assert gs.levels[0]["order_id"] == "pre_existing"


# ------------------------------------------------------------------
# Tests del ciclo principal
# ------------------------------------------------------------------

class TestBotLoopCiclo:
    def test_corre_n_ciclos(self, sample_config, mock_client, tmp_path, mocker):
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        bot.run(max_cycles=5)

        assert bot.total_cycles == 5

    def test_ciclo_con_precio_fuera_de_rango_no_es_error(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        mocker.patch("time.sleep")
        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 99999.0
        pr.is_price_in_range.return_value = False  # fuera de rango
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol="BTC/USDT",
            max_order_usdt=2.5,
        )

        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        bot.run(max_cycles=2)

        # Los ciclos se cuentan aunque el precio esté fuera de rango
        assert bot.total_cycles == 2
        # No debe haber circuit breaker
        assert rm.circuit_open is False

    def test_sleep_llamado_entre_ciclos(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        # Parchear directamente el tiempo en bot_loop para inspeccionar las llamadas
        sleep_mock = mocker.patch(_SLEEP_PATH)
        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 67500.0
        pr.is_price_in_range.return_value = True
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol="BTC/USDT",
            max_order_usdt=2.5,
        )
        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        bot.run(max_cycles=3)

        # sleep se llama entre ciclos pero no después del último
        assert sleep_mock.call_count == 2

    def test_errores_activan_circuit_breaker(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        mocker.patch(_SLEEP_PATH)
        # Pre-poblar estado para que la inicialización sea omitida; sólo el loop fallará
        gs = GridState(state_file=tmp_path / "grid_state.json")
        gs.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 2},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "ord0", "side": "buy", "amount": 0.00003}],
        )
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        # Los ciclos del loop lanzan excepción (no la inicialización)
        pr.get_current_price.side_effect = RuntimeError("error de red simulado")
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol="BTC/USDT",
            max_order_usdt=2.5,
        )

        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        bot.run(max_cycles=10)

        # max_consecutive_errors=3 → circuit breaker activo antes del ciclo 10
        assert rm.circuit_open is True
        assert bot.total_cycles < 10


# ------------------------------------------------------------------
# Tests de shutdown
# ------------------------------------------------------------------

class TestBotLoopShutdown:
    def test_shutdown_cancela_ordenes_abiertas(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        mocker.patch("time.sleep")
        gs = GridState(state_file=tmp_path / "grid_state.json")

        # Pre-poblar estado con órdenes abiertas
        gs.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 75000.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )

        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 67500.0
        pr.is_price_in_range.return_value = True
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol="BTC/USDT",
            max_order_usdt=2.5,
        )

        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        bot.run(max_cycles=1)

        # Tras el shutdown, las órdenes deben estar canceladas
        for level in gs.levels:
            assert level["status"] == "cancelled"


# ------------------------------------------------------------------
# Tests de propiedades
# ------------------------------------------------------------------

class TestBotLoopPropiedades:
    def test_total_cycles_empieza_en_cero(self, sample_config, mock_client, tmp_path, mocker):
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        assert bot.total_cycles == 0

    def test_is_running_false_antes_de_run(self, sample_config, mock_client, tmp_path, mocker):
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        assert bot.is_running is False

    def test_is_running_false_tras_run(self, sample_config, mock_client, tmp_path, mocker):
        # make_bot_loop ya parchea time.sleep internamente
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        bot.run(max_cycles=1)
        # _shutdown() debe poner _running en False
        assert bot.is_running is False
