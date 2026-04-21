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
        assert (tmp_path / "grid_state.json").exists()

    def test_cancela_ordenes_huerfanas_antes_de_inicializar(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """En modo live, cancela órdenes abiertas del exchange antes de crear la grilla."""
        sample_config["dry_run"] = False
        mock_client.dry_run = False
        mock_client._exchange.fetch_open_orders.return_value = [
            {"id": "orphan_1", "symbol": "BTC/USDT"},
            {"id": "orphan_2", "symbol": "BTC/USDT"},
        ]

        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        bot.run(max_cycles=1)

        assert mock_client._exchange.cancel_order.call_count == 2

    def test_no_cancela_huerfanas_en_dry_run(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """En dry_run, no se llama al exchange para cancelar órdenes huérfanas."""
        bot = make_bot_loop(sample_config, mock_client, tmp_path, mocker)
        bot.run(max_cycles=1)

        mock_client._exchange.cancel_order.assert_not_called()

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
# Tests de chequeo de balance antes de recentrar
# ------------------------------------------------------------------

class TestBotLoopRecenterBalance:
    """Verifica que _recenter_grid bloquea el recentrado si no hay USDT."""

    def _make_bot_out_of_range(self, sample_config, mock_client, tmp_path, mocker):
        """Crea un BotLoop con precio permanentemente fuera de rango."""
        mocker.patch(_SLEEP_PATH)
        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 99999.0
        pr.is_price_in_range.return_value = False
        from src.core.order_manager import OrderManager as _OM
        om = _OM(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol=sample_config["grid"]["symbol"],
            max_order_usdt=sample_config["grid"]["max_order_usdt"],
        )
        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        return bot, om, gs

    def test_recenter_bloqueado_si_usdt_insuficiente(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Si USDT disponible < min_order_usdt, _recenter_grid retorna False y no coloca órdenes."""
        mock_client._exchange.fetch_balance.return_value = {
            "free": {"USDT": 0.1},
        }
        sample_config["risk"]["min_order_usdt"] = 0.5

        bot, om, gs = self._make_bot_out_of_range(
            sample_config, mock_client, tmp_path, mocker
        )
        bot.run(max_cycles=1)

        # El ciclo se cuenta aunque el recentrado haya sido bloqueado
        assert bot.total_cycles == 1
        # No debe haber llamadas a create_limit_order (recentrado fue bloqueado)
        mock_client._exchange.create_order.assert_not_called()

    def test_recenter_procede_si_usdt_suficiente(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Si hay USDT suficiente, _recenter_grid procede normalmente."""
        mock_client._exchange.fetch_balance.return_value = {
            "free": {"USDT": 10.0},
        }
        sample_config["risk"]["min_order_usdt"] = 0.5

        bot, om, gs = self._make_bot_out_of_range(
            sample_config, mock_client, tmp_path, mocker
        )
        bot.run(max_cycles=1)

        assert bot.total_cycles == 1
        # No debe activar circuit breaker
        assert bot._risk_manager.circuit_open is False

    def _make_bot_no_usdt(self, sample_config, mock_client, tmp_path, mocker):
        """Crea un BotLoop con precio fuera de rango y USDT insuficiente."""
        mocker.patch(_SLEEP_PATH)
        mock_client._exchange.fetch_balance.return_value = {"free": {"USDT": 0.0}}
        sample_config["risk"]["min_order_usdt"] = 0.5

        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 99999.0
        pr.is_price_in_range.return_value = False
        mock_notifier = mocker.MagicMock()
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol=sample_config["grid"]["symbol"],
            max_order_usdt=sample_config["grid"]["max_order_usdt"],
        )
        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
            notifier=mock_notifier,
        )
        return bot, mock_notifier

    def test_notifier_llamado_cuando_usdt_insuficiente(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Verifica que se notifica por Telegram en el primer ciclo sin USDT."""
        bot, mock_notifier = self._make_bot_no_usdt(
            sample_config, mock_client, tmp_path, mocker
        )
        bot.run(max_cycles=1)

        mock_notifier.notify_no_usdt_available.assert_called_once_with(
            usdt_available=0.0,
            required=0.5,
        )

    def test_notifier_no_se_repite_en_ciclos_seguidos(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """El throttling silencia las notificaciones repetidas dentro de la ventana de 6h."""
        bot, mock_notifier = self._make_bot_no_usdt(
            sample_config, mock_client, tmp_path, mocker
        )
        bot.run(max_cycles=5)

        # Solo debe haberse enviado UNA notificación aunque hubo 5 ciclos sin USDT
        mock_notifier.notify_no_usdt_available.assert_called_once()

    def test_notifier_se_resetea_cuando_hay_usdt_de_nuevo(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Tras un recentrado exitoso, el timestamp se limpia y el próximo bloqueo notifica."""
        mocker.patch(_SLEEP_PATH)
        sample_config["risk"]["min_order_usdt"] = 0.5
        # cooldown=1 para que en solo 3 ciclos se puedan probar 2 notificaciones:
        # Ciclo 1: sin USDT → bloquea, notifica, cooldown=1
        # Ciclo 2: cooldown 1→0 (decrementa), intenta → USDT=10 → recentra, resetea timestamp
        # Ciclo 3: cooldown=0, intenta → sin USDT → bloquea, notifica de nuevo
        sample_config.setdefault("trailing", {})["cooldown_cycles_after_error"] = 1

        # _initialize_grid siempre lee balance (1 llamada extra al inicio)
        mock_client._exchange.fetch_balance.side_effect = [
            {"free": {"USDT": 10.0}},  # _initialize_grid
            {"free": {"USDT": 0.0}},   # ciclo 1: bloqueado
            {"free": {"USDT": 10.0}},  # ciclo 2: procede (cooldown ya decrementado a 0)
            {"free": {"USDT": 0.0}},   # ciclo 3: bloqueado de nuevo
        ]

        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 99999.0
        pr.is_price_in_range.return_value = False
        mock_notifier = mocker.MagicMock()
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol=sample_config["grid"]["symbol"],
            max_order_usdt=sample_config["grid"]["max_order_usdt"],
        )
        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
            notifier=mock_notifier,
        )
        bot.run(max_cycles=3)

        # Ciclo 1 y ciclo 3 notifican; ciclo 2 no (hay USDT, procede el recentrado)
        assert mock_notifier.notify_no_usdt_available.call_count == 2

    def test_recenter_usa_default_min_order_si_no_configurado(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Si min_order_usdt no está en risk_cfg, usa el default 0.5."""
        # Remover min_order_usdt del config para forzar el default
        sample_config["risk"].pop("min_order_usdt", None)
        mock_client._exchange.fetch_balance.return_value = {
            "free": {"USDT": 0.3},  # < 0.5 (default)
        }

        bot, om, gs = self._make_bot_out_of_range(
            sample_config, mock_client, tmp_path, mocker
        )
        bot.run(max_cycles=1)

        # Con 0.3 USDT y default de 0.5, el recentrado debe bloquearse
        mock_client._exchange.create_order.assert_not_called()


# ------------------------------------------------------------------
# Tests de cooldown tras recentrado fallido
# ------------------------------------------------------------------

class TestBotLoopRecenterCooldown:
    """Verifica que el cooldown evita el loop infinito de reintentos."""

    def _make_bot_out_of_range_no_usdt(self, sample_config, mock_client, tmp_path, mocker, cooldown):
        mocker.patch(_SLEEP_PATH)
        sample_config["risk"]["min_order_usdt"] = 0.5
        sample_config.setdefault("trailing", {})["cooldown_cycles_after_error"] = cooldown
        mock_client._exchange.fetch_balance.return_value = {"free": {"USDT": 0.0}}

        gs = GridState(state_file=tmp_path / "grid_state.json")
        rm = RiskManager(sample_config["risk"], sample_config["loop"])
        pr = mocker.MagicMock(spec=PriceReader)
        pr.get_current_price.return_value = 99999.0
        pr.is_price_in_range.return_value = False
        om = OrderManager(
            client=mock_client,
            risk_manager=rm,
            grid_state=gs,
            symbol=sample_config["grid"]["symbol"],
            max_order_usdt=sample_config["grid"]["max_order_usdt"],
        )
        bot = BotLoop(
            config=sample_config,
            client=mock_client,
            price_reader=pr,
            risk_manager=rm,
            grid_state=gs,
            order_manager=om,
        )
        return bot

    def test_cooldown_limita_llamadas_a_fetch_balance(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Con cooldown=10 y 6 ciclos, solo hay 2 llamadas a fetch_balance: init + ciclo 1.
        El cooldown supera el número de ciclos, por lo que nunca se reintenta dentro de la ventana."""
        bot = self._make_bot_out_of_range_no_usdt(
            sample_config, mock_client, tmp_path, mocker, cooldown=10
        )
        bot.run(max_cycles=6)

        # _initialize_grid (1) + primer intento fallido (1) = 2 llamadas totales.
        # Los ciclos 2-6 omiten fetch_balance porque el cooldown (10) no expira antes.
        assert mock_client._exchange.fetch_balance.call_count == 2

    def test_cooldown_reintenta_despues_de_n_ciclos(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Con cooldown=2, el tercer ciclo vuelve a intentar el recentrado."""
        bot = self._make_bot_out_of_range_no_usdt(
            sample_config, mock_client, tmp_path, mocker, cooldown=2
        )
        # 4 ciclos: init(1) + ciclo1 falla(1) + ciclos2-3 cooldown + ciclo4 reintento(1)
        bot.run(max_cycles=4)

        assert mock_client._exchange.fetch_balance.call_count == 3

    def test_sin_cooldown_reintenta_en_cada_ciclo(
        self, sample_config, mock_client, tmp_path, mocker
    ):
        """Con cooldown=0, cada ciclo out-of-range intenta el recentrado (comportamiento anterior)."""
        bot = self._make_bot_out_of_range_no_usdt(
            sample_config, mock_client, tmp_path, mocker, cooldown=0
        )
        bot.run(max_cycles=3)

        # init(1) + ciclo1(1) + ciclo2(1) + ciclo3(1) = 4 llamadas
        assert mock_client._exchange.fetch_balance.call_count == 4


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
