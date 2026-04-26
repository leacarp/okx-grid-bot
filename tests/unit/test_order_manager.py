"""
Tests unitarios para OrderManager.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import ccxt

from src.core.order_manager import OrderManager
from src.core.pnl_tracker import PnLTracker
from src.risk.risk_manager import RiskManager
from src.strategy.grid_calculator import GridCalculator
from src.strategy.grid_state import GridState


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_order_manager(mock_client, grid_state, risk_manager=None, pnl_tracker=None):
    if risk_manager is None:
        risk_manager = RiskManager(
            risk_config={"max_daily_loss_usdt": 2.0, "max_open_orders": 10},
            loop_config={"max_consecutive_errors": 5},
        )
    return OrderManager(
        client=mock_client,
        risk_manager=risk_manager,
        grid_state=grid_state,
        symbol="BTC/USDT",
        max_order_usdt=2.5,
        pnl_tracker=pnl_tracker,
    )


def make_grid_levels():
    calc = GridCalculator(60000, 75000, 5, 10.0, 2.5)
    return calc.calculate()


# ------------------------------------------------------------------
# Tests de place_initial_orders
# ------------------------------------------------------------------

class TestPlaceInitialOrders:
    def test_coloca_ordenes_en_dry_run(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()
        current_price = 67500.0

        placed = om.place_initial_orders(levels, current_price)

        assert placed > 0
        assert len(grid_state.levels) == 5

    def test_buy_debajo_del_precio_actual(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()
        current_price = 67500.0

        om.place_initial_orders(levels, current_price)

        for level in grid_state.levels:
            if level["price"] < current_price:
                assert level["side"] == "buy"
            else:
                assert level["side"] == "sell"

    def test_sell_encima_del_precio_actual(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()
        current_price = 67500.0

        om.place_initial_orders(levels, current_price)

        high_levels = [l for l in grid_state.levels if l["price"] >= current_price]
        assert all(l["side"] == "sell" for l in high_levels)

    def test_ordenes_tienen_order_id(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()

        om.place_initial_orders(levels, current_price=67500.0)

        for level in grid_state.levels:
            if level["status"].endswith("_open"):
                assert level["order_id"] is not None

    def test_no_coloca_si_ya_existe_estado(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()

        # Primera llamada: coloca
        om.place_initial_orders(levels, 67500.0)

        # Segunda llamada: debe omitir
        placed = om.place_initial_orders(levels, 67500.0)
        assert placed == 0

    def test_ordenes_bloqueadas_por_riesgo(self, mock_client, grid_state):
        risk = RiskManager(
            risk_config={"max_daily_loss_usdt": 2.0, "max_open_orders": 10},
            loop_config={"max_consecutive_errors": 5},
        )
        risk._circuit_open = True  # forzar circuit abierto

        om = make_order_manager(mock_client, grid_state, risk_manager=risk)
        levels = make_grid_levels()
        placed = om.place_initial_orders(levels, 67500.0)

        assert placed == 0
        # Solo los niveles BUY pasan por el risk check; los SELL son sell_pending
        buy_levels = [l for l in levels if l.price < 67500.0]
        blocked = [l for l in grid_state.levels if l["status"] == "blocked"]
        assert len(blocked) == len(buy_levels)

    def test_estado_persiste_en_disco(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        om.place_initial_orders(make_grid_levels(), 67500.0)

        assert grid_state._path.exists()

    def test_insufficient_funds_no_aborta_inicializacion(
        self, mock_client_live, grid_state
    ):
        """
        Si OKX responde InsufficientFunds, la orden falla silenciosamente
        (level queda en 'error'), pero place_initial_orders no propaga la excepción.
        El bot puede seguir corriendo con 0 órdenes colocadas.
        """
        mock_client_live._exchange.create_order.side_effect = ccxt.InsufficientFunds(
            "Order failed. Your available USDT balance is insufficient."
        )
        om = make_order_manager(mock_client_live, grid_state)
        levels = make_grid_levels()

        placed = om.place_initial_orders(levels, 67500.0)

        assert placed == 0
        error_levels = [l for l in grid_state.levels if l["status"] == "error"]
        buy_levels = [l for l in levels if l.price < 67500.0]
        assert len(error_levels) == len(buy_levels)


    def test_base_disponible_bajo_minimo_okx_no_coloca_sells(self, mock_client, grid_state):
        """
        Si el BTC disponible dividido entre los niveles SELL queda por debajo de
        0.00001 BTC (mínimo OKX), no debe colocarse ninguna orden SELL.
        Todos los niveles sobre el precio actual deben quedar en 'sell_pending'.
        """
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()
        current_price = 67500.0

        # 0.00001127 BTC / 2 niveles SELL ≈ 0.000005635 BTC < 0.00001 → debajo del mínimo
        base_available_insuficiente = 0.00001127

        placed = om.place_initial_orders(
            levels, current_price, base_available=base_available_insuficiente
        )

        sell_levels = [l for l in grid_state.levels if l["price"] >= current_price]
        # Ningún nivel SELL debe estar en sell_open
        assert all(l["status"] == "sell_pending" for l in sell_levels)
        # Solo los BUYs deben estar colocados
        buy_levels = [l for l in grid_state.levels if l["price"] < current_price]
        assert placed == len([l for l in buy_levels if l["status"] == "buy_open"])

    def test_base_disponible_suficiente_coloca_sells(self, mock_client, grid_state):
        """Si el BTC por nivel supera 0.00001, las órdenes SELL deben colocarse."""
        om = make_order_manager(mock_client, grid_state)
        levels = make_grid_levels()
        current_price = 67500.0

        # 0.0001 BTC / 2 niveles SELL = 0.00005 BTC > 0.00001 → válido
        base_available_ok = 0.0001

        om.place_initial_orders(levels, current_price, base_available=base_available_ok)

        sell_open = [
            l for l in grid_state.levels
            if l["price"] >= current_price and l["status"] == "sell_open"
        ]
        assert len(sell_open) > 0


# ------------------------------------------------------------------
# Tests de check_filled_orders
# ------------------------------------------------------------------

class TestCheckFilledOrders:
    def test_dry_run_sin_precio_retorna_lista_vacia(self, mock_client, grid_state):
        assert mock_client.dry_run is True
        om = make_order_manager(mock_client, grid_state)
        om.place_initial_orders(make_grid_levels(), 67500.0)

        # Sin current_price no se simula ningún fill
        filled = om.check_filled_orders()
        assert filled == []

    # ------------------------------------------------------------------
    # Simulación de fills por cruce de precio (DRY_RUN)
    # ------------------------------------------------------------------

    def _setup_state_con_dos_ordenes(self, grid_state):
        """Estado con BUY a 60000 y SELL a 75000."""
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 75000.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )

    def test_dry_run_buy_ejecutada_cuando_precio_baja(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_state_con_dos_ordenes(grid_state)

        # Precio baja hasta el nivel de compra exacto
        filled = om.check_filled_orders(current_price=60000.0)

        assert len(filled) == 1
        assert filled[0]["side"] == "buy"
        assert filled[0]["price"] == 60000.0

    def test_dry_run_buy_ejecutada_cuando_precio_cae_debajo(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_state_con_dos_ordenes(grid_state)

        # Precio cae por debajo del nivel de compra
        filled = om.check_filled_orders(current_price=59000.0)

        assert any(l["side"] == "buy" for l in filled)

    def test_dry_run_sell_ejecutada_cuando_precio_sube(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_state_con_dos_ordenes(grid_state)

        # Precio sube hasta el nivel de venta exacto
        filled = om.check_filled_orders(current_price=75000.0)

        assert len(filled) == 1
        assert filled[0]["side"] == "sell"
        assert filled[0]["price"] == 75000.0

    def test_dry_run_precio_en_medio_no_ejecuta_nada(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_state_con_dos_ordenes(grid_state)

        # Precio entre BUY y SELL → nadie se ejecuta
        filled = om.check_filled_orders(current_price=67500.0)

        assert filled == []

    def test_dry_run_no_ejecuta_ordenes_ya_filled(self, mock_client, grid_state):
        """Órdenes con status != _open no deben aparecer como filled."""
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_filled",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 75000.0, "status": "cancelled",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )
        om = make_order_manager(mock_client, grid_state)

        # Aunque el precio cruce ambos niveles, no se reportan
        filled = om.check_filled_orders(current_price=50000.0)
        assert filled == []

    def test_dry_run_ejecuta_multiples_niveles_simultaneamente(self, mock_client, grid_state):
        """Una caída brusca puede ejecutar varios BUYs a la vez."""
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 3},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 67500.0, "status": "buy_open",
                 "order_id": "buy_1", "side": "buy", "amount": 0.00003},
                {"index": 2, "price": 75000.0, "status": "sell_open",
                 "order_id": "sell_2", "side": "sell", "amount": 0.00003},
            ],
        )
        om = make_order_manager(mock_client, grid_state)

        # Precio cae a 55000 → ambas BUYs ejecutadas
        filled = om.check_filled_orders(current_price=55000.0)

        buy_filled = [l for l in filled if l["side"] == "buy"]
        assert len(buy_filled) == 2

    def test_live_detecta_orden_ejecutada(self, mock_client_live, grid_state, mock_exchange):
        om = make_order_manager(mock_client_live, grid_state)

        # Simular estado con una orden "abierta"
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 5},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "ord_open_1", "side": "buy", "amount": 0.00003}],
        )

        # El exchange NO reporta esa orden como abierta → se ejecutó
        mock_exchange.fetch_open_orders.return_value = []

        filled = om.check_filled_orders()
        assert len(filled) == 1
        assert filled[0]["order_id"] == "ord_open_1"

    def test_live_orden_aun_abierta_no_aparece(self, mock_client_live, grid_state, mock_exchange):
        om = make_order_manager(mock_client_live, grid_state)

        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 5},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "ord_still_open", "side": "buy", "amount": 0.00003}],
        )

        # El exchange sí reporta la orden como abierta
        mock_exchange.fetch_open_orders.return_value = [{"id": "ord_still_open"}]

        filled = om.check_filled_orders()
        assert filled == []


# ------------------------------------------------------------------
# Tests de recycle_order
# ------------------------------------------------------------------

class TestRecycleOrder:
    def _setup_two_levels(self, grid_state):
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 63750, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 63750.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )

    def test_buy_ejecutada_crea_sell_en_nivel_superior(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_two_levels(grid_state)

        filled_level = grid_state.levels[0]  # index=0, side=buy
        new_order = om.recycle_order(filled_level)

        assert new_order is not None
        # El nivel 0 debe marcarse filled
        level_0 = next(l for l in grid_state.levels if l["index"] == 0)
        assert level_0["status"] == "buy_filled"

    def test_sell_ejecutada_crea_buy_en_nivel_inferior(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        self._setup_two_levels(grid_state)

        filled_level = grid_state.levels[1]  # index=1, side=sell
        new_order = om.recycle_order(filled_level)

        assert new_order is not None
        level_1 = next(l for l in grid_state.levels if l["index"] == 1)
        assert level_1["status"] == "sell_filled"

    def test_reciclar_sin_nivel_objetivo_retorna_none(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)

        # Sólo un nivel, no hay nivel superior para reciclar
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 1},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "buy_0", "side": "buy", "amount": 0.00003}],
        )

        filled = {"index": 0, "price": 60000.0, "side": "buy", "amount": 0.00003}
        result = om.recycle_order(filled)
        assert result is None


# ------------------------------------------------------------------
# Tests de cancel_all_orders
# ------------------------------------------------------------------

class TestCancelAllOrders:
    def test_cancela_ordenes_abiertas(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 3},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 67500.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
                {"index": 2, "price": 75000.0, "status": "buy_filled",
                 "order_id": "buy_2", "side": "buy", "amount": 0.00003},
            ],
        )

        cancelled = om.cancel_all_orders()

        assert cancelled == 2  # solo los que tienen status _open

    def test_sin_ordenes_abiertas_retorna_cero(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 1},
            [{"index": 0, "price": 60000.0, "status": "buy_filled",
              "order_id": "buy_0", "side": "buy", "amount": 0.00003}],
        )

        cancelled = om.cancel_all_orders()
        assert cancelled == 0

    def test_niveles_marcados_cancelled_tras_cancelar(self, mock_client, grid_state):
        om = make_order_manager(mock_client, grid_state)
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 75000, "num_levels": 1},
            [{"index": 0, "price": 60000.0, "status": "buy_open",
              "order_id": "buy_0", "side": "buy", "amount": 0.00003}],
        )

        om.cancel_all_orders()

        level = grid_state.levels[0]
        assert level["status"] == "cancelled"


# ------------------------------------------------------------------
# Tests de integración con PnLTracker
# ------------------------------------------------------------------

class TestPnLTrackerIntegracion:
    def _setup_two_levels(self, grid_state):
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 63750, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 63750.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )

    def test_reciclar_sell_registra_ciclo_en_pnl_tracker(self, mock_client, grid_state, tmp_path):
        tracker = PnLTracker(history_file=tmp_path / "hist.json")
        om = make_order_manager(mock_client, grid_state, pnl_tracker=tracker)
        self._setup_two_levels(grid_state)

        # SELL en index=1 se ejecuta → ciclo completo con BUY en index=0
        filled_sell = grid_state.levels[1]
        om.recycle_order(filled_sell)

        summary = tracker.get_summary()
        assert summary.total_cycles == 1
        assert summary.gross_profit > 0  # 63750 > 60000

    def test_reciclar_buy_no_registra_ciclo(self, mock_client, grid_state, tmp_path):
        tracker = PnLTracker(history_file=tmp_path / "hist.json")
        om = make_order_manager(mock_client, grid_state, pnl_tracker=tracker)
        self._setup_two_levels(grid_state)

        # BUY en index=0 se ejecuta → coloca SELL, pero el ciclo no está completo
        filled_buy = grid_state.levels[0]
        om.recycle_order(filled_buy)

        summary = tracker.get_summary()
        assert summary.total_cycles == 0  # sin ciclo completado

    def test_reciclar_sell_actualiza_grid_state_profit(self, mock_client, grid_state, tmp_path):
        tracker = PnLTracker(history_file=tmp_path / "hist.json")
        om = make_order_manager(mock_client, grid_state, pnl_tracker=tracker)
        self._setup_two_levels(grid_state)

        filled_sell = grid_state.levels[1]
        om.recycle_order(filled_sell)

        # grid_state.record_profit fue llamado → total_profit > 0
        assert grid_state.total_profit > 0

    def test_sin_pnl_tracker_reciclar_funciona_igual(self, mock_client, grid_state):
        """Sin PnLTracker el reciclado sigue funcionando sin errores."""
        om = make_order_manager(mock_client, grid_state, pnl_tracker=None)
        self._setup_two_levels(grid_state)

        filled_sell = grid_state.levels[1]
        result = om.recycle_order(filled_sell)

        assert result is not None  # la orden se recicló correctamente

    def test_live_usa_fills_reales_para_calcular_profit(
        self,
        mock_client_live,
        grid_state,
        tmp_path,
        mock_exchange,
    ):
        tracker = PnLTracker(history_file=tmp_path / "hist.json")
        om = make_order_manager(mock_client_live, grid_state, pnl_tracker=tracker)
        self._setup_two_levels(grid_state)

        # BUY previo ejecutado a 60010 con fee 0.05, SELL actual a 63740 con fee 0.07
        # net = (63740 - 60010) * 0.00003 - (0.05 + 0.07) = -0.0081
        mock_exchange.fetch_order_trades.side_effect = [
            [{"amount": 0.00003, "cost": 1.8003, "fee": {"cost": 0.05, "currency": "USDT"}}],
            [{"amount": 0.00003, "cost": 1.9122, "fee": {"cost": 0.07, "currency": "USDT"}}],
        ]

        filled_sell = grid_state.levels[1]
        om.recycle_order(filled_sell)

        summary = tracker.get_summary()
        assert summary.total_cycles == 1
        assert summary.net_profit == pytest.approx(-0.0081, abs=1e-8)
        assert grid_state.total_profit == pytest.approx(-0.0081, abs=1e-8)


# ------------------------------------------------------------------
# Tests de pre_order_check cableado en recycle_order
# ------------------------------------------------------------------

class TestPreOrderCheckEnRecycle:
    def test_circuit_breaker_bloquea_reciclar(self, mock_client, grid_state):
        risk = RiskManager(
            risk_config={"max_daily_loss_usdt": 2.0, "max_open_orders": 10},
            loop_config={"max_consecutive_errors": 5},
        )
        risk._circuit_open = True

        om = make_order_manager(mock_client, grid_state, risk_manager=risk)
        grid_state.initialize(
            "BTC/USDT",
            {"price_min": 60000, "price_max": 63750, "num_levels": 2},
            [
                {"index": 0, "price": 60000.0, "status": "buy_open",
                 "order_id": "buy_0", "side": "buy", "amount": 0.00003},
                {"index": 1, "price": 63750.0, "status": "sell_open",
                 "order_id": "sell_1", "side": "sell", "amount": 0.00003},
            ],
        )

        filled_sell = grid_state.levels[1]
        result = om.recycle_order(filled_sell)

        assert result is None  # bloqueado por circuit breaker
