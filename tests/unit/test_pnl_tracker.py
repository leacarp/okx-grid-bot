"""
Tests unitarios para PnLTracker y OKXClient.fetch_order_trades.

Cobertura objetivo: >= 85%
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.connectors.okx_client import OKXClient
from src.core.pnl_tracker import (
    CompletedCycle,
    FillRecord,
    PnLSummary,
    PnLTracker,
)


# ---------------------------------------------------------------------------
# Helpers y fixtures
# ---------------------------------------------------------------------------


def _make_fill(
    side: str = "buy",
    price: float = 80_000.0,
    amount: float = 0.001,
    fee: float = 0.08,
    order_id: str = "ord_001",
    symbol: str = "BTC/USDT",
) -> FillRecord:
    return FillRecord(
        order_id=order_id,
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        fee=fee,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture()
def tracker(tmp_path: Path) -> PnLTracker:
    """PnLTracker aislado en directorio temporal."""
    return PnLTracker(history_file=tmp_path / "trade_history.json")


@pytest.fixture()
def buy_fill() -> FillRecord:
    return _make_fill(side="buy", price=80_000.0, fee=0.08, order_id="buy_001")


@pytest.fixture()
def sell_fill() -> FillRecord:
    return _make_fill(side="sell", price=81_000.0, fee=0.081, order_id="sell_001")


@pytest.fixture()
def okx_dry() -> OKXClient:
    return OKXClient(
        api_key="key",
        api_secret="secret",
        passphrase="pass",
        dry_run=True,
        exchange=MagicMock(),
    )


@pytest.fixture()
def okx_live() -> OKXClient:
    return OKXClient(
        api_key="key",
        api_secret="secret",
        passphrase="pass",
        dry_run=False,
        exchange=MagicMock(),
    )


# ---------------------------------------------------------------------------
# record_fill
# ---------------------------------------------------------------------------


class TestRecordFill:
    def test_retorna_fill_record_con_datos_correctos(self, tracker: PnLTracker) -> None:
        fill = tracker.record_fill("ord_1", "BTC/USDT", "buy", 0.001, 80_000.0, 0.08)

        assert isinstance(fill, FillRecord)
        assert fill.order_id == "ord_1"
        assert fill.symbol == "BTC/USDT"
        assert fill.side == "buy"
        assert fill.amount == 0.001
        assert fill.price == 80_000.0
        assert fill.fee == 0.08

    def test_timestamp_es_iso_utc(self, tracker: PnLTracker) -> None:
        fill = tracker.record_fill("ord_2", "BTC/USDT", "sell", 0.001, 81_000.0, 0.081)
        dt = datetime.fromisoformat(fill.timestamp)
        assert dt.tzinfo is not None

    def test_fee_cero_es_valido(self, tracker: PnLTracker) -> None:
        fill = tracker.record_fill("ord_3", "ETH/USDT", "buy", 0.1, 2_000.0, 0.0)
        assert fill.fee == 0.0

    def test_side_invalido_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="side inválido"):
            tracker.record_fill("x", "BTC/USDT", "hold", 0.001, 80_000.0, 0.0)

    def test_amount_cero_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="amount"):
            tracker.record_fill("x", "BTC/USDT", "buy", 0.0, 80_000.0, 0.0)

    def test_amount_negativo_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="amount"):
            tracker.record_fill("x", "BTC/USDT", "buy", -1.0, 80_000.0, 0.0)

    def test_price_cero_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="price"):
            tracker.record_fill("x", "BTC/USDT", "buy", 0.001, 0.0, 0.0)

    def test_price_negativo_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="price"):
            tracker.record_fill("x", "BTC/USDT", "buy", 0.001, -1.0, 0.0)

    def test_fee_negativa_lanza_value_error(self, tracker: PnLTracker) -> None:
        with pytest.raises(ValueError, match="fee"):
            tracker.record_fill("x", "BTC/USDT", "buy", 0.001, 80_000.0, -0.01)


# ---------------------------------------------------------------------------
# calculate_cycle_profit
# ---------------------------------------------------------------------------


class TestCalculateCycleProfit:
    def test_profit_positivo_tipico(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        # gross = (81_000 - 80_000) * 0.001 = 1.0
        # fees  = 0.08 + 0.081 = 0.161
        # net   = 1.0 - 0.161 = 0.839
        net = tracker.calculate_cycle_profit(buy_fill, sell_fill)
        assert abs(net - 0.839) < 1e-6

    def test_profit_negativo_cuando_fees_superan_spread(self, tracker: PnLTracker) -> None:
        buy = _make_fill("buy", price=80_000.0, amount=0.00001, fee=1.0)
        sell = _make_fill("sell", price=80_010.0, amount=0.00001, fee=1.0)
        # gross ≈ 0.0001  →  net = 0.0001 - 2.0 < 0
        net = tracker.calculate_cycle_profit(buy, sell)
        assert net < 0

    def test_usa_el_menor_amount_para_gross(self, tracker: PnLTracker) -> None:
        buy = _make_fill("buy", price=80_000.0, amount=0.002, fee=0.0)
        sell = _make_fill("sell", price=81_000.0, amount=0.001, fee=0.0)
        # min(0.002, 0.001) = 0.001  →  gross = 1_000 * 0.001 = 1.0
        net = tracker.calculate_cycle_profit(buy, sell)
        assert abs(net - 1.0) < 1e-6

    def test_sin_fees_profit_es_gross(self, tracker: PnLTracker) -> None:
        buy = _make_fill("buy", price=80_000.0, amount=0.001, fee=0.0)
        sell = _make_fill("sell", price=81_000.0, amount=0.001, fee=0.0)
        net = tracker.calculate_cycle_profit(buy, sell)
        assert abs(net - 1.0) < 1e-6

    def test_mismo_precio_solo_paga_fees(self, tracker: PnLTracker) -> None:
        buy = _make_fill("buy", price=80_000.0, fee=0.1)
        sell = _make_fill("sell", price=80_000.0, fee=0.1)
        net = tracker.calculate_cycle_profit(buy, sell)
        assert abs(net - (-0.2)) < 1e-6

    def test_retorna_float(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        net = tracker.calculate_cycle_profit(buy_fill, sell_fill)
        assert isinstance(net, float)


# ---------------------------------------------------------------------------
# record_cycle
# ---------------------------------------------------------------------------


class TestRecordCycle:
    def test_retorna_completed_cycle(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        cycle = tracker.record_cycle(buy_fill, sell_fill)

        assert isinstance(cycle, CompletedCycle)
        assert cycle.buy_fill == buy_fill
        assert cycle.sell_fill == sell_fill
        assert isinstance(cycle.completed_at, str)
        datetime.fromisoformat(cycle.completed_at)  # no debe lanzar

    def test_net_profit_coincide_con_calculate(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        expected = tracker.calculate_cycle_profit(buy_fill, sell_fill)
        cycle = tracker.record_cycle(buy_fill, sell_fill)
        assert abs(cycle.net_profit - expected) < 1e-8

    def test_persiste_json_en_disco(
        self, tmp_path: Path, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        path = tmp_path / "hist.json"
        t = PnLTracker(history_file=path)
        t.record_cycle(buy_fill, sell_fill)

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["buy_fill"]["order_id"] == buy_fill.order_id
        assert data[0]["sell_fill"]["order_id"] == sell_fill.order_id

    def test_acumula_multiples_ciclos(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)
        tracker.record_cycle(buy_fill, sell_fill)
        assert tracker.get_summary().total_cycles == 2

    def test_total_fees_es_suma_de_ambos_lados(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        cycle = tracker.record_cycle(buy_fill, sell_fill)
        assert abs(cycle.total_fees - (buy_fill.fee + sell_fill.fee)) < 1e-8


# ---------------------------------------------------------------------------
# get_daily_pnl
# ---------------------------------------------------------------------------


class TestGetDailyPnl:
    def test_cero_sin_ciclos(self, tracker: PnLTracker) -> None:
        assert tracker.get_daily_pnl() == 0.0

    def test_incluye_ciclos_de_hoy(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)
        daily = tracker.get_daily_pnl()
        expected = tracker.get_summary().net_profit
        assert abs(daily - expected) < 1e-8

    def test_excluye_ciclos_anteriores(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)
        # Reemplazar el ciclo para que parezca de hace 5 años
        old_cycle = tracker._cycles[0]
        tracker._cycles[0] = CompletedCycle(
            buy_fill=old_cycle.buy_fill,
            sell_fill=old_cycle.sell_fill,
            gross_profit=old_cycle.gross_profit,
            total_fees=old_cycle.total_fees,
            net_profit=old_cycle.net_profit,
            completed_at="2020-01-01T00:00:00+00:00",
        )
        assert tracker.get_daily_pnl() == 0.0

    def test_suma_multiples_ciclos_de_hoy(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)
        tracker.record_cycle(buy_fill, sell_fill)
        daily = tracker.get_daily_pnl()
        expected = tracker.get_summary().net_profit
        assert abs(daily - expected) < 1e-8

    def test_mezcla_hoy_y_anteriores(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)  # hoy
        tracker.record_cycle(buy_fill, sell_fill)  # moveremos a pasado
        old = tracker._cycles[1]
        tracker._cycles[1] = CompletedCycle(
            buy_fill=old.buy_fill,
            sell_fill=old.sell_fill,
            gross_profit=old.gross_profit,
            total_fees=old.total_fees,
            net_profit=old.net_profit,
            completed_at="2020-06-15T12:00:00+00:00",
        )
        daily = tracker.get_daily_pnl()
        one_cycle_net = tracker._cycles[0].net_profit
        assert abs(daily - one_cycle_net) < 1e-8


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_resumen_inicial_vacio(self, tracker: PnLTracker) -> None:
        s = tracker.get_summary()
        assert isinstance(s, PnLSummary)
        assert s.total_cycles == 0
        assert s.gross_profit == 0.0
        assert s.total_fees == 0.0
        assert s.net_profit == 0.0

    def test_resumen_un_ciclo(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        cycle = tracker.record_cycle(buy_fill, sell_fill)
        s = tracker.get_summary()

        assert s.total_cycles == 1
        assert abs(s.gross_profit - cycle.gross_profit) < 1e-8
        assert abs(s.total_fees - cycle.total_fees) < 1e-8
        assert abs(s.net_profit - cycle.net_profit) < 1e-8

    def test_resumen_multiples_ciclos_acumula(
        self, tracker: PnLTracker, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        tracker.record_cycle(buy_fill, sell_fill)
        tracker.record_cycle(buy_fill, sell_fill)
        s = tracker.get_summary()

        assert s.total_cycles == 2
        assert s.net_profit == round(s.gross_profit - s.total_fees, 8)

    def test_net_profit_puede_ser_negativo(self, tracker: PnLTracker) -> None:
        buy = _make_fill("buy", price=80_000.0, amount=0.00001, fee=5.0)
        sell = _make_fill("sell", price=80_010.0, amount=0.00001, fee=5.0)
        tracker.record_cycle(buy, sell)
        s = tracker.get_summary()
        assert s.net_profit < 0


# ---------------------------------------------------------------------------
# Persistencia (carga desde disco)
# ---------------------------------------------------------------------------


class TestPersistencia:
    def test_carga_historial_existente(
        self, tmp_path: Path, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        path = tmp_path / "hist.json"
        t1 = PnLTracker(history_file=path)
        t1.record_cycle(buy_fill, sell_fill)

        t2 = PnLTracker(history_file=path)
        s1 = t1.get_summary()
        s2 = t2.get_summary()

        assert s2.total_cycles == s1.total_cycles
        assert abs(s2.net_profit - s1.net_profit) < 1e-8

    def test_multiples_ciclos_se_recargan(
        self, tmp_path: Path, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        path = tmp_path / "hist.json"
        t1 = PnLTracker(history_file=path)
        t1.record_cycle(buy_fill, sell_fill)
        t1.record_cycle(buy_fill, sell_fill)

        t2 = PnLTracker(history_file=path)
        assert t2.get_summary().total_cycles == 2

    def test_crea_directorio_si_no_existe(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "nested" / "hist.json"
        PnLTracker(history_file=path)
        assert (tmp_path / "subdir" / "nested").is_dir()

    def test_historial_inexistente_inicia_vacio(self, tmp_path: Path) -> None:
        path = tmp_path / "no_existe.json"
        t = PnLTracker(history_file=path)
        assert t.get_summary().total_cycles == 0

    def test_json_es_lista_de_dicts(
        self, tmp_path: Path, buy_fill: FillRecord, sell_fill: FillRecord
    ) -> None:
        path = tmp_path / "hist.json"
        t = PnLTracker(history_file=path)
        t.record_cycle(buy_fill, sell_fill)

        raw = json.loads(path.read_text())
        assert isinstance(raw, list)
        assert "buy_fill" in raw[0]
        assert "sell_fill" in raw[0]
        assert "net_profit" in raw[0]
        assert "completed_at" in raw[0]


# ---------------------------------------------------------------------------
# OKXClient.fetch_order_trades
# ---------------------------------------------------------------------------


class TestFetchOrderTrades:
    def test_dry_run_retorna_dict_con_dry_run_true(self, okx_dry: OKXClient) -> None:
        result = okx_dry.fetch_order_trades("ord_001", "BTC/USDT")
        assert result["dry_run"] is True

    def test_dry_run_incluye_campos_requeridos(self, okx_dry: OKXClient) -> None:
        result = okx_dry.fetch_order_trades("ord_001", "BTC/USDT")
        assert result["order_id"] == "ord_001"
        assert result["symbol"] == "BTC/USDT"
        assert "price" in result
        assert "amount" in result
        assert "fee" in result
        assert "fee_currency" in result

    def test_dry_run_no_llama_al_exchange(self, okx_dry: OKXClient) -> None:
        okx_dry.fetch_order_trades("ord_001", "BTC/USDT")
        okx_dry._exchange.fetch_order_trades.assert_not_called()

    def test_live_agrega_fills_parciales(self, okx_live: OKXClient) -> None:
        okx_live._exchange.fetch_order_trades.return_value = [
            {"amount": 0.0005, "cost": 40.0, "fee": {"cost": 0.04, "currency": "USDT"}},
            {"amount": 0.0005, "cost": 40.0, "fee": {"cost": 0.04, "currency": "USDT"}},
        ]
        result = okx_live.fetch_order_trades("ord_001", "BTC/USDT")

        assert abs(result["amount"] - 0.001) < 1e-8
        assert abs(result["price"] - 80_000.0) < 1e-4
        assert abs(result["fee"] - 0.08) < 1e-8
        assert result["fee_currency"] == "USDT"

    def test_live_sin_trades_retorna_ceros(self, okx_live: OKXClient) -> None:
        okx_live._exchange.fetch_order_trades.return_value = []
        result = okx_live.fetch_order_trades("ord_empty", "BTC/USDT")

        assert result["amount"] == 0.0
        assert result["price"] == 0.0
        assert result["fee"] == 0.0

    def test_live_precio_promedio_ponderado(self, okx_live: OKXClient) -> None:
        # Dos fills a distintos precios: promedio ponderado debe ser correcto
        okx_live._exchange.fetch_order_trades.return_value = [
            {"amount": 0.001, "cost": 80.0, "fee": {"cost": 0.08, "currency": "USDT"}},
            {"amount": 0.001, "cost": 82.0, "fee": {"cost": 0.082, "currency": "USDT"}},
        ]
        result = okx_live.fetch_order_trades("ord_002", "BTC/USDT")
        # avg_price = (80 + 82) / 0.002 = 81_000
        assert abs(result["price"] - 81_000.0) < 1e-4
        assert abs(result["amount"] - 0.002) < 1e-8

    def test_live_preserva_order_id_y_symbol(self, okx_live: OKXClient) -> None:
        okx_live._exchange.fetch_order_trades.return_value = [
            {"amount": 0.001, "cost": 80.0, "fee": {"cost": 0.08, "currency": "USDT"}},
        ]
        result = okx_live.fetch_order_trades("ord_xyz", "ETH/USDT")
        assert result["order_id"] == "ord_xyz"
        assert result["symbol"] == "ETH/USDT"
