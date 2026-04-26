"""
Tests unitarios para GridState y state_file_for_symbol.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.strategy.grid_state import GridState, state_file_for_symbol


SAMPLE_LEVELS = [
    {"index": 0, "price": 60000.0, "status": "buy_open", "order_id": "ord_0", "side": "buy", "amount": 0.00003},
    {"index": 1, "price": 63750.0, "status": "buy_open", "order_id": "ord_1", "side": "buy", "amount": 0.00003},
    {"index": 2, "price": 67500.0, "status": "sell_open", "order_id": "ord_2", "side": "sell", "amount": 0.00003},
]

SAMPLE_GRID_CONFIG = {"price_min": 60000, "price_max": 75000, "num_levels": 5}


class TestGridStateInitialize:
    def test_initialize_crea_estado_correcto(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)

        state = grid_state.state
        assert state["symbol"] == "BTC/USDT"
        assert state["grid_config"] == SAMPLE_GRID_CONFIG
        assert len(state["levels"]) == 3
        assert state["total_profit_usdt"] == 0.0
        assert state["total_trades"] == 0
        assert "created_at" in state

    def test_initialize_persiste_en_disco(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        assert grid_state._path.exists()

    def test_initialize_json_valido(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        with grid_state._path.open() as f:
            data = json.load(f)
        assert data["symbol"] == "BTC/USDT"


class TestGridStateSaveLoad:
    def test_load_retorna_false_si_no_existe(self, grid_state):
        result = grid_state.load()
        assert result is False

    def test_save_load_roundtrip(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)

        nuevo_state = GridState(state_file=grid_state._path)
        loaded = nuevo_state.load()

        assert loaded is True
        assert nuevo_state.state["symbol"] == "BTC/USDT"
        assert len(nuevo_state.levels) == 3

    def test_save_escritura_atomica_no_corrompe(self, grid_state):
        """Verifica que save() produce JSON válido."""
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.save()

        with grid_state._path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)


class TestGridStateUpdateLevel:
    def test_update_level_cambia_status(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.update_level(0, "buy_filled")

        level = next(l for l in grid_state.levels if l["index"] == 0)
        assert level["status"] == "buy_filled"

    def test_update_level_cambia_order_id(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.update_level(1, "sell_open", order_id="nuevo_order_id")

        level = next(l for l in grid_state.levels if l["index"] == 1)
        assert level["order_id"] == "nuevo_order_id"
        assert level["status"] == "sell_open"

    def test_update_level_inexistente_no_lanza(self, grid_state, caplog):
        import logging
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        with caplog.at_level(logging.WARNING):
            grid_state.update_level(99, "buy_open")
        assert "update_level_no_encontrado" in caplog.text

    def test_update_level_sin_order_id_no_sobreescribe(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.update_level(0, "buy_filled")  # sin order_id

        level = next(l for l in grid_state.levels if l["index"] == 0)
        assert level["order_id"] == "ord_0"  # original sin cambios


class TestGridStateRecordProfit:
    def test_record_profit_acumula(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.record_profit(0.05)
        grid_state.record_profit(0.03)

        assert grid_state.total_profit == pytest.approx(0.08, abs=1e-8)

    def test_record_profit_incrementa_trades(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        grid_state.record_profit(0.05)
        grid_state.record_profit(0.05)

        assert grid_state.total_trades == 2


class TestGridStatePropiedades:
    def test_levels_retorna_lista(self, grid_state):
        grid_state.initialize("BTC/USDT", SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        assert isinstance(grid_state.levels, list)
        assert len(grid_state.levels) == 3

    def test_levels_vacio_antes_de_initialize(self, grid_state):
        assert grid_state.levels == []


class TestStateFileForSymbol:
    def test_btc_usdt(self):
        path = state_file_for_symbol("BTC/USDT")
        assert path == Path("data/grid_state_BTC_USDT.json")

    def test_eth_usdt(self):
        path = state_file_for_symbol("ETH/USDT")
        assert path == Path("data/grid_state_ETH_USDT.json")

    def test_sol_usdt(self):
        path = state_file_for_symbol("SOL/USDT")
        assert path == Path("data/grid_state_SOL_USDT.json")

    def test_slash_reemplazado_por_guion_bajo(self):
        path = state_file_for_symbol("BTC/USDT")
        assert "/" not in path.name

    def test_guion_reemplazado_por_guion_bajo(self):
        path = state_file_for_symbol("BTC-USDT")
        assert path == Path("data/grid_state_BTC_USDT.json")

    def test_retorna_path_object(self):
        path = state_file_for_symbol("BTC/USDT")
        assert isinstance(path, Path)

    def test_directorio_es_data(self):
        path = state_file_for_symbol("ETH/USDT")
        assert path.parent == Path("data")

    def test_extension_es_json(self):
        path = state_file_for_symbol("SOL/USDT")
        assert path.suffix == ".json"

    def test_simbolos_distintos_generan_paths_distintos(self):
        btc = state_file_for_symbol("BTC/USDT")
        eth = state_file_for_symbol("ETH/USDT")
        sol = state_file_for_symbol("SOL/USDT")
        assert btc != eth
        assert eth != sol
        assert btc != sol

    def test_grid_state_usa_path_correcto(self, tmp_path):
        """GridState construido con state_file_for_symbol usa el path esperado."""
        symbol = "ETH/USDT"
        expected_name = "grid_state_ETH_USDT.json"
        gs = GridState(state_file=tmp_path / expected_name)
        gs.initialize(symbol, SAMPLE_GRID_CONFIG, SAMPLE_LEVELS)
        assert gs._path.name == expected_name
        assert gs._path.exists()
