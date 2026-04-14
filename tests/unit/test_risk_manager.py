"""
Tests unitarios para RiskManager.
"""
from __future__ import annotations

import pytest

from src.risk.risk_manager import RiskManager


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(
        risk_config={"max_daily_loss_usdt": 2.0, "max_open_orders": 5},
        loop_config={"max_consecutive_errors": 3},
    )


class TestCanPlaceOrder:
    def test_orden_dentro_del_limite_permitida(self, rm):
        ok, reason = rm.can_place_order(1.5, max_order_usdt=2.0)
        assert ok is True
        assert reason == "ok"

    def test_orden_exactamente_en_limite_permitida(self, rm):
        ok, reason = rm.can_place_order(2.0, max_order_usdt=2.0)
        assert ok is True

    def test_orden_supera_limite_rechazada(self, rm):
        ok, reason = rm.can_place_order(3.0, max_order_usdt=2.0)
        assert ok is False
        assert "orden_excede_maximo" in reason

    def test_circuit_open_bloquea_todo(self, rm):
        rm._circuit_open = True
        ok, reason = rm.can_place_order(0.5, max_order_usdt=2.0)
        assert ok is False
        assert "circuit_breaker" in reason


class TestCheckDailyLoss:
    def test_perdida_menor_al_limite_ok(self, rm):
        ok, reason = rm.check_daily_loss(1.0)
        assert ok is True
        assert reason == "ok"

    def test_perdida_igual_al_limite_activa_circuit(self, rm):
        ok, reason = rm.check_daily_loss(2.0)
        assert ok is False
        assert "perdida_diaria_excedida" in reason
        assert rm.circuit_open is True

    def test_perdida_mayor_al_limite_activa_circuit(self, rm):
        ok, _ = rm.check_daily_loss(5.0)
        assert ok is False
        assert rm.circuit_open is True

    def test_circuit_activado_persiste(self, rm):
        rm.check_daily_loss(9.99)
        ok, _ = rm.can_place_order(0.1, 2.0)
        assert ok is False


class TestCheckOpenOrders:
    def test_por_debajo_del_maximo_ok(self, rm):
        ok, reason = rm.check_open_orders(4)
        assert ok is True

    def test_exactamente_en_maximo_rechazado(self, rm):
        ok, reason = rm.check_open_orders(5)
        assert ok is False
        assert "max_ordenes_abiertas_alcanzado" in reason

    def test_cero_ordenes_ok(self, rm):
        ok, _ = rm.check_open_orders(0)
        assert ok is True


class TestCheckBalance:
    def test_balance_suficiente_ok(self, rm):
        ok, reason = rm.check_balance(available=5.0, required=3.0)
        assert ok is True

    def test_balance_exactamente_suficiente_ok(self, rm):
        ok, _ = rm.check_balance(available=3.0, required=3.0)
        assert ok is True

    def test_balance_insuficiente_rechazado(self, rm):
        ok, reason = rm.check_balance(available=1.0, required=3.0)
        assert ok is False
        assert "balance_insuficiente" in reason

    def test_balance_cero_rechazado(self, rm):
        ok, _ = rm.check_balance(available=0.0, required=0.01)
        assert ok is False


class TestCircuitBreaker:
    def test_primer_error_no_activa_circuit(self, rm):
        triggered = rm.register_error()
        assert triggered is False
        assert rm.circuit_open is False
        assert rm.consecutive_errors == 1

    def test_errores_consecutivos_activan_circuit(self, rm):
        rm.register_error()
        rm.register_error()
        triggered = rm.register_error()  # 3er error = max_consecutive_errors

        assert triggered is True
        assert rm.circuit_open is True

    def test_reset_errores_tras_ciclo_exitoso(self, rm):
        rm.register_error()
        rm.register_error()
        rm.reset_errors()

        assert rm.consecutive_errors == 0
        assert rm.circuit_open is False

    def test_reset_no_cierra_circuit_activado(self, rm):
        """El reset de errores no desactiva un circuit breaker ya activado."""
        rm.register_error()
        rm.register_error()
        rm.register_error()
        assert rm.circuit_open is True

        rm.reset_errors()
        # El circuit sigue abierto aunque los errores se resetearon
        assert rm.circuit_open is True

    def test_consecutivos_incremental(self, rm):
        for i in range(1, 3):
            rm.register_error()
            assert rm.consecutive_errors == i
