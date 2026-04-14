"""
Tests unitarios para GridCalculator.
"""
from __future__ import annotations

import pytest

from src.strategy.grid_calculator import GridCalculator, GridLevel, MIN_ORDER_BASE


class TestGridCalculatorValidacion:
    def test_price_min_cero_lanza_error(self):
        with pytest.raises(ValueError, match="price_min debe ser > 0"):
            GridCalculator(0, 90000, 5, 10.0, 2.5)

    def test_price_max_menor_que_min_lanza_error(self):
        with pytest.raises(ValueError, match="price_max .* debe ser > price_min"):
            GridCalculator(90000, 80000, 5, 10.0, 2.5)

    def test_price_max_igual_a_min_lanza_error(self):
        with pytest.raises(ValueError, match="price_max .* debe ser > price_min"):
            GridCalculator(80000, 80000, 5, 10.0, 2.5)

    def test_num_levels_uno_lanza_error(self):
        with pytest.raises(ValueError, match="num_levels debe ser >= 2"):
            GridCalculator(80000, 90000, 1, 10.0, 2.5)

    def test_capital_negativo_lanza_error(self):
        with pytest.raises(ValueError, match="total_capital_usdt debe ser > 0"):
            GridCalculator(80000, 90000, 5, -1.0, 2.5)

    def test_max_order_cero_lanza_error(self):
        with pytest.raises(ValueError, match="max_order_usdt debe ser > 0"):
            GridCalculator(80000, 90000, 5, 10.0, 0.0)


class TestGridCalculatorCalculo:
    def test_cinco_niveles_precios_correctos(self):
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate()

        assert len(levels) == 5
        assert levels[0].price == 80000.0
        assert levels[-1].price == 90000.0

        # Step = (90000-80000) / (5-1) = 2500
        for i in range(1, len(levels)):
            diff = round(levels[i].price - levels[i - 1].price, 2)
            assert diff == pytest.approx(2500.0)

    def test_retorna_lista_de_grid_levels(self):
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate()
        assert all(isinstance(l, GridLevel) for l in levels)

    def test_indices_son_secuenciales(self):
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate()
        assert [l.index for l in levels] == list(range(5))

    def test_capital_por_nivel_limitado_por_max_order(self):
        # total_capital/num_levels = 10/5 = 2.0, max_order = 1.5 → usa 1.5
        calc = GridCalculator(80000, 90000, 5, 10.0, 1.5)
        levels = calc.calculate()
        for level in levels:
            assert level.order_size_quote == pytest.approx(1.5, abs=0.0001)

    def test_capital_por_nivel_no_excede_total_dividido_niveles(self):
        # total_capital/num_levels = 10/5 = 2.0, max_order = 5.0 → usa 2.0
        calc = GridCalculator(80000, 90000, 5, 10.0, 5.0)
        levels = calc.calculate()
        for level in levels:
            assert level.order_size_quote == pytest.approx(2.0, abs=0.0001)

    def test_order_size_base_calculado_correctamente(self):
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate()
        # capital_per_level = min(10/5, 2.5) = 2.0
        # level 0 price = 80000 → size_base = 2.0 / 80000 = 0.000025
        expected_base = 2.0 / 80000.0
        assert levels[0].order_size_base == pytest.approx(expected_base, rel=1e-5)

    def test_order_size_base_mayor_que_minimo_exchange(self):
        calc = GridCalculator(60000, 75000, 5, 10.0, 2.5)
        levels = calc.calculate()
        for level in levels:
            assert level.order_size_base >= MIN_ORDER_BASE

    def test_dos_niveles_es_valido(self):
        calc = GridCalculator(80000, 90000, 2, 10.0, 5.0)
        levels = calc.calculate()
        assert len(levels) == 2
        assert levels[0].price == 80000.0
        assert levels[1].price == 90000.0

    def test_capital_insuficiente_lanza_error(self):
        # Con precio muy alto y capital muy bajo, order_size_base < MIN_ORDER_BASE
        with pytest.raises(ValueError, match="mínimo del exchange"):
            calc = GridCalculator(
                price_min=100_000_000,  # precio absurdo para forzar el error
                price_max=200_000_000,
                num_levels=5,
                total_capital_usdt=0.001,
                max_order_usdt=0.001,
            )
            calc.calculate()

    def test_spread_insuficiente_loggea_warning(self, caplog):
        import logging
        # fee_rate=0.5 (50%), step_pct será mucho menor
        calc = GridCalculator(80000, 80001, 2, 10.0, 5.0, fee_rate=0.5)
        with caplog.at_level(logging.WARNING):
            calc.calculate()
        assert "spread_insuficiente" in caplog.text
