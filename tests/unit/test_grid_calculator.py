"""
Tests unitarios para GridCalculator.
"""
from __future__ import annotations

import pytest

from src.strategy.grid_calculator import (
    GridCalculator,
    GridLevel,
    MIN_ORDER_BASE,
    REGIME_RANGING,
    REGIME_TRENDING_DOWN,
    REGIME_TRENDING_UP,
)


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
        # score=1.0 para aislar el comportamiento de max_order (sin descuento por score)
        calc = GridCalculator(80000, 90000, 5, 10.0, 1.5)
        levels = calc.calculate(score=1.0)
        for level in levels:
            assert level.order_size_quote == pytest.approx(1.5, abs=0.0001)

    def test_capital_por_nivel_no_excede_total_dividido_niveles(self):
        # total_capital/num_levels = 10/5 = 2.0, max_order = 5.0 → usa 2.0
        # score=1.0 para aislar el comportamiento de capital/niveles (sin descuento)
        calc = GridCalculator(80000, 90000, 5, 10.0, 5.0)
        levels = calc.calculate(score=1.0)
        for level in levels:
            assert level.order_size_quote == pytest.approx(2.0, abs=0.0001)

    def test_order_size_base_calculado_correctamente(self):
        # score=1.0 para aislar el cálculo de order_size_base sin descuento de score
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=1.0)
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

    def test_side_vacio_por_defecto_en_ranging(self):
        """RANGING no asigna side: OrderManager decide por precio."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(regime=REGIME_RANGING)
        for level in levels:
            assert level.side == ""

    def test_calculate_sin_regime_usa_ranging(self):
        """Sin parámetro regime se comporta igual que RANGING."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate()
        for level in levels:
            assert level.side == ""


class TestGridCalculatorRegimenes:
    """Tests de distribución asimétrica según régimen de mercado."""

    def test_trending_up_70_pct_buy_6_niveles(self):
        """Con 6 niveles y TRENDING_UP: 4 BUY, 2 SELL (round(0.7*6)=4)."""
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_UP)

        buy_levels = [l for l in levels if l.side == "buy"]
        sell_levels = [l for l in levels if l.side == "sell"]

        assert len(buy_levels) == 4
        assert len(sell_levels) == 2

    def test_trending_up_70_pct_buy_5_niveles(self):
        """Con 5 niveles y TRENDING_UP: round(0.7*5)=4 BUY, 1 SELL."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_UP)

        buy_count = sum(1 for l in levels if l.side == "buy")
        sell_count = sum(1 for l in levels if l.side == "sell")

        assert buy_count == round(0.7 * 5)
        assert sell_count == 5 - buy_count

    def test_trending_up_buy_en_niveles_inferiores(self):
        """TRENDING_UP: los primeros niveles (precio más bajo) son BUY."""
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_UP)

        buy_prices = [l.price for l in levels if l.side == "buy"]
        sell_prices = [l.price for l in levels if l.side == "sell"]

        assert max(buy_prices) < min(sell_prices)

    def test_trending_down_70_pct_sell_6_niveles(self):
        """Con 6 niveles y TRENDING_DOWN: 2 BUY, 4 SELL (round(0.7*6)=4 SELL)."""
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_DOWN)

        buy_count = sum(1 for l in levels if l.side == "buy")
        sell_count = sum(1 for l in levels if l.side == "sell")

        assert sell_count == 4
        assert buy_count == 2

    def test_trending_down_70_pct_sell_5_niveles(self):
        """Con 5 niveles y TRENDING_DOWN: round(0.7*5)=4 SELL, 1 BUY."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_DOWN)

        sell_count = sum(1 for l in levels if l.side == "sell")
        buy_count = sum(1 for l in levels if l.side == "buy")

        # round(0.7 * 5) = round(3.5) = 4 (Python banker's rounding)
        expected_sell = round(0.7 * 5)
        assert sell_count == expected_sell
        assert buy_count == 5 - expected_sell

    def test_trending_down_sell_en_niveles_superiores(self):
        """TRENDING_DOWN: los últimos niveles (precio más alto) son SELL."""
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_DOWN)

        buy_prices = [l.price for l in levels if l.side == "buy"]
        sell_prices = [l.price for l in levels if l.side == "sell"]

        assert max(buy_prices) < min(sell_prices)

    def test_regime_invalido_lanza_error(self):
        """Un régimen desconocido lanza ValueError."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        with pytest.raises(ValueError, match="regime inválido"):
            calc.calculate(regime="SIDEWAYS")

    def test_trending_up_todos_los_niveles_tienen_side(self):
        """En TRENDING_UP todos los GridLevel tienen side asignado."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_UP)
        assert all(l.side in ("buy", "sell") for l in levels)

    def test_trending_down_todos_los_niveles_tienen_side(self):
        """En TRENDING_DOWN todos los GridLevel tienen side asignado."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(regime=REGIME_TRENDING_DOWN)
        assert all(l.side in ("buy", "sell") for l in levels)

    def test_trending_up_minimo_un_sell(self):
        """Siempre al menos 1 SELL en TRENDING_UP, aunque num_levels sea muy pequeño."""
        calc = GridCalculator(80000, 90000, 2, 10.0, 5.0)
        levels = calc.calculate(regime=REGIME_TRENDING_UP)
        assert sum(1 for l in levels if l.side == "sell") >= 1
        assert sum(1 for l in levels if l.side == "buy") >= 1

    def test_trending_down_minimo_un_buy(self):
        """Siempre al menos 1 BUY en TRENDING_DOWN, aunque num_levels sea muy pequeño."""
        calc = GridCalculator(80000, 90000, 2, 10.0, 5.0)
        levels = calc.calculate(regime=REGIME_TRENDING_DOWN)
        assert sum(1 for l in levels if l.side == "buy") >= 1
        assert sum(1 for l in levels if l.side == "sell") >= 1

    def test_precios_no_cambian_con_regime(self):
        """El regime solo afecta el campo side, no los precios de los niveles."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels_ranging = calc.calculate(regime=REGIME_RANGING)
        levels_up = calc.calculate(regime=REGIME_TRENDING_UP)
        levels_down = calc.calculate(regime=REGIME_TRENDING_DOWN)

        prices_ranging = [l.price for l in levels_ranging]
        prices_up = [l.price for l in levels_up]
        prices_down = [l.price for l in levels_down]

        assert prices_ranging == prices_up == prices_down

    def test_trending_up_loggea_grid_asimetrica(self, caplog):
        """TRENDING_UP loggea la distribución asimétrica."""
        import logging
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        with caplog.at_level(logging.INFO):
            calc.calculate(regime=REGIME_TRENDING_UP)
        assert "grid_asimetrica" in caplog.text
        assert "TRENDING_UP" in caplog.text

    def test_trending_down_loggea_grid_asimetrica(self, caplog):
        """TRENDING_DOWN loggea la distribución asimétrica."""
        import logging
        calc = GridCalculator(80000, 90000, 6, 12.0, 2.5)
        with caplog.at_level(logging.INFO):
            calc.calculate(regime=REGIME_TRENDING_DOWN)
        assert "grid_asimetrica" in caplog.text
        assert "TRENDING_DOWN" in caplog.text


class TestGridCalculatorPositionSizing:
    """Tests de position sizing dinámico según score de mercado."""

    def test_score_alto_usa_100_pct_capital(self):
        """score >= 0.7 → multiplicador 1.0 (100% del capital por nivel)."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=0.7)
        expected = min(10.0 / 5, 2.5) * 1.0  # 2.0
        for level in levels:
            assert level.order_size_quote == pytest.approx(expected, abs=0.0001)

    def test_score_alto_exacto_borde_superior(self):
        """score = 1.0 → 100% del capital."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=1.0)
        for level in levels:
            assert level.order_size_quote == pytest.approx(2.0, abs=0.0001)

    def test_score_medio_usa_75_pct_capital(self):
        """0.4 <= score < 0.7 → multiplicador 0.75."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=0.5)
        expected = min(10.0 / 5, 2.5) * 0.75  # 1.5
        for level in levels:
            assert level.order_size_quote == pytest.approx(expected, abs=0.0001)

    def test_score_medio_borde_inferior(self):
        """score = 0.4 → sigue siendo 75%."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=0.4)
        expected = min(10.0 / 5, 2.5) * 0.75
        for level in levels:
            assert level.order_size_quote == pytest.approx(expected, abs=0.0001)

    def test_score_bajo_usa_50_pct_capital(self):
        """score < 0.4 → multiplicador 0.5."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=0.2)
        expected = min(10.0 / 5, 2.5) * 0.5  # 1.0
        for level in levels:
            assert level.order_size_quote == pytest.approx(expected, abs=0.0001)

    def test_score_cero_usa_50_pct_capital(self):
        """score = 0.0 → mínimo: 50% del capital."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=0.0)
        expected = min(10.0 / 5, 2.5) * 0.5
        for level in levels:
            assert level.order_size_quote == pytest.approx(expected, abs=0.0001)

    def test_score_multiplier_no_baja_de_min_order_usdt(self):
        """El capital efectivo nunca baja de min_order_usdt aunque el score sea bajo."""
        # capital_per_level = min(10/5, 2.5) = 2.0; * 0.5 = 1.0
        # min_order_usdt = 1.5 → debe usar 1.5
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, min_order_usdt=1.5)
        levels = calc.calculate(score=0.0)
        for level in levels:
            assert level.order_size_quote == pytest.approx(1.5, abs=0.0001)

    def test_score_multiplier_respeta_min_order_usdt_con_score_medio(self):
        """min_order_usdt = 1.8 supera el capital * 0.75 = 1.5 → usa 1.8."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, min_order_usdt=1.8)
        levels = calc.calculate(score=0.5)
        for level in levels:
            assert level.order_size_quote == pytest.approx(1.8, abs=0.0001)

    def test_score_multiplier_no_aplica_floor_cuando_capital_mayor(self):
        """Cuando el capital * multiplier > min_order_usdt, usa el capital calculado."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, min_order_usdt=0.5)
        levels = calc.calculate(score=0.7)  # 2.0 * 1.0 = 2.0 > 0.5
        for level in levels:
            assert level.order_size_quote == pytest.approx(2.0, abs=0.0001)

    def test_score_loggea_multiplier_en_grid_calculada(self, caplog):
        """calculate() loggea el score, multiplier y capital efectivo."""
        import logging
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        with caplog.at_level(logging.INFO):
            calc.calculate(score=0.3)
        assert "grid_calculada" in caplog.text
        assert "multiplier" in caplog.text

    def test_score_multiplier_estatico(self):
        """_score_multiplier retorna los valores correctos en los tres rangos."""
        assert GridCalculator._score_multiplier(1.0) == 1.0
        assert GridCalculator._score_multiplier(0.7) == 1.0
        assert GridCalculator._score_multiplier(0.69) == 0.75
        assert GridCalculator._score_multiplier(0.4) == 0.75
        assert GridCalculator._score_multiplier(0.39) == 0.5
        assert GridCalculator._score_multiplier(0.0) == 0.5


class TestGridCalculatorATRAdjustment:
    """Tests de auto-ajuste de rango según ATR."""

    def test_atr_sin_range_width_pct_no_modifica_rango(self):
        """Si range_width_pct es None, atr_pct no altera los precios."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5)
        levels = calc.calculate(score=1.0, atr_pct=0.10)  # ATR 10%: muy grande
        assert levels[0].price == pytest.approx(80000.0, abs=1.0)
        assert levels[-1].price == pytest.approx(90000.0, abs=1.0)

    def test_atr_pequeno_no_expande_rango(self):
        """ATR pequeño: 2*atr < rango_actual → no se expande."""
        # rango actual = 10000 / midpoint 85000 = ~11.76%
        # range_width_pct = 5.0; atr_pct = 0.03 → 2*atr = 6%
        # required = max(5.0, 6.0) = 6.0 < 11.76% → NO expande
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        levels = calc.calculate(score=1.0, atr_pct=0.03)
        assert levels[0].price == pytest.approx(80000.0, abs=1.0)
        assert levels[-1].price == pytest.approx(90000.0, abs=1.0)

    def test_atr_grande_expande_rango(self):
        """ATR grande (2*atr > range_width_pct): el rango SE expande."""
        # range configurado: 80000-90000 → 11.76% de 85000
        # range_width_pct = 5.0; atr_pct = 0.08 → 2*atr = 16% > 5% → expande
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        levels = calc.calculate(score=1.0, atr_pct=0.08)
        # midpoint = 85000; required = 16%; half = 85000 * 0.16/2 = 6800
        # new_min = 78200, new_max = 91800
        assert levels[0].price < 80000.0
        assert levels[-1].price > 90000.0

    def test_atr_expande_simetricamente_alrededor_del_midpoint(self):
        """El rango expandido es simétrico alrededor del midpoint original."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        levels = calc.calculate(score=1.0, atr_pct=0.08)
        midpoint = (levels[0].price + levels[-1].price) / 2
        assert midpoint == pytest.approx(85000.0, abs=5.0)

    def test_atr_ajuste_loggea_mensaje(self, caplog):
        """El ajuste por ATR se loggea cuando ocurre una expansión."""
        import logging
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        with caplog.at_level(logging.INFO):
            calc.calculate(score=1.0, atr_pct=0.08)
        assert "rango_ajustado_por_atr" in caplog.text

    def test_atr_sin_expansion_no_loggea_ajuste(self, caplog):
        """Si el rango ya es suficiente, no se loggea el ajuste."""
        import logging
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        with caplog.at_level(logging.INFO):
            calc.calculate(score=1.0, atr_pct=0.01)  # ATR 1%: muy pequeño
        assert "rango_ajustado_por_atr" not in caplog.text

    def test_atr_none_no_modifica_rango(self):
        """atr_pct=None → el rango permanece igual."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        levels = calc.calculate(score=1.0, atr_pct=None)
        assert levels[0].price == pytest.approx(80000.0, abs=1.0)
        assert levels[-1].price == pytest.approx(90000.0, abs=1.0)

    def test_atr_expansion_mantiene_num_levels(self):
        """El número de niveles no cambia al expandir el rango."""
        calc = GridCalculator(80000, 90000, 5, 10.0, 2.5, range_width_pct=5.0)
        levels = calc.calculate(score=1.0, atr_pct=0.08)
        assert len(levels) == 5
