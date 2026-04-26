"""
Tests unitarios para PairSelector.

Cobertura objetivo: >= 85%.
MarketAnalyzer se mockea; no se realizan llamadas reales al exchange.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.market_analyzer import RANGING, TRENDING_DOWN, TRENDING_UP, MarketRegime
from src.strategy.pair_selector import (
    PairSelection,
    PairSelector,
    _DEFAULT_HYSTERESIS_MARGIN,
    _DEFAULT_MIN_SCORE,
    _DEFAULT_PAIR_COOLDOWN_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regime(
    symbol: str = "BTC/USDT",
    regime: str = RANGING,
    atr_pct: float = 0.03,
    spread_pct: float = 0.001,
    score: float = 0.0,
) -> MarketRegime:
    """Genera un MarketRegime con valores controlados."""
    return MarketRegime(
        regime=regime,
        atr_pct=atr_pct,
        spread_pct=spread_pct,
        score=score,
        symbol=symbol,
    )


def _make_selector(
    tmp_path: Path,
    min_score: float = 0.0,
    vol_weight: float = 10.0,
    spread_weight: float = 100.0,
    hysteresis_margin: float = 0.20,
    pair_cooldown_seconds: float = 0.0,  # sin cooldown por defecto en tests
) -> PairSelector:
    """Crea un PairSelector con state_file en un directorio temporal."""
    return PairSelector(
        min_score_to_trade=min_score,
        vol_weight=vol_weight,
        spread_weight=spread_weight,
        hysteresis_margin=hysteresis_margin,
        pair_cooldown_seconds=pair_cooldown_seconds,
        state_file=tmp_path / "pair_state.json",
    )


# ---------------------------------------------------------------------------
# PairSelection dataclass
# ---------------------------------------------------------------------------


class TestPairSelectionDataclass:
    def test_campos_basicos(self):
        regime = _make_regime("ETH/USDT", RANGING)
        sel = PairSelection(symbol="ETH/USDT", regime=regime, score=0.5, reason="test")
        assert sel.symbol == "ETH/USDT"
        assert sel.score == 0.5
        assert sel.reason == "test"
        assert sel.regime is regime

    def test_acepta_cualquier_regime(self):
        for reg in [RANGING, TRENDING_UP, TRENDING_DOWN]:
            r = _make_regime(regime=reg)
            sel = PairSelection(symbol="X", regime=r, score=0.1, reason="ok")
            assert sel.regime.regime == reg


# ---------------------------------------------------------------------------
# _compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_ranging_sin_penalizacion(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        regime = _make_regime(regime=RANGING, atr_pct=0.03, spread_pct=0.001)
        # raw = 0.03*10 - 0.001*100 = 0.30 - 0.10 = 0.20; no penalty
        score = ps._compute_score(regime)
        assert score == pytest.approx(0.20, rel=1e-6)

    def test_trending_up_aplica_penalizacion_40pct(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        regime_ranging = _make_regime(regime=RANGING, atr_pct=0.03, spread_pct=0.001)
        regime_trending = _make_regime(regime=TRENDING_UP, atr_pct=0.03, spread_pct=0.001)
        score_r = ps._compute_score(regime_ranging)
        score_t = ps._compute_score(regime_trending)
        assert score_t == pytest.approx(score_r * 0.6, rel=1e-6)

    def test_trending_down_aplica_misma_penalizacion(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        regime_up = _make_regime(regime=TRENDING_UP, atr_pct=0.05, spread_pct=0.001)
        regime_dn = _make_regime(regime=TRENDING_DOWN, atr_pct=0.05, spread_pct=0.001)
        assert ps._compute_score(regime_up) == pytest.approx(ps._compute_score(regime_dn))

    def test_score_minimo_es_cero(self, tmp_path: Path):
        ps = _make_selector(tmp_path)
        regime = _make_regime(regime=RANGING, atr_pct=0.0, spread_pct=1.0)
        assert ps._compute_score(regime) == 0.0

    def test_score_maximo_es_uno(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        regime = _make_regime(regime=RANGING, atr_pct=100.0, spread_pct=0.0)
        assert ps._compute_score(regime) == 1.0

    def test_spread_alto_reduce_score(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        bajo = _make_regime(regime=RANGING, atr_pct=0.05, spread_pct=0.0001)
        alto = _make_regime(regime=RANGING, atr_pct=0.05, spread_pct=0.002)
        assert ps._compute_score(bajo) > ps._compute_score(alto)

    def test_score_siempre_entre_0_y_1(self, tmp_path: Path):
        ps = _make_selector(tmp_path, vol_weight=10.0, spread_weight=100.0)
        for atr in [0.0, 0.01, 0.05, 0.2]:
            for spread in [0.0, 0.001, 0.01]:
                for reg in [RANGING, TRENDING_UP, TRENDING_DOWN]:
                    regime = _make_regime(regime=reg, atr_pct=atr, spread_pct=spread)
                    s = ps._compute_score(regime)
                    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# select_best_pair — casos básicos
# ---------------------------------------------------------------------------


class TestSelectBestPairBasic:
    def test_sin_analisis_retorna_none(self, tmp_path: Path):
        ps = _make_selector(tmp_path)
        assert ps.select_best_pair([]) is None

    def test_retorna_pair_selection(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.001)
        result = ps.select_best_pair([regime])
        assert isinstance(result, PairSelection)

    def test_elige_el_par_con_mayor_score(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.02, spread_pct=0.001)
        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.06, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert result.symbol == "ETH/USDT"

    def test_score_inferior_a_minimo_retorna_none(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.9)
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.001, spread_pct=0.001)
        result = ps.select_best_pair([regime])
        assert result is None

    def test_primer_par_razón_es_primer_par(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("SOL/USDT", RANGING, atr_pct=0.05)
        result = ps.select_best_pair([regime])
        assert result is not None
        assert result.reason == "primer_par"

    def test_mismo_par_razon_es_mismo_par(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.05)
        ps.select_best_pair([regime])  # establece el par actual
        result = ps.select_best_pair([regime])
        assert result is not None
        assert result.reason == "mismo_par"

    def test_symbol_se_propaga_en_resultado(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("SOL/USDT", RANGING, atr_pct=0.04)
        result = ps.select_best_pair([regime])
        assert result is not None
        assert result.symbol == "SOL/USDT"

    def test_regime_se_propaga_en_resultado(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("ETH/USDT", TRENDING_UP, atr_pct=0.04)
        result = ps.select_best_pair([regime])
        assert result is not None
        assert result.regime.regime == TRENDING_UP


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_cambio_bloqueado_si_no_supera_margen(self, tmp_path: Path):
        """El nuevo par no supera el margen de hysteresis → se mantiene el actual."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.20,
            pair_cooldown_seconds=0.0,
        )
        # Establecer BTC con score ~0.30 (atr=0.05, spread=0.002 → raw=0.5-0.2=0.30)
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.002)
        ps.select_best_pair([btc])
        assert ps.current_symbol == "BTC/USDT"

        # ETH con score marginalmente mayor (no supera 0.30 * 1.20 = 0.36)
        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.053, spread_pct=0.002)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        # Debe mantener BTC
        assert result.symbol == "BTC/USDT"

    def test_cambio_autorizado_si_supera_margen(self, tmp_path: Path):
        """El nuevo par supera el margen de hysteresis → se permite el cambio."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.20,
            pair_cooldown_seconds=0.0,
        )
        # BTC con score bajo
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.02, spread_pct=0.001)
        ps.select_best_pair([btc])

        # ETH con score muy superior
        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.10, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert result.symbol == "ETH/USDT"
        assert result.reason == "mejor_score"

    def test_razon_cuando_hysteresis_bloquea(self, tmp_path: Path):
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.20,
            pair_cooldown_seconds=0.0,
        )
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.001)
        ps.select_best_pair([btc])

        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.055, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert "hysteresis" in result.reason


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_bloquea_cambio(self, tmp_path: Path):
        """No se cambia de par si no pasó el cooldown."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.0,       # sin hysteresis para aislar cooldown
            pair_cooldown_seconds=9999.0,  # cooldown muy largo
        )
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.03, spread_pct=0.001)
        ps.select_best_pair([btc])
        assert ps.current_symbol == "BTC/USDT"

        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.10, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert result.symbol == "BTC/USDT"  # bloqueado por cooldown

    def test_cooldown_permite_cambio_tras_expirar(self, tmp_path: Path):
        """Una vez expirado el cooldown, se permite el cambio."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.0,
            pair_cooldown_seconds=0.001,  # casi inmediato
        )
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.02, spread_pct=0.001)
        ps.select_best_pair([btc])

        time.sleep(0.002)  # esperar que expire el cooldown

        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.10, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert result.symbol == "ETH/USDT"

    def test_cooldown_razon(self, tmp_path: Path):
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            hysteresis_margin=0.0,
            pair_cooldown_seconds=9999.0,
        )
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.001)
        ps.select_best_pair([btc])

        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.10, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert "cooldown" in result.reason

    def test_cooldown_con_par_actual_sin_score_minimo_retorna_fallback(self, tmp_path: Path):
        """Cooldown activo, par actual no está en la lista → fallback."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            hysteresis_margin=0.0,
            pair_cooldown_seconds=9999.0,
        )
        # Primer par: BTC
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.001)
        ps.select_best_pair([btc])

        # Ahora solo hay ETH (BTC no está en la lista)
        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.10, spread_pct=0.001)
        result = ps.select_best_pair([eth])
        assert result is not None
        # Cooldown impide cambio a ETH, BTC no está disponible → fallback
        assert "cooldown" in result.reason


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


class TestPersistencia:
    def test_estado_se_guarda_en_json(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        regime = _make_regime("ETH/USDT", RANGING, atr_pct=0.05)
        ps.select_best_pair([regime])

        state_file = tmp_path / "pair_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["current_symbol"] == "ETH/USDT"
        assert "current_score" in data
        assert "last_switch_time" in data

    def test_estado_se_carga_al_inicializar(self, tmp_path: Path):
        state_file = tmp_path / "pair_state.json"
        state_data = {
            "current_symbol": "SOL/USDT",
            "current_score": 0.45,
            "last_switch_time": time.time() - 100,
        }
        state_file.write_text(json.dumps(state_data))

        ps = PairSelector(
            min_score_to_trade=0.0,
            state_file=state_file,
            pair_cooldown_seconds=0.0,
        )
        assert ps.current_symbol == "SOL/USDT"
        assert ps.current_score == pytest.approx(0.45)

    def test_estado_invalido_no_rompe_el_selector(self, tmp_path: Path):
        state_file = tmp_path / "pair_state.json"
        state_file.write_text("JSON ROTO {{{{")

        ps = PairSelector(min_score_to_trade=0.0, state_file=state_file)
        assert ps.current_symbol is None  # valores por defecto

    def test_directorio_se_crea_si_no_existe(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "pair_state.json"
        ps = PairSelector(min_score_to_trade=0.0, state_file=nested)
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.05)
        ps.select_best_pair([regime])
        assert nested.exists()

    def test_error_al_guardar_no_propaga_excepcion(self, tmp_path: Path):
        """_save_state captura cualquier excepción de I/O; no debe propagarla."""
        # Apuntamos el state_file a un directorio (no a un archivo), lo que hace
        # que write_text falle con IsADirectoryError en Linux o PermissionError en Windows.
        # En cualquier caso _save_state debe capturarlo silenciosamente.
        dir_path = tmp_path / "es_un_directorio"
        dir_path.mkdir()
        ps = PairSelector(min_score_to_trade=0.0, state_file=dir_path, pair_cooldown_seconds=0.0)
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.05)
        # No debe levantar excepción
        ps.select_best_pair([regime])


# ---------------------------------------------------------------------------
# Propiedades públicas
# ---------------------------------------------------------------------------


class TestPropiedades:
    def test_current_symbol_none_si_no_hay_seleccion(self, tmp_path: Path):
        ps = _make_selector(tmp_path)
        assert ps.current_symbol is None

    def test_current_score_cero_si_no_hay_seleccion(self, tmp_path: Path):
        ps = _make_selector(tmp_path)
        assert ps.current_score == 0.0

    def test_last_switch_time_cero_si_no_hay_seleccion(self, tmp_path: Path):
        ps = _make_selector(tmp_path)
        assert ps.last_switch_time == 0.0

    def test_propiedades_se_actualizan_tras_seleccion(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0)
        before = time.time()
        regime = _make_regime("BTC/USDT", RANGING, atr_pct=0.05)
        ps.select_best_pair([regime])
        after = time.time()

        assert ps.current_symbol == "BTC/USDT"
        assert ps.current_score > 0.0
        assert before <= ps.last_switch_time <= after


# ---------------------------------------------------------------------------
# Scores con multiples pares
# ---------------------------------------------------------------------------


class TestMultiplesPares:
    def test_ranging_supera_trending_con_misma_volatilidad(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0, vol_weight=10.0, spread_weight=100.0)
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.05, spread_pct=0.001)
        eth = _make_regime("ETH/USDT", TRENDING_UP, atr_pct=0.05, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth])
        assert result is not None
        assert result.symbol == "BTC/USDT"

    def test_tres_pares_elige_el_mejor(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0, vol_weight=10.0, spread_weight=100.0)
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.02, spread_pct=0.001)
        eth = _make_regime("ETH/USDT", RANGING, atr_pct=0.04, spread_pct=0.001)
        sol = _make_regime("SOL/USDT", RANGING, atr_pct=0.08, spread_pct=0.001)
        result = ps.select_best_pair([btc, eth, sol])
        assert result is not None
        assert result.symbol == "SOL/USDT"

    def test_trending_con_atr_muy_alto_puede_superar_ranging_bajo(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.0, vol_weight=10.0, spread_weight=100.0)
        # BTC RANGING pero ATR muy bajo
        btc = _make_regime("BTC/USDT", RANGING, atr_pct=0.001, spread_pct=0.001)
        # ETH TRENDING_UP pero ATR enorme → score*0.6 sigue siendo mayor
        eth = _make_regime("ETH/USDT", TRENDING_UP, atr_pct=0.20, spread_pct=0.001)

        btc_score = ps._compute_score(btc)
        eth_score = ps._compute_score(eth)
        result = ps.select_best_pair([btc, eth])
        if eth_score > btc_score:
            assert result is not None
            assert result.symbol == "ETH/USDT"
        else:
            assert result is not None
            assert result.symbol == "BTC/USDT"

    def test_todos_por_debajo_del_minimo_retorna_none(self, tmp_path: Path):
        ps = _make_selector(tmp_path, min_score=0.99)
        analyses = [
            _make_regime("BTC/USDT", RANGING, atr_pct=0.001, spread_pct=0.001),
            _make_regime("ETH/USDT", TRENDING_UP, atr_pct=0.001, spread_pct=0.001),
        ]
        assert ps.select_best_pair(analyses) is None


# ---------------------------------------------------------------------------
# Integración con MarketAnalyzer mockeado
# ---------------------------------------------------------------------------


class TestConMarketAnalyzerMockeado:
    def test_flujo_completo_primer_par(self, tmp_path: Path):
        """Simula el flujo como si viniera de MarketAnalyzer.analyze()."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = [
            _make_regime("BTC/USDT", RANGING, atr_pct=0.03, spread_pct=0.001),
            _make_regime("ETH/USDT", RANGING, atr_pct=0.05, spread_pct=0.001),
            _make_regime("SOL/USDT", TRENDING_DOWN, atr_pct=0.07, spread_pct=0.002),
        ]

        ps = _make_selector(tmp_path, min_score=0.0)
        analyses = [mock_analyzer.analyze(s) for s in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]]
        result = ps.select_best_pair(analyses)

        assert result is not None
        assert result.symbol == "ETH/USDT"  # mayor score RANGING vs SOL con penalty

    def test_cambio_correcto_tras_varios_ciclos(self, tmp_path: Path):
        """Simula varios ciclos de evaluación verificando el comportamiento de hysteresis."""
        ps = _make_selector(
            tmp_path,
            min_score=0.0,
            vol_weight=10.0,
            spread_weight=100.0,
            hysteresis_margin=0.20,
            pair_cooldown_seconds=0.0,
        )

        # Ciclo 1: BTC es el mejor
        btc_high = _make_regime("BTC/USDT", RANGING, atr_pct=0.06, spread_pct=0.001)
        eth_low = _make_regime("ETH/USDT", RANGING, atr_pct=0.03, spread_pct=0.001)
        r1 = ps.select_best_pair([btc_high, eth_low])
        assert r1 is not None
        assert r1.symbol == "BTC/USDT"

        # Ciclo 2: ETH sube mucho, supera hysteresis
        btc_low = _make_regime("BTC/USDT", RANGING, atr_pct=0.02, spread_pct=0.001)
        eth_high = _make_regime("ETH/USDT", RANGING, atr_pct=0.12, spread_pct=0.001)
        r2 = ps.select_best_pair([btc_low, eth_high])
        assert r2 is not None
        assert r2.symbol == "ETH/USDT"
        assert r2.reason == "mejor_score"
