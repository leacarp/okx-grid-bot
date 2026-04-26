"""
Tests unitarios para MarketAnalyzer y OKXClient.fetch_ohlcv.

Cobertura objetivo: >= 85%
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.connectors.okx_client import OKXClient
from src.strategy.market_analyzer import (
    OHLCV,
    RANGING,
    TRENDING_DOWN,
    TRENDING_UP,
    MarketAnalyzer,
    MarketRegime,
    _MIN_CANDLES_REQUIRED,
)


# ---------------------------------------------------------------------------
# Helpers y fixtures
# ---------------------------------------------------------------------------


def _make_candle(
    close: float,
    high_offset: float = 0.01,
    low_offset: float = 0.02,
    ts: int = 0,
) -> OHLCV:
    """Genera una vela OHLCV a partir del precio de cierre."""
    high = close * (1 + high_offset)
    low = close * (1 - low_offset)
    return [ts, close * 0.999, high, low, close, 1000.0]


def _make_candles(
    prices: list[float],
    high_offset: float = 0.01,
    low_offset: float = 0.02,
) -> list[OHLCV]:
    """Genera una lista de velas OHLCV a partir de precios de cierre."""
    return [
        _make_candle(p, high_offset=high_offset, low_offset=low_offset, ts=i * 3_600_000)
        for i, p in enumerate(prices)
    ]


def _flat_prices(n: int = 25, base: float = 100.0) -> list[float]:
    return [base] * n


def _rising_prices(n: int = 25, base: float = 100.0, step: float = 1.0) -> list[float]:
    return [base + i * step for i in range(n)]


def _falling_prices(n: int = 25, base: float = 100.0, step: float = 1.0) -> list[float]:
    return [base - i * step for i in range(n)]


def _make_ticker(bid: float = 99.5, ask: float = 100.5, last: float = 100.0) -> dict:
    return {"bid": bid, "ask": ask, "last": last}


@pytest.fixture()
def mock_client() -> MagicMock:
    """OKXClient mockeado para no llamar al exchange."""
    return MagicMock(spec=OKXClient)


@pytest.fixture()
def analyzer(mock_client: MagicMock) -> MarketAnalyzer:
    """MarketAnalyzer con parámetros por defecto y client mockeado."""
    return MarketAnalyzer(
        client=mock_client,
        timeframe="1h",
        candle_limit=24,
        atr_period=14,
        sma_period=20,
        slope_threshold=0.0005,
        bb_threshold=0.04,
        vol_weight=10.0,
        spread_weight=100.0,
    )


# ---------------------------------------------------------------------------
# OKXClient.fetch_ohlcv
# ---------------------------------------------------------------------------


class TestFetchOhlcvOKXClient:
    def test_delega_al_exchange(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = [[1, 100, 101, 99, 100, 500]]

        client = OKXClient(
            api_key="k",
            api_secret="s",
            passphrase="p",
            dry_run=True,
            exchange=mock_exchange,
        )
        result = client.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=1)

        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", timeframe="1h", limit=1)
        assert result == [[1, 100, 101, 99, 100, 500]]

    def test_usa_parametros_por_defecto(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        client = OKXClient(
            api_key="k",
            api_secret="s",
            passphrase="p",
            dry_run=True,
            exchange=mock_exchange,
        )
        client.fetch_ohlcv("ETH/USDT")

        mock_exchange.fetch_ohlcv.assert_called_once_with("ETH/USDT", timeframe="1h", limit=24)

    def test_retorna_lista_vacia_si_el_exchange_retorna_vacia(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        client = OKXClient(
            api_key="k",
            api_secret="s",
            passphrase="p",
            dry_run=True,
            exchange=mock_exchange,
        )
        assert client.fetch_ohlcv("BTC/USDT") == []


# ---------------------------------------------------------------------------
# Cálculo de ATR
# ---------------------------------------------------------------------------


class TestCalcAtr:
    def test_datos_insuficientes_retorna_cero(self):
        """Con menos de period+1 velas debe retornar 0."""
        candles = _make_candles([100.0] * 5)
        assert MarketAnalyzer._calc_atr(candles, period=14) == 0.0

    def test_atr_con_precios_uniformes(self):
        """Precios constantes: ATR = rango de cada vela (high - low)."""
        price = 100.0
        candles = _make_candles([price] * 20, high_offset=0.01, low_offset=0.01)
        atr = MarketAnalyzer._calc_atr(candles, period=14)
        # high = 101, low = 99 → TR por vela = 2 (no hay gaps)
        assert atr == pytest.approx(2.0, rel=1e-3)

    def test_atr_refleja_volatilidad(self):
        """Precios con mayor variación dan ATR mayor."""
        candles_bajo = _make_candles([100.0] * 20, high_offset=0.005, low_offset=0.005)
        candles_alto = _make_candles([100.0] * 20, high_offset=0.05, low_offset=0.05)
        atr_bajo = MarketAnalyzer._calc_atr(candles_bajo, period=14)
        atr_alto = MarketAnalyzer._calc_atr(candles_alto, period=14)
        assert atr_alto > atr_bajo

    def test_atr_usa_solo_las_ultimas_period_velas(self):
        """Cambio brusco al final debe modificar el ATR respecto a datos planos."""
        prices_plano = [100.0] * 30
        candles_plano = _make_candles(prices_plano, high_offset=0.01, low_offset=0.01)
        atr_base = MarketAnalyzer._calc_atr(candles_plano, period=14)

        # Reemplazar las últimas 14 velas con alta volatilidad
        candles_volat = _make_candles(prices_plano[:-14], high_offset=0.01, low_offset=0.01)
        candles_volat += _make_candles([100.0] * 14, high_offset=0.10, low_offset=0.10)
        atr_volat = MarketAnalyzer._calc_atr(candles_volat, period=14)

        assert atr_volat > atr_base

    def test_atr_con_gaps_entre_velas(self):
        """ATR debe capturar gaps (|high - prev_close| y |low - prev_close|)."""
        # Vela 0: close=100; vela 1: open gap up, high=120, low=118, close=119
        candles = [
            [0, 99, 101, 98, 100, 500],
            [1, 115, 120, 118, 119, 500],  # gap enorme
        ]
        atr = MarketAnalyzer._calc_atr(candles, period=1)
        # TR = max(120-118, |120-100|, |118-100|) = max(2, 20, 18) = 20
        assert atr == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Cálculo de pendiente de SMA
# ---------------------------------------------------------------------------


class TestCalcSmaSlope:
    def test_datos_insuficientes_retorna_cero(self):
        candles = _make_candles([100.0] * 10)
        assert MarketAnalyzer._calc_sma_slope(candles, period=20) == 0.0

    def test_precios_planos_da_pendiente_cero(self):
        candles = _make_candles(_flat_prices(25, 100.0))
        slope = MarketAnalyzer._calc_sma_slope(candles, period=20)
        assert slope == pytest.approx(0.0, abs=1e-9)

    def test_precios_crecientes_dan_pendiente_positiva(self):
        candles = _make_candles(_rising_prices(25, 100.0, step=1.0))
        slope = MarketAnalyzer._calc_sma_slope(candles, period=20)
        assert slope > 0.0

    def test_precios_decrecientes_dan_pendiente_negativa(self):
        candles = _make_candles(_falling_prices(25, 100.0, step=1.0))
        slope = MarketAnalyzer._calc_sma_slope(candles, period=20)
        assert slope < 0.0

    def test_tendencia_mas_fuerte_da_pendiente_mayor(self):
        candles_leve = _make_candles(_rising_prices(25, 100.0, step=0.5))
        candles_fuerte = _make_candles(_rising_prices(25, 100.0, step=2.0))
        slope_leve = MarketAnalyzer._calc_sma_slope(candles_leve, period=20)
        slope_fuerte = MarketAnalyzer._calc_sma_slope(candles_fuerte, period=20)
        assert slope_fuerte > slope_leve

    def test_first_sma_cero_retorna_cero(self):
        """Si la primera SMA es cero no se divide por cero."""
        candles = _make_candles([0.0] * 25)
        slope = MarketAnalyzer._calc_sma_slope(candles, period=20)
        assert slope == 0.0

    def test_exactamente_period_mas_uno_funciona(self):
        """El mínimo viable es period+1 velas."""
        candles = _make_candles(_rising_prices(21, 100.0, step=1.0))
        slope = MarketAnalyzer._calc_sma_slope(candles, period=20)
        assert slope > 0.0


# ---------------------------------------------------------------------------
# Cálculo de Bollinger Bandwidth
# ---------------------------------------------------------------------------


class TestCalcBollingerBandwidth:
    def test_datos_insuficientes_retorna_cero(self):
        candles = _make_candles([100.0] * 5)
        assert MarketAnalyzer._calc_bollinger_bandwidth(candles, period=20) == 0.0

    def test_precios_iguales_da_bandwidth_cero(self):
        candles = _make_candles(_flat_prices(25, 100.0), high_offset=0.0, low_offset=0.0)
        # Todos los closes iguales → std = 0 → bandwidth = 0
        bw = MarketAnalyzer._calc_bollinger_bandwidth(candles, period=20)
        assert bw == pytest.approx(0.0, abs=1e-9)

    def test_alta_variacion_da_bandwidth_mayor(self):
        closes_estable = _flat_prices(25, 100.0)
        closes_volat = [100.0 + (i % 2) * 5 for i in range(25)]  # oscila entre 100 y 105

        bw_estable = MarketAnalyzer._calc_bollinger_bandwidth(
            _make_candles(closes_estable), period=20
        )
        bw_volat = MarketAnalyzer._calc_bollinger_bandwidth(
            _make_candles(closes_volat), period=20
        )
        assert bw_volat > bw_estable

    def test_usa_solo_las_ultimas_period_velas(self):
        """El cálculo solo considera las últimas `period` velas."""
        # Primeras velas: alta variabilidad (no deben influir)
        candles_inicio = _make_candles([100.0 + (i % 2) * 10 for i in range(10)])
        candles_final = _make_candles(_flat_prices(25, 100.0), high_offset=0.0, low_offset=0.0)
        candles = candles_inicio + candles_final

        bw = MarketAnalyzer._calc_bollinger_bandwidth(candles, period=20)
        assert bw == pytest.approx(0.0, abs=1e-9)

    def test_sma_cero_retorna_cero(self):
        candles = _make_candles([0.0] * 25, high_offset=0.0, low_offset=0.0)
        bw = MarketAnalyzer._calc_bollinger_bandwidth(candles, period=20)
        assert bw == 0.0


# ---------------------------------------------------------------------------
# Clasificación de régimen
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    def setup_method(self):
        self.analyzer = MarketAnalyzer(
            client=MagicMock(),
            slope_threshold=0.0005,
            bb_threshold=0.04,
        )

    def test_pendiente_positiva_fuerte_es_trending_up(self):
        regime = self.analyzer._classify_regime(slope=0.001, bb_bandwidth=0.08)
        assert regime == TRENDING_UP

    def test_pendiente_negativa_fuerte_es_trending_down(self):
        regime = self.analyzer._classify_regime(slope=-0.001, bb_bandwidth=0.08)
        assert regime == TRENDING_DOWN

    def test_pendiente_plana_es_ranging(self):
        regime = self.analyzer._classify_regime(slope=0.0001, bb_bandwidth=0.02)
        assert regime == RANGING

    def test_pendiente_exactamente_en_threshold_es_ranging(self):
        regime = self.analyzer._classify_regime(slope=0.0005, bb_bandwidth=0.02)
        # slope == threshold (no >) → RANGING
        assert regime == RANGING

    def test_pendiente_justo_encima_del_threshold_es_trending_up(self):
        regime = self.analyzer._classify_regime(slope=0.00051, bb_bandwidth=0.02)
        assert regime == TRENDING_UP

    def test_pendiente_cero_es_ranging(self):
        regime = self.analyzer._classify_regime(slope=0.0, bb_bandwidth=0.01)
        assert regime == RANGING


# ---------------------------------------------------------------------------
# Cálculo de score
# ---------------------------------------------------------------------------


class TestCalcScore:
    def setup_method(self):
        self.analyzer = MarketAnalyzer(
            client=MagicMock(),
            vol_weight=10.0,
            spread_weight=100.0,
        )

    def test_ranging_con_alta_volatilidad_da_score_alto(self):
        score = self.analyzer._calc_score(atr_pct=0.05, spread_pct=0.0001, regime=RANGING)
        # raw = 0.05 * 10 * 1.0 - 0.0001 * 100 = 0.5 - 0.01 = 0.49
        assert score == pytest.approx(0.49, rel=1e-3)

    def test_trending_penaliza_el_score(self):
        score_ranging = self.analyzer._calc_score(atr_pct=0.05, spread_pct=0.0001, regime=RANGING)
        score_trending = self.analyzer._calc_score(
            atr_pct=0.05, spread_pct=0.0001, regime=TRENDING_UP
        )
        assert score_ranging > score_trending

    def test_trending_down_penaliza_igual_que_trending_up(self):
        score_up = self.analyzer._calc_score(atr_pct=0.05, spread_pct=0.0001, regime=TRENDING_UP)
        score_down = self.analyzer._calc_score(
            atr_pct=0.05, spread_pct=0.0001, regime=TRENDING_DOWN
        )
        assert score_up == pytest.approx(score_down)

    def test_spread_alto_reduce_score(self):
        score_bajo = self.analyzer._calc_score(atr_pct=0.03, spread_pct=0.0001, regime=RANGING)
        score_alto = self.analyzer._calc_score(atr_pct=0.03, spread_pct=0.002, regime=RANGING)
        assert score_bajo > score_alto

    def test_score_minimo_es_cero(self):
        score = self.analyzer._calc_score(atr_pct=0.0, spread_pct=0.5, regime=RANGING)
        assert score == 0.0

    def test_score_maximo_es_uno(self):
        score = self.analyzer._calc_score(atr_pct=100.0, spread_pct=0.0, regime=RANGING)
        assert score == 1.0

    def test_score_siempre_entre_cero_y_uno(self):
        for atr in [0.0, 0.01, 0.05, 0.2]:
            for spread in [0.0, 0.001, 0.01]:
                for regime in [RANGING, TRENDING_UP, TRENDING_DOWN]:
                    score = self.analyzer._calc_score(atr, spread, regime)
                    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Cálculo de spread
# ---------------------------------------------------------------------------


class TestCalcSpreadPct:
    def test_spread_normal(self):
        ticker = {"bid": 99.0, "ask": 101.0, "last": 100.0}
        spread = MarketAnalyzer._calc_spread_pct(ticker)
        # mid = 100, spread = 2/100 = 0.02
        assert spread == pytest.approx(0.02, rel=1e-6)

    def test_bid_cero_retorna_cero(self):
        ticker = {"bid": 0.0, "ask": 101.0, "last": 100.0}
        assert MarketAnalyzer._calc_spread_pct(ticker) == 0.0

    def test_ask_cero_retorna_cero(self):
        ticker = {"bid": 99.0, "ask": 0.0, "last": 100.0}
        assert MarketAnalyzer._calc_spread_pct(ticker) == 0.0

    def test_bid_ausente_retorna_cero(self):
        ticker = {"last": 100.0}
        assert MarketAnalyzer._calc_spread_pct(ticker) == 0.0

    def test_bid_none_retorna_cero(self):
        ticker = {"bid": None, "ask": 101.0, "last": 100.0}
        assert MarketAnalyzer._calc_spread_pct(ticker) == 0.0

    def test_spread_simetrico(self):
        ticker = {"bid": 100.0, "ask": 100.0, "last": 100.0}
        assert MarketAnalyzer._calc_spread_pct(ticker) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fetch_candles
# ---------------------------------------------------------------------------


class TestFetchCandles:
    def test_delega_a_client_fetch_ohlcv(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        expected = _make_candles([100.0] * 24)
        mock_client.fetch_ohlcv.return_value = expected

        result = analyzer.fetch_candles("BTC/USDT")

        mock_client.fetch_ohlcv.assert_called_once_with("BTC/USDT", timeframe="1h", limit=24)
        assert result == expected

    def test_acepta_parametros_custom(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        mock_client.fetch_ohlcv.return_value = []
        analyzer.fetch_candles("ETH/USDT", timeframe="4h", limit=50)
        mock_client.fetch_ohlcv.assert_called_once_with("ETH/USDT", timeframe="4h", limit=50)

    def test_usa_timeframe_del_constructor_si_no_se_provee(
        self, mock_client: MagicMock
    ):
        an = MarketAnalyzer(client=mock_client, timeframe="4h", candle_limit=10)
        mock_client.fetch_ohlcv.return_value = []
        an.fetch_candles("SOL/USDT")
        mock_client.fetch_ohlcv.assert_called_once_with("SOL/USDT", timeframe="4h", limit=10)


# ---------------------------------------------------------------------------
# analyze — integración con mocks
# ---------------------------------------------------------------------------


class TestAnalyze:
    def _setup_ranging(self, mock_client: MagicMock, price: float = 100.0) -> None:
        """Configura el mock para un mercado lateral."""
        candles = _make_candles(_flat_prices(25, price))
        mock_client.fetch_ohlcv.return_value = candles
        mock_client.fetch_ticker.return_value = _make_ticker(
            bid=price - 0.5, ask=price + 0.5, last=price
        )

    def _setup_trending_up(self, mock_client: MagicMock, price: float = 100.0) -> None:
        """Configura el mock para mercado alcista fuerte."""
        candles = _make_candles(_rising_prices(25, price, step=2.0))
        mock_client.fetch_ohlcv.return_value = candles
        mock_client.fetch_ticker.return_value = _make_ticker(
            bid=price - 0.5, ask=price + 0.5, last=price + 48
        )

    def _setup_trending_down(self, mock_client: MagicMock, price: float = 150.0) -> None:
        """Configura el mock para mercado bajista fuerte."""
        candles = _make_candles(_falling_prices(25, price, step=2.0))
        mock_client.fetch_ohlcv.return_value = candles
        mock_client.fetch_ticker.return_value = _make_ticker(
            bid=price - 48.5, ask=price - 47.5, last=price - 48
        )

    def test_retorna_marketregime(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        self._setup_ranging(mock_client)
        result = analyzer.analyze("BTC/USDT")
        assert isinstance(result, MarketRegime)

    def test_symbol_se_propaga_en_el_resultado(
        self, analyzer: MarketAnalyzer, mock_client: MagicMock
    ):
        self._setup_ranging(mock_client)
        result = analyzer.analyze("ETH/USDT")
        assert result.symbol == "ETH/USDT"

    def test_mercado_plano_clasifica_ranging(
        self, analyzer: MarketAnalyzer, mock_client: MagicMock
    ):
        self._setup_ranging(mock_client)
        result = analyzer.analyze("BTC/USDT")
        assert result.regime == RANGING

    def test_mercado_alcista_clasifica_trending_up(
        self, mock_client: MagicMock
    ):
        an = MarketAnalyzer(
            client=mock_client,
            slope_threshold=0.0001,  # umbral bajo para detectar tendencias moderadas
        )
        self._setup_trending_up(mock_client)
        result = an.analyze("BTC/USDT")
        assert result.regime == TRENDING_UP

    def test_mercado_bajista_clasifica_trending_down(
        self, mock_client: MagicMock
    ):
        an = MarketAnalyzer(
            client=mock_client,
            slope_threshold=0.0001,
        )
        self._setup_trending_down(mock_client)
        result = an.analyze("BTC/USDT")
        assert result.regime == TRENDING_DOWN

    def test_atr_pct_es_positivo(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        self._setup_ranging(mock_client)
        result = analyzer.analyze("BTC/USDT")
        assert result.atr_pct >= 0.0

    def test_spread_pct_calculado_correctamente(
        self, analyzer: MarketAnalyzer, mock_client: MagicMock
    ):
        price = 100.0
        candles = _make_candles(_flat_prices(25, price))
        mock_client.fetch_ohlcv.return_value = candles
        mock_client.fetch_ticker.return_value = _make_ticker(
            bid=99.0, ask=101.0, last=100.0
        )
        result = analyzer.analyze("BTC/USDT")
        # spread = (101-99) / 100 = 0.02
        assert result.spread_pct == pytest.approx(0.02, rel=1e-4)

    def test_score_entre_cero_y_uno(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        self._setup_ranging(mock_client)
        result = analyzer.analyze("BTC/USDT")
        assert 0.0 <= result.score <= 1.0

    def test_fetch_ohlcv_llamado_una_vez(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        self._setup_ranging(mock_client)
        analyzer.analyze("BTC/USDT")
        mock_client.fetch_ohlcv.assert_called_once()

    def test_fetch_ticker_llamado_una_vez(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        self._setup_ranging(mock_client)
        analyzer.analyze("BTC/USDT")
        mock_client.fetch_ticker.assert_called_once_with("BTC/USDT")

    def test_datos_insuficientes_retorna_ranging_con_score_cero(
        self, analyzer: MarketAnalyzer, mock_client: MagicMock
    ):
        """Cuando hay pocas velas, retorna RANGING con score=0 y atr_pct=0."""
        mock_client.fetch_ohlcv.return_value = _make_candles([100.0] * 5)
        mock_client.fetch_ticker.return_value = _make_ticker()

        result = analyzer.analyze("BTC/USDT")

        assert result.regime == RANGING
        assert result.atr_pct == 0.0
        assert result.score == 0.0

    def test_precio_cero_en_ticker_retorna_fallback(
        self, analyzer: MarketAnalyzer, mock_client: MagicMock
    ):
        """Si el ticker no tiene precio (last=0), se activa la ruta de datos insuficientes."""
        mock_client.fetch_ohlcv.return_value = _make_candles(_flat_prices(25, 100.0))
        mock_client.fetch_ticker.return_value = {"bid": 0.0, "ask": 0.0, "last": 0.0}

        result = analyzer.analyze("BTC/USDT")

        assert result.regime == RANGING
        assert result.atr_pct == 0.0

    def test_ranging_da_score_mayor_que_trending(self, mock_client: MagicMock):
        """El score de RANGING debe superar al de TRENDING con misma volatilidad."""
        an = MarketAnalyzer(client=mock_client, slope_threshold=0.0001)

        price = 100.0
        self._setup_ranging(mock_client)
        result_ranging = an.analyze("BTC/USDT")

        self._setup_trending_up(mock_client)
        result_trending = an.analyze("BTC/USDT")

        assert result_ranging.score >= result_trending.score

    def test_uses_last_price_from_ticker(self, analyzer: MarketAnalyzer, mock_client: MagicMock):
        """Verifica que se usa `last` del ticker para normalizar el ATR."""
        price = 50_000.0
        candles = _make_candles(_flat_prices(25, price))
        mock_client.fetch_ohlcv.return_value = candles
        mock_client.fetch_ticker.return_value = _make_ticker(
            bid=price - 50, ask=price + 50, last=price
        )
        result = analyzer.analyze("BTC/USDT")
        # atr_pct = atr / price → un precio más alto da un atr_pct más pequeño
        assert result.atr_pct < 0.1
