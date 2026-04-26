"""
Análisis de régimen de mercado para optimizar la estrategia de grid.

Calcula indicadores técnicos (ATR, pendiente de SMA, Bollinger Bandwidth)
y clasifica el mercado en RANGING, TRENDING_UP o TRENDING_DOWN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient

# OHLCV: [timestamp_ms, open, high, low, close, volume]
OHLCV = list[float]

logger = structlog.get_logger(__name__)

# Regímenes posibles
RANGING = "RANGING"
TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"

# Mínimo de velas necesarias para calcular todos los indicadores
_MIN_CANDLES_REQUIRED = 22  # max(atr_period+1=15, sma_period+1=21, bb_period=20)


@dataclass
class MarketRegime:
    """
    Resultado del análisis de mercado para un símbolo.

    Attributes:
        regime: Clasificación del mercado (RANGING, TRENDING_UP, TRENDING_DOWN).
        atr_pct: ATR expresado como porcentaje del precio actual (0.03 = 3%).
        spread_pct: Spread bid/ask normalizado (0.001 = 0.1%).
        score: Score de utilidad para grid trading (0.0 a 1.0).
        symbol: Par analizado.
    """

    regime: str
    atr_pct: float
    spread_pct: float
    score: float
    symbol: str = field(default="")


class MarketAnalyzer:
    """
    Analiza el régimen de mercado de un símbolo usando indicadores técnicos.

    Indicadores calculados:
    - ATR período 14: mide la volatilidad absoluta del mercado.
    - Pendiente de SMA período 20: detecta dirección y fuerza de la tendencia.
    - Bollinger Bandwidth: mide si el precio está en rango o en expansión.

    Thresholds de clasificación:
    - TRENDING_UP: pendiente SMA > slope_threshold
    - TRENDING_DOWN: pendiente SMA < -slope_threshold
    - RANGING: pendiente dentro de ±slope_threshold (mercado lateral)
    """

    def __init__(
        self,
        client: "OKXClient",
        timeframe: str = "1h",
        candle_limit: int = 24,
        atr_period: int = 14,
        sma_period: int = 20,
        slope_threshold: float = 0.0005,
        bb_threshold: float = 0.04,
        vol_weight: float = 10.0,
        spread_weight: float = 100.0,
    ) -> None:
        """
        Args:
            client: Instancia de OKXClient para llamadas al exchange.
            timeframe: Intervalo de velas (ej: "1h", "4h").
            candle_limit: Número de velas a descargar.
            atr_period: Período del ATR (14 por convención).
            sma_period: Período de la SMA (20 por convención).
            slope_threshold: Umbral de pendiente (fracción/vela) para detectar tendencia.
            bb_threshold: Umbral de Bollinger Bandwidth para diferenciar rango de expansión.
            vol_weight: Peso de la volatilidad en el score.
            spread_weight: Peso del spread en el score (penalización).
        """
        self._client = client
        self._timeframe = timeframe
        self._candle_limit = candle_limit
        self._atr_period = atr_period
        self._sma_period = sma_period
        self._slope_threshold = slope_threshold
        self._bb_threshold = bb_threshold
        self._vol_weight = vol_weight
        self._spread_weight = spread_weight

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """
        Descarga velas OHLCV para el símbolo dado.

        Args:
            symbol: Par de trading (ej: "BTC/USDT").
            timeframe: Intervalo de velas. Si None usa el configurado en el constructor.
            limit: Número de velas. Si None usa el configurado en el constructor.

        Returns:
            Lista de velas OHLCV [timestamp, open, high, low, close, volume].
        """
        tf = timeframe or self._timeframe
        lim = limit or self._candle_limit
        candles: list[OHLCV] = self._client.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
        logger.debug(
            "velas_descargadas",
            symbol=symbol,
            timeframe=tf,
            limit=lim,
            count=len(candles),
        )
        return candles

    def analyze(self, symbol: str) -> MarketRegime:
        """
        Analiza el régimen de mercado actual para el símbolo dado.

        Descarga velas y ticker, calcula ATR, pendiente de SMA y Bollinger
        Bandwidth, clasifica el régimen y devuelve un MarketRegime.

        Args:
            symbol: Par de trading (ej: "BTC/USDT").

        Returns:
            MarketRegime con regime, atr_pct, spread_pct y score.
        """
        candles = self.fetch_candles(symbol)
        ticker = self._client.fetch_ticker(symbol)

        spread_pct = self._calc_spread_pct(ticker)
        current_price = float(ticker.get("last") or ticker.get("close") or 0.0)

        if len(candles) < _MIN_CANDLES_REQUIRED or current_price <= 0:
            logger.warning(
                "datos_insuficientes_para_analisis",
                symbol=symbol,
                candles=len(candles),
                current_price=current_price,
            )
            return MarketRegime(
                regime=RANGING,
                atr_pct=0.0,
                spread_pct=spread_pct,
                score=0.0,
                symbol=symbol,
            )

        atr = self._calc_atr(candles, self._atr_period)
        atr_pct = atr / current_price if current_price > 0 else 0.0

        slope = self._calc_sma_slope(candles, self._sma_period)
        bb_bw = self._calc_bollinger_bandwidth(candles, self._sma_period)

        regime = self._classify_regime(slope, bb_bw)
        score = self._calc_score(atr_pct, spread_pct, regime)

        logger.info(
            "analisis_mercado",
            symbol=symbol,
            regime=regime,
            atr_pct=round(atr_pct * 100, 4),
            spread_pct=round(spread_pct * 100, 4),
            sma_slope=round(slope, 6),
            bb_bandwidth=round(bb_bw, 4),
            score=round(score, 4),
        )

        return MarketRegime(
            regime=regime,
            atr_pct=atr_pct,
            spread_pct=spread_pct,
            score=score,
            symbol=symbol,
        )

    # ------------------------------------------------------------------
    # Indicadores técnicos (métodos estáticos para facilitar testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_atr(candles: list[OHLCV], period: int = 14) -> float:
        """
        Calcula el Average True Range (ATR) del período indicado.

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = promedio simple de los últimos `period` True Ranges.

        Requiere al menos period + 1 velas.
        """
        if len(candles) < period + 1:
            return 0.0

        true_ranges: list[float] = []
        for i in range(1, len(candles)):
            high = float(candles[i][2])
            low = float(candles[i][3])
            prev_close = float(candles[i - 1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return sum(true_ranges[-period:]) / period

    @staticmethod
    def _calc_sma_slope(candles: list[OHLCV], period: int = 20) -> float:
        """
        Calcula la pendiente normalizada de la SMA.

        Compara la SMA calculada sobre las primeras `period` velas con la SMA
        sobre las últimas `period` velas. La pendiente se expresa como fracción
        del precio inicial por vela (adimensional).

        Retorna un valor positivo en tendencia alcista y negativo en bajista.
        Requiere al menos period + 1 velas para tener dos SMAs distintas.
        """
        if len(candles) < period + 1:
            return 0.0

        closes = [float(c[4]) for c in candles]
        first_sma = sum(closes[:period]) / period
        last_sma = sum(closes[-period:]) / period

        n_steps = len(closes) - period
        if n_steps == 0 or first_sma == 0:
            return 0.0

        return (last_sma - first_sma) / (first_sma * n_steps)

    @staticmethod
    def _calc_bollinger_bandwidth(candles: list[OHLCV], period: int = 20) -> float:
        """
        Calcula el Bollinger Bandwidth sobre las últimas `period` velas.

        Bollinger Bandwidth = (banda_superior - banda_inferior) / SMA
                            = 4 * desviación_estándar / SMA

        Un valor bajo indica mercado en rango; uno alto indica expansión/tendencia.
        Requiere al menos `period` velas.
        """
        if len(candles) < period:
            return 0.0

        closes = [float(c[4]) for c in candles[-period:]]
        sma = sum(closes) / period
        if sma == 0:
            return 0.0

        variance = sum((c - sma) ** 2 for c in closes) / period
        std = variance ** 0.5
        return (4.0 * std) / sma

    # ------------------------------------------------------------------
    # Lógica de clasificación y scoring
    # ------------------------------------------------------------------

    def _classify_regime(self, slope: float, bb_bandwidth: float) -> str:
        """
        Clasifica el régimen de mercado combinando pendiente de SMA y Bollinger BW.

        Prioridad:
        1. Si la pendiente supera ±slope_threshold → TRENDING (sube o baja).
        2. Si el bandwidth es bajo (< bb_threshold) → RANGING.
        3. Si el bandwidth es alto pero la pendiente es plana → RANGING también
           (volatilidad sin dirección, ej: mercado oscilante).
        """
        if slope > self._slope_threshold:
            return TRENDING_UP
        if slope < -self._slope_threshold:
            return TRENDING_DOWN
        return RANGING

    def _calc_score(self, atr_pct: float, spread_pct: float, regime: str) -> float:
        """
        Calcula el score de utilidad para grid trading (0.0 a 1.0).

        Mayor score = mejor oportunidad para grid:
        - Alta volatilidad (ATR alto) → más cruces de niveles → más trades.
        - Spread bajo → fills más baratos.
        - Régimen lateral → el precio oscila dentro del rango en lugar de escapar.

        Fórmula: score = atr_pct * vol_weight * regime_factor - spread_pct * spread_weight
        Clampeado a [0.0, 1.0].
        """
        regime_factor = 1.0 if regime == RANGING else 0.3
        raw = atr_pct * self._vol_weight * regime_factor - spread_pct * self._spread_weight
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _calc_spread_pct(ticker: dict) -> float:
        """
        Calcula el spread bid/ask normalizado respecto al mid-price.

        spread_pct = (ask - bid) / mid_price
        Si bid o ask no están disponibles retorna 0.0.
        """
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid if mid > 0 else 0.0
