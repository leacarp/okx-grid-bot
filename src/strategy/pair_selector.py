"""
Selección inteligente del par óptimo para grid trading.

Evalúa un conjunto de análisis de mercado (MarketRegime) y elige el par con
mayor score de utilidad para grid, aplicando:
  - Penalización del 40% en régimen tendencial (TRENDING_UP / TRENDING_DOWN).
  - Hysteresis: solo cambia de par si el nuevo supera al actual en ≥20%.
  - Cooldown: mínimo 2 horas entre cambios de par.
  - Persistencia: almacena el estado en JSON para sobrevivir reinicios.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.strategy.market_analyzer import RANGING, MarketRegime

logger = structlog.get_logger(__name__)

_DEFAULT_STATE_FILE = Path("data/pair_selector_state.json")
_DEFAULT_PAIR_COOLDOWN_SECONDS: float = 7200.0   # 2 horas
_DEFAULT_HYSTERESIS_MARGIN: float = 0.20          # 20%
_DEFAULT_VOL_WEIGHT: float = 10.0
_DEFAULT_SPREAD_WEIGHT: float = 100.0
_DEFAULT_MIN_SCORE: float = 0.30


@dataclass
class PairSelection:
    """
    Resultado de la selección de par óptimo.

    Attributes:
        symbol: Par seleccionado (ej. "ETH/USDT").
        regime: Análisis completo del mercado para ese par.
        score: Score de utilidad calculado por PairSelector (0.0 a 1.0).
        reason: Motivo de la selección (para logging/auditoría).
    """

    symbol: str
    regime: MarketRegime
    score: float
    reason: str


class PairSelector:
    """
    Selecciona el par con mejor score para grid trading.

    Scoring:
        raw = atr_pct * vol_weight - spread_pct * spread_weight
        Si el régimen es TRENDING_UP o TRENDING_DOWN: raw *= 0.6 (penaliza 40%)
        score = clamp(raw, 0.0, 1.0)

    Hysteresis:
        Solo cambia de par si score_nuevo > score_actual * (1 + hysteresis_margin).
        No puede cambiar si no pasó pair_cooldown_seconds desde el último cambio.

    Persistencia:
        El par actual y el timestamp del último cambio se guardan en state_file.
    """

    def __init__(
        self,
        min_score_to_trade: float = _DEFAULT_MIN_SCORE,
        vol_weight: float = _DEFAULT_VOL_WEIGHT,
        spread_weight: float = _DEFAULT_SPREAD_WEIGHT,
        hysteresis_margin: float = _DEFAULT_HYSTERESIS_MARGIN,
        pair_cooldown_seconds: float = _DEFAULT_PAIR_COOLDOWN_SECONDS,
        state_file: Path | str = _DEFAULT_STATE_FILE,
    ) -> None:
        """
        Args:
            min_score_to_trade: Score mínimo para operar cualquier par.
            vol_weight: Peso de la volatilidad (ATR%) en el score.
            spread_weight: Peso del spread (penalización) en el score.
            hysteresis_margin: Fracción mínima de mejora para cambiar de par (0.20 = 20%).
            pair_cooldown_seconds: Mínimo de segundos entre cambios de par.
            state_file: Ruta del archivo JSON de persistencia.
        """
        self._min_score_to_trade = min_score_to_trade
        self._vol_weight = vol_weight
        self._spread_weight = spread_weight
        self._hysteresis_margin = hysteresis_margin
        self._pair_cooldown_seconds = pair_cooldown_seconds
        self._state_file = Path(state_file)

        # Estado interno (se sobrescribe en _load_state si existe)
        self._current_symbol: str | None = None
        self._current_score: float = 0.0
        self._last_switch_time: float = 0.0

        self._load_state()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def select_best_pair(self, analyses: list[MarketRegime]) -> PairSelection | None:
        """
        Evalúa los análisis y retorna la mejor selección de par.

        Aplica hysteresis y cooldown antes de permitir un cambio.
        Si ningún par supera min_score_to_trade retorna None.

        Args:
            analyses: Lista de MarketRegime, uno por par candidato.

        Returns:
            PairSelection con el par elegido, o None si ninguno es apto.
        """
        if not analyses:
            logger.warning("pair_selector_sin_analisis")
            return None

        scored = [
            (self._compute_score(a), a)
            for a in analyses
        ]
        scored.sort(key=lambda t: t[0], reverse=True)

        best_score, best_analysis = scored[0]

        logger.debug(
            "pair_selector_scores",
            scores=[
                {"symbol": a.symbol, "score": round(s, 4), "regime": a.regime}
                for s, a in scored
            ],
        )

        if best_score < self._min_score_to_trade:
            logger.warning(
                "pair_selector_ninguno_apto",
                best_symbol=best_analysis.symbol,
                best_score=round(best_score, 4),
                min_requerido=self._min_score_to_trade,
            )
            return None

        best_symbol = best_analysis.symbol

        # Si es el mismo par actual → devolver sin hysteresis check
        if best_symbol == self._current_symbol:
            self._current_score = best_score
            return PairSelection(
                symbol=best_symbol,
                regime=best_analysis,
                score=best_score,
                reason="mismo_par",
            )

        # Primer par (todavía no hay selección previa)
        if self._current_symbol is None:
            selection = PairSelection(
                symbol=best_symbol,
                regime=best_analysis,
                score=best_score,
                reason="primer_par",
            )
            self._record_switch(best_symbol, best_score)
            return selection

        # Verificar cooldown
        elapsed = time.time() - self._last_switch_time
        if elapsed < self._pair_cooldown_seconds:
            remaining = self._pair_cooldown_seconds - elapsed
            logger.info(
                "pair_selector_cooldown_activo",
                actual=self._current_symbol,
                candidato=best_symbol,
                segundos_restantes=round(remaining, 0),
            )
            # Devolver el par actual si aún es apto
            current_analysis = next(
                (a for _, a in scored if a.symbol == self._current_symbol), None
            )
            if current_analysis is not None:
                current_score = self._compute_score(current_analysis)
                if current_score >= self._min_score_to_trade:
                    self._current_score = current_score
                    return PairSelection(
                        symbol=self._current_symbol,
                        regime=current_analysis,
                        score=current_score,
                        reason="cooldown_mantiene_actual",
                    )
            # Si el par actual no está en la lista o tiene score < mínimo
            # → mantener sin análisis actualizado
            return PairSelection(
                symbol=self._current_symbol,
                regime=best_analysis,  # fallback: datos del mejor disponible
                score=self._current_score,
                reason="cooldown_fallback",
            )

        # Verificar hysteresis: el nuevo debe superar al actual en ≥ margin
        threshold = self._current_score * (1.0 + self._hysteresis_margin)
        if best_score <= threshold:
            logger.info(
                "pair_selector_hysteresis_no_supera",
                actual=self._current_symbol,
                candidato=best_symbol,
                score_candidato=round(best_score, 4),
                score_threshold=round(threshold, 4),
            )
            # Devolver el par actual con su score recalculado
            current_analysis = next(
                (a for _, a in scored if a.symbol == self._current_symbol), None
            )
            if current_analysis is not None:
                current_score = self._compute_score(current_analysis)
                self._current_score = current_score
                return PairSelection(
                    symbol=self._current_symbol,
                    regime=current_analysis,
                    score=current_score,
                    reason="hysteresis_mantiene_actual",
                )
            return PairSelection(
                symbol=self._current_symbol,
                regime=best_analysis,
                score=self._current_score,
                reason="hysteresis_fallback",
            )

        # Cambio autorizado
        selection = PairSelection(
            symbol=best_symbol,
            regime=best_analysis,
            score=best_score,
            reason="mejor_score",
        )
        logger.info(
            "pair_selector_cambio_autorizado",
            anterior=self._current_symbol,
            nuevo=best_symbol,
            score_nuevo=round(best_score, 4),
            score_anterior=round(self._current_score, 4),
            threshold=round(threshold, 4),
        )
        self._record_switch(best_symbol, best_score)
        return selection

    # ------------------------------------------------------------------
    # Scoring interno
    # ------------------------------------------------------------------

    def _compute_score(self, analysis: MarketRegime) -> float:
        """
        Calcula el score de utilidad para un par.

        Fórmula:
            raw = atr_pct * vol_weight - spread_pct * spread_weight
            Si trending: raw *= 0.6  (penalización del 40%)
            score = clamp(raw, 0.0, 1.0)

        Args:
            analysis: Resultado del análisis de mercado.

        Returns:
            Score entre 0.0 y 1.0.
        """
        raw = (
            analysis.atr_pct * self._vol_weight
            - analysis.spread_pct * self._spread_weight
        )
        if analysis.regime != RANGING:
            raw *= 0.6
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _record_switch(self, symbol: str, score: float) -> None:
        """Actualiza el estado interno y persiste tras un cambio de par."""
        self._current_symbol = symbol
        self._current_score = score
        self._last_switch_time = time.time()
        self._save_state()

    def _save_state(self) -> None:
        """Persiste el estado en el archivo JSON."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state: dict[str, Any] = {
                "current_symbol": self._current_symbol,
                "current_score": self._current_score,
                "last_switch_time": self._last_switch_time,
            }
            self._state_file.write_text(json.dumps(state, indent=2))
            logger.debug(
                "pair_selector_estado_guardado",
                symbol=self._current_symbol,
                score=round(self._current_score, 4),
            )
        except Exception as exc:
            logger.warning("pair_selector_error_guardando_estado", error=str(exc))

    def _load_state(self) -> None:
        """Carga el estado desde el archivo JSON si existe."""
        try:
            if not self._state_file.exists():
                return
            raw = json.loads(self._state_file.read_text())
            self._current_symbol = raw.get("current_symbol")
            self._current_score = float(raw.get("current_score", 0.0))
            self._last_switch_time = float(raw.get("last_switch_time", 0.0))
            logger.info(
                "pair_selector_estado_cargado",
                symbol=self._current_symbol,
                score=round(self._current_score, 4),
                last_switch_ago_s=round(time.time() - self._last_switch_time, 0),
            )
        except Exception as exc:
            logger.warning("pair_selector_error_cargando_estado", error=str(exc))

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def current_symbol(self) -> str | None:
        """Par actualmente seleccionado (None si todavía no se eligió ninguno)."""
        return self._current_symbol

    @property
    def current_score(self) -> float:
        """Score del par actualmente seleccionado."""
        return self._current_score

    @property
    def last_switch_time(self) -> float:
        """Timestamp UNIX del último cambio de par."""
        return self._last_switch_time
