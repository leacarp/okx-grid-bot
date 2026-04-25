"""
Rastreador de PnL real por trade.

Registra fills con precio real de ejecución, cantidad y fees,
calcula ganancias netas por ciclo completado (BUY + SELL)
y persiste el historial en data/trade_history.json.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_HISTORY_FILE = Path("data/trade_history.json")


# ---------------------------------------------------------------------------
# Dataclasses públicos
# ---------------------------------------------------------------------------


@dataclass
class FillRecord:
    """Registro de un fill individual (una orden ejecutada)."""

    order_id: str
    symbol: str
    side: str        # "buy" o "sell"
    amount: float    # cantidad ejecutada en moneda base
    price: float     # precio real de ejecución
    fee: float       # fee pagada en moneda quote (USDT)
    timestamp: str   # ISO 8601 UTC


@dataclass
class CompletedCycle:
    """Un ciclo completo BUY + SELL con su ganancia calculada."""

    buy_fill: FillRecord
    sell_fill: FillRecord
    gross_profit: float   # (sell_price - buy_price) * amount
    total_fees: float     # buy_fee + sell_fee
    net_profit: float     # gross_profit - total_fees
    completed_at: str     # ISO 8601 UTC


@dataclass
class PnLSummary:
    """Resumen total del PnL acumulado en todos los ciclos."""

    total_cycles: int
    gross_profit: float
    total_fees: float
    net_profit: float


# ---------------------------------------------------------------------------
# PnLTracker
# ---------------------------------------------------------------------------


class PnLTracker:
    """
    Rastreador de PnL real por trade.

    Flujo de uso:
      1. Llamar ``record_fill()`` para cada orden ejecutada (BUY o SELL).
      2. Cuando ambos lados del nivel se completan, llamar ``record_cycle()``
         con los dos FillRecord correspondientes.
      3. Consultar ``get_daily_pnl()`` desde RiskManager para el circuit breaker.
      4. Consultar ``get_summary()`` para métricas globales.

    Persiste todos los ciclos completados en ``data/trade_history.json``
    usando escritura atómica (tempfile + os.replace) para evitar corrupción.
    """

    def __init__(self, history_file: Path | str = DEFAULT_HISTORY_FILE) -> None:
        self._path = Path(history_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cycles: list[CompletedCycle] = []
        self._load()

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        fee: float,
    ) -> FillRecord:
        """
        Crea y retorna un FillRecord para un fill individual.

        No persiste automáticamente; la persistencia ocurre al registrar
        un ciclo completo con ``record_cycle()``.

        Args:
            order_id: ID de la orden en el exchange.
            symbol: Par de trading (ej: "BTC/USDT").
            side: "buy" o "sell".
            amount: Cantidad ejecutada en moneda base.
            price: Precio real de ejecución.
            fee: Fee pagada en moneda quote (USDT).

        Returns:
            FillRecord con los datos del fill y timestamp UTC.

        Raises:
            ValueError: Si algún parámetro es inválido.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side inválido: {side!r}. Debe ser 'buy' o 'sell'.")
        if amount <= 0:
            raise ValueError(f"amount debe ser > 0, recibido: {amount}")
        if price <= 0:
            raise ValueError(f"price debe ser > 0, recibido: {price}")
        if fee < 0:
            raise ValueError(f"fee no puede ser negativo, recibido: {fee}")

        fill = FillRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=fee,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        logger.debug(
            "fill_registrado",
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=fee,
        )
        return fill

    def calculate_cycle_profit(
        self,
        buy_fill: FillRecord,
        sell_fill: FillRecord,
    ) -> float:
        """
        Calcula la ganancia neta de un ciclo completado (BUY + SELL).

        Ganancia bruta = (sell_price - buy_price) * min(buy_amount, sell_amount).
        Se descuentan las fees reales de ambos lados.

        Args:
            buy_fill: FillRecord del lado comprador.
            sell_fill: FillRecord del lado vendedor.

        Returns:
            Ganancia neta en USDT (puede ser negativa si las fees superan el spread).
        """
        amount = min(buy_fill.amount, sell_fill.amount)
        gross = (sell_fill.price - buy_fill.price) * amount
        net = round(gross - buy_fill.fee - sell_fill.fee, 8)

        logger.info(
            "ciclo_calculado",
            symbol=buy_fill.symbol,
            buy_price=buy_fill.price,
            sell_price=sell_fill.price,
            amount=amount,
            gross_profit=round(gross, 8),
            total_fees=round(buy_fill.fee + sell_fill.fee, 8),
            net_profit=net,
        )
        return net

    def record_cycle(
        self,
        buy_fill: FillRecord,
        sell_fill: FillRecord,
    ) -> CompletedCycle:
        """
        Registra un ciclo completado y lo persiste en disco.

        Args:
            buy_fill: FillRecord del lado comprador.
            sell_fill: FillRecord del lado vendedor.

        Returns:
            CompletedCycle con todos los datos calculados y el timestamp de cierre.
        """
        amount = min(buy_fill.amount, sell_fill.amount)
        gross = round((sell_fill.price - buy_fill.price) * amount, 8)
        total_fees = round(buy_fill.fee + sell_fill.fee, 8)
        net = round(gross - total_fees, 8)

        cycle = CompletedCycle(
            buy_fill=buy_fill,
            sell_fill=sell_fill,
            gross_profit=gross,
            total_fees=total_fees,
            net_profit=net,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._cycles.append(cycle)
        self._save()

        logger.info(
            "ciclo_registrado",
            symbol=buy_fill.symbol,
            net_profit=net,
            total_cycles=len(self._cycles),
        )
        return cycle

    def get_daily_pnl(self) -> float:
        """
        Retorna el PnL neto del día actual (UTC).

        Returns:
            Suma de net_profit de todos los ciclos completados hoy (UTC).
        """
        today = datetime.now(timezone.utc).date()
        daily = sum(
            c.net_profit
            for c in self._cycles
            if datetime.fromisoformat(c.completed_at).date() == today
        )
        return round(daily, 8)

    def get_summary(self) -> PnLSummary:
        """
        Retorna el resumen total del PnL acumulado.

        Returns:
            PnLSummary con total de ciclos, profit bruto, fees totales y profit neto.
        """
        gross = round(sum(c.gross_profit for c in self._cycles), 8)
        fees = round(sum(c.total_fees for c in self._cycles), 8)
        net = round(sum(c.net_profit for c in self._cycles), 8)

        return PnLSummary(
            total_cycles=len(self._cycles),
            gross_profit=gross,
            total_fees=fees,
            net_profit=net,
        )

    # ------------------------------------------------------------------
    # Persistencia interna
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persiste el historial de ciclos en disco de forma atómica."""
        data = [self._cycle_to_dict(c) for c in self._cycles]
        dir_ = self._path.parent

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(dir_),
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=2)
            tmp_path = f.name

        os.replace(tmp_path, str(self._path))
        logger.debug(
            "trade_history_guardado",
            path=str(self._path),
            cycles=len(self._cycles),
        )

    def _load(self) -> None:
        """Carga el historial desde disco si existe."""
        if not self._path.exists():
            logger.info("trade_history_no_existe", path=str(self._path))
            return

        with self._path.open(encoding="utf-8") as f:
            raw: list[dict] = json.load(f)

        self._cycles = [self._dict_to_cycle(d) for d in raw]
        logger.info(
            "trade_history_cargado",
            path=str(self._path),
            cycles=len(self._cycles),
        )

    # ------------------------------------------------------------------
    # Helpers de serialización
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_to_dict(fill: FillRecord) -> dict:
        return asdict(fill)

    @staticmethod
    def _dict_to_fill(data: dict) -> FillRecord:
        return FillRecord(**data)

    def _cycle_to_dict(self, cycle: CompletedCycle) -> dict:
        return {
            "buy_fill": self._fill_to_dict(cycle.buy_fill),
            "sell_fill": self._fill_to_dict(cycle.sell_fill),
            "gross_profit": cycle.gross_profit,
            "total_fees": cycle.total_fees,
            "net_profit": cycle.net_profit,
            "completed_at": cycle.completed_at,
        }

    def _dict_to_cycle(self, data: dict) -> CompletedCycle:
        return CompletedCycle(
            buy_fill=self._dict_to_fill(data["buy_fill"]),
            sell_fill=self._dict_to_fill(data["sell_fill"]),
            gross_profit=data["gross_profit"],
            total_fees=data["total_fees"],
            net_profit=data["net_profit"],
            completed_at=data["completed_at"],
        )
