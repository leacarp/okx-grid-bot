"""
Persistencia del estado de la grilla en data/grid_state.json.

Usa escritura atómica (archivo temporal + rename) para garantizar
que el JSON nunca quede corrupto ante una interrupción abrupta.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path("data/grid_state.json")


class GridState:
    """
    Persiste y gestiona el estado de la grilla.

    El estado incluye: configuración de la grilla, niveles con sus órdenes,
    ganancia acumulada y número de trades completados.
    """

    def __init__(self, state_file: Path | str = DEFAULT_STATE_FILE) -> None:
        self._path = Path(state_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = {}

    def initialize(
        self,
        symbol: str,
        grid_config: dict[str, Any],
        levels: list[dict[str, Any]],
    ) -> None:
        """
        Inicializa el estado desde cero con la configuración de la grilla.

        Sobreescribe cualquier estado previo en memoria y disco.
        """
        self._state = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "grid_config": grid_config,
            "levels": levels,
            "total_profit_usdt": 0.0,
            "total_trades": 0,
        }
        self.save()
        logger.info(
            "grid_state_inicializado | symbol=%s levels=%d",
            symbol,
            len(levels),
        )

    def save(self) -> None:
        """
        Persiste el estado en disco de forma atómica.

        Escribe a un archivo temporal y luego hace rename para evitar
        archivos corruptos ante interrupciones.
        """
        dir_ = self._path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(dir_),
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as f:
            json.dump(self._state, f, indent=2)
            tmp_path = f.name

        os.replace(tmp_path, str(self._path))
        logger.debug("grid_state_guardado | path=%s", self._path)

    def load(self) -> bool:
        """
        Carga el estado desde disco.

        Returns:
            True si se cargó correctamente, False si el archivo no existe.
        """
        if not self._path.exists():
            logger.info("grid_state_no_existe | path=%s", self._path)
            return False

        with self._path.open(encoding="utf-8") as f:
            self._state = json.load(f)

        logger.info(
            "grid_state_cargado | path=%s levels=%d",
            self._path,
            len(self._state.get("levels", [])),
        )
        return True

    def update_level(
        self,
        index: int,
        status: str,
        order_id: str | None = None,
    ) -> None:
        """
        Actualiza el status (y opcionalmente el order_id) de un nivel.

        Args:
            index: Índice del nivel a actualizar.
            status: Nuevo status (ej: "buy_open", "buy_filled", "sell_open", "cancelled").
            order_id: ID de la nueva orden, si aplica.
        """
        for level in self._state.get("levels", []):
            if level["index"] == index:
                level["status"] = status
                if order_id is not None:
                    level["order_id"] = order_id
                return

        logger.warning("update_level_no_encontrado | index=%d", index)

    def clear_levels(self) -> None:
        """
        Vacía la lista de niveles para permitir reinicialización tras un recentrado.

        Preserva el historial de profit y trades acumulados.
        """
        self._state["levels"] = []
        logger.debug("grid_state_niveles_limpiados")

    def record_profit(self, amount: float) -> None:
        """Acumula la ganancia de un ciclo completado (BUY + SELL ejecutados)."""
        self._state["total_profit_usdt"] = round(
            self._state.get("total_profit_usdt", 0.0) + amount, 8
        )
        self._state["total_trades"] = self._state.get("total_trades", 0) + 1

    @property
    def state(self) -> dict[str, Any]:
        """Retorna el estado completo (solo lectura recomendada)."""
        return self._state

    @property
    def levels(self) -> list[dict[str, Any]]:
        """Retorna los niveles de la grilla."""
        return self._state.get("levels", [])

    @property
    def total_profit(self) -> float:
        return self._state.get("total_profit_usdt", 0.0)

    @property
    def total_trades(self) -> int:
        return self._state.get("total_trades", 0)
