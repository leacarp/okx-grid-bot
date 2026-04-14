"""
Gestión de órdenes: colocación inicial, monitoreo y reciclado.

La lógica de reciclado:
  - BUY en nivel N ejecutada  → coloca SELL en nivel N+1
  - SELL en nivel N ejecutada → coloca BUY en nivel N-1
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient
    from src.risk.risk_manager import RiskManager
    from src.strategy.grid_calculator import GridLevel
    from src.strategy.grid_state import GridState

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Coloca y gestiona órdenes en la grilla.

    En DRY_RUN (controlado por OKXClient), las órdenes son simuladas y
    el log registra cada "orden simulada" sin enviar nada al exchange.
    """

    def __init__(
        self,
        client: "OKXClient",
        risk_manager: "RiskManager",
        grid_state: "GridState",
        symbol: str,
        max_order_usdt: float,
    ) -> None:
        self._client = client
        self._risk = risk_manager
        self._state = grid_state
        self._symbol = symbol
        self._max_order_usdt = max_order_usdt

    # ------------------------------------------------------------------
    # Colocación inicial
    # ------------------------------------------------------------------

    def place_initial_orders(
        self,
        grid_levels: list["GridLevel"],
        current_price: float,
    ) -> int:
        """
        Coloca órdenes BUY solo en niveles por debajo del precio actual.

        Los niveles por encima se registran como 'sell_pending': el precio
        queda guardado en estado para que el reciclado pueda colocar la SELL
        correspondiente cuando una BUY se ejecute. No se intenta colocar SELL
        en el arranque porque no hay BTC disponible, solo USDT.

        Si ya existe estado persistido, omite la colocación inicial
        para evitar duplicar órdenes al reiniciar el bot.

        Returns:
            Número de órdenes BUY colocadas exitosamente.
        """
        if self._state.levels:
            logger.info("ordenes_iniciales_omitidas | estado_existente_cargado")
            return 0

        levels_data: list[dict[str, Any]] = []
        placed = 0

        for level in grid_levels:
            if level.price >= current_price:
                levels_data.append(self._level_dict(level, "sell", "sell_pending", None))
                logger.debug(
                    "nivel_reservado_para_sell | price=%.2f index=%d",
                    level.price,
                    level.index,
                )
                continue

            ok, reason = self._risk.can_place_order(
                level.order_size_quote, self._max_order_usdt
            )
            if not ok:
                logger.warning(
                    "orden_bloqueada_por_riesgo | reason=%s level=%d",
                    reason,
                    level.index,
                )
                levels_data.append(self._level_dict(level, "buy", "blocked", None))
                continue

            order = self._client.create_limit_order(
                symbol=self._symbol,
                side="buy",
                amount=level.order_size_base,
                price=level.price,
            )

            if order:
                levels_data.append(
                    self._level_dict(level, "buy", "buy_open", order["id"])
                )
                placed += 1
                logger.info(
                    "orden_inicial_colocada | symbol=%s side=buy price=%.2f "
                    "amount=%.8f order_id=%s",
                    self._symbol,
                    level.price,
                    level.order_size_base,
                    order["id"],
                )
            else:
                levels_data.append(self._level_dict(level, "buy", "error", None))

        grid_config = {
            "price_min": grid_levels[0].price if grid_levels else 0.0,
            "price_max": grid_levels[-1].price if grid_levels else 0.0,
            "num_levels": len(grid_levels),
        }
        self._state.initialize(
            symbol=self._symbol,
            grid_config=grid_config,
            levels=levels_data,
        )

        logger.info(
            "ordenes_iniciales_completas | buy_colocadas=%d sell_pendientes=%d total=%d",
            placed,
            sum(1 for l in levels_data if l["status"] == "sell_pending"),
            len(grid_levels),
        )
        return placed

    # ------------------------------------------------------------------
    # Monitoreo de órdenes ejecutadas
    # ------------------------------------------------------------------

    def check_filled_orders(
        self,
        current_price: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detecta órdenes que ya se ejecutaron.

        En DRY_RUN simula ejecuciones por cruce de precio:
          - BUY a precio P → filled cuando current_price <= P
            (el mercado bajó hasta el nivel de compra)
          - SELL a precio P → filled cuando current_price >= P
            (el mercado subió hasta el nivel de venta)
        Si current_price es None no se simula ningún fill.

        En modo real compara el estado local contra las órdenes
        abiertas en el exchange.

        Returns:
            Lista de niveles cuyas órdenes fueron ejecutadas.
        """
        if self._client.dry_run:
            if current_price is None:
                return []

            filled = []
            for level in self._state.levels:
                if not level.get("status", "").endswith("_open"):
                    continue
                order_price: float = level.get("price", 0.0)
                side: str = level.get("side", "")

                if side == "buy" and current_price <= order_price:
                    logger.info(
                        "cruce_detectado | side=buy order_price=%.2f current_price=%.2f "
                        "→ FILLED (precio bajo hasta el nivel de compra)",
                        order_price,
                        current_price,
                    )
                    filled.append(level)
                elif side == "sell" and current_price >= order_price:
                    logger.info(
                        "cruce_detectado | side=sell order_price=%.2f current_price=%.2f "
                        "→ FILLED (precio subió hasta el nivel de venta)",
                        order_price,
                        current_price,
                    )
                    filled.append(level)
                else:
                    logger.debug(
                        "sin_cruce | side=%s order_price=%.2f current_price=%.2f",
                        side,
                        order_price,
                        current_price,
                    )

            if filled:
                logger.info(
                    "ordenes_simuladas_ejecutadas | count=%d current_price=%.2f",
                    len(filled),
                    current_price,
                )
            return filled

        open_order_ids = {
            o["id"] for o in self._client.fetch_open_orders(self._symbol)
        }

        filled = []
        for level in self._state.levels:
            order_id = level.get("order_id")
            status = level.get("status", "")
            if order_id and status.endswith("_open") and order_id not in open_order_ids:
                filled.append(level)

        if filled:
            logger.info("ordenes_ejecutadas_detectadas | count=%d", len(filled))
        return filled

    # ------------------------------------------------------------------
    # Reciclado de órdenes
    # ------------------------------------------------------------------

    def recycle_order(self, filled_level: dict[str, Any]) -> dict[str, Any] | None:
        """
        Recicla una orden ejecutada colocando la orden opuesta en el nivel adyacente.

        BUY en N ejecutada  → SELL en N+1
        SELL en N ejecutada → BUY en N-1

        Returns:
            La nueva orden colocada, o None si no fue posible.
        """
        side: str = filled_level.get("side", "")
        index: int = filled_level.get("index", -1)
        amount: float = filled_level.get("amount", 0.0)

        if side == "buy":
            target_index = index + 1
            new_side = "sell"
        elif side == "sell":
            target_index = index - 1
            new_side = "buy"
        else:
            logger.warning("reciclar_orden_side_invalido | side=%s index=%d", side, index)
            return None

        target_levels = [l for l in self._state.levels if l["index"] == target_index]
        if not target_levels:
            logger.warning(
                "reciclar_orden_sin_nivel_objetivo | index=%d side=%s target=%d",
                index,
                side,
                target_index,
            )
            return None

        target_level = target_levels[0]
        new_price: float = target_level["price"]

        order = self._client.create_limit_order(
            symbol=self._symbol,
            side=new_side,
            amount=amount,
            price=new_price,
        )

        if order:
            self._state.update_level(target_index, f"{new_side}_open", order["id"])
            self._state.update_level(index, f"{side}_filled")
            logger.info(
                "orden_reciclada | filled_side=%s filled_index=%d "
                "new_side=%s new_price=%.2f new_order_id=%s",
                side,
                index,
                new_side,
                new_price,
                order["id"],
            )
            return order

        logger.error(
            "reciclar_orden_fallida | side=%s index=%d", new_side, target_index
        )
        return None

    # ------------------------------------------------------------------
    # Limpieza / shutdown
    # ------------------------------------------------------------------

    def cancel_all_orders(self) -> int:
        """
        Cancela todas las órdenes abiertas. Llamado en shutdown.

        Returns:
            Número de órdenes canceladas.
        """
        cancelled = 0
        for level in self._state.levels:
            order_id = level.get("order_id")
            status = level.get("status", "")
            if order_id and status.endswith("_open"):
                self._client.cancel_order(order_id, self._symbol)
                self._state.update_level(level["index"], "cancelled")
                cancelled += 1

        if cancelled > 0:
            self._state.save()

        logger.info("ordenes_canceladas | total=%d", cancelled)
        return cancelled

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _level_dict(
        level: "GridLevel",
        side: str,
        status: str,
        order_id: str | None,
    ) -> dict[str, Any]:
        return {
            "index": level.index,
            "price": level.price,
            "status": status,
            "order_id": order_id,
            "side": side,
            "amount": level.order_size_base,
        }
