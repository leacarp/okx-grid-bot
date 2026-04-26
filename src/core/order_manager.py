"""
Gestión de órdenes: colocación inicial, monitoreo y reciclado.

La lógica de reciclado:
  - BUY en nivel N ejecutada  → coloca SELL en nivel N+1
  - SELL en nivel N ejecutada → coloca BUY en nivel N-1
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient
    from src.core.pnl_tracker import PnLTracker
    from src.risk.risk_manager import RiskManager
    from src.strategy.grid_calculator import GridLevel
    from src.strategy.grid_state import GridState

logger = structlog.get_logger(__name__)

# OKX minimum order size for BTC/USDT (exchange hard limit)
_MIN_ORDER_BTC = 0.00001


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
        pnl_tracker: "PnLTracker | None" = None,
    ) -> None:
        self._client = client
        self._risk = risk_manager
        self._state = grid_state
        self._symbol = symbol
        self._max_order_usdt = max_order_usdt
        self._pnl_tracker = pnl_tracker

    # ------------------------------------------------------------------
    # Colocación inicial
    # ------------------------------------------------------------------

    def place_initial_orders(
        self,
        grid_levels: list["GridLevel"],
        current_price: float,
        base_available: float = 0.0,
        symbol: str | None = None,
    ) -> int:
        """
        Coloca órdenes BUY en niveles por debajo del precio actual y SELL en
        niveles por encima usando el balance de la moneda base disponible.

        Cuando base_available > 0 (ej. BTC ya comprado de sesiones anteriores),
        lo distribuye equitativamente entre los niveles de venta y coloca órdenes
        SELL reales en lugar de marcarlos como 'sell_pending'. Esto permite que el
        bot retome la venta del BTC acumulado tras un reinicio con estado perdido.

        Si ya existe estado persistido, omite la colocación inicial
        para evitar duplicar órdenes al reiniciar el bot.

        Args:
            grid_levels: Niveles calculados por GridCalculator.
            current_price: Precio actual del mercado.
            base_available: Balance libre de la moneda base (ej. BTC) para SELLs.

        Returns:
            Número de órdenes colocadas exitosamente (BUYs + SELLs).
        """
        _sym = symbol or self._symbol

        if self._state.levels:
            logger.info("ordenes_iniciales_omitidas", motivo="estado_existente_cargado")
            return 0

        levels_data: list[dict[str, Any]] = []
        placed_buys = 0
        placed_sells = 0

        # Distribuir el BTC disponible entre los niveles de venta
        sell_levels = [l for l in grid_levels if l.price >= current_price]
        btc_per_sell = (base_available / len(sell_levels)) if (sell_levels and base_available > 0) else 0.0
        btc_remaining = base_available

        # Verificar que cada orden SELL supere el mínimo de OKX antes de intentar colocarlas.
        if btc_per_sell > 0 and btc_per_sell < _MIN_ORDER_BTC:
            valor_aprox_usdt = btc_per_sell * current_price
            logger.warning(
                "btc_por_nivel_debajo_del_minimo_okx",
                btc_por_nivel=round(btc_per_sell, 8),
                min_btc=_MIN_ORDER_BTC,
                valor_aprox_usdt=round(valor_aprox_usdt, 4),
                total_btc=round(base_available, 8),
                niveles_sell=len(sell_levels),
            )
            btc_per_sell = 0.0
            btc_remaining = 0.0

        if btc_per_sell > 0:
            logger.info(
                "btc_disponible_para_sells",
                amount=round(base_available, 8),
                niveles_sell=len(sell_levels),
                btc_por_nivel=round(btc_per_sell, 8),
            )

        open_count = 0  # contador de órdenes abiertas durante la inicialización

        for level in grid_levels:
            # Si el nivel tiene side pre-asignado (grilla asimétrica), respetarlo.
            # Si no (RANGING), decidir por precio como siempre.
            is_sell_level = (
                level.side == "sell"
                if level.side
                else level.price >= current_price
            )
            if is_sell_level:
                if btc_per_sell > 0 and btc_remaining >= btc_per_sell:
                    order = self._client.create_limit_order(
                        symbol=_sym,
                        side="sell",
                        amount=btc_per_sell,
                        price=level.price,
                    )
                    if order:
                        entry = self._level_dict(level, "sell", "sell_open", order["id"])
                        entry["amount"] = btc_per_sell
                        levels_data.append(entry)
                        btc_remaining -= btc_per_sell
                        placed_sells += 1
                        open_count += 1
                        logger.info(
                            "orden_inicial_colocada",
                            symbol=_sym,
                            side="sell",
                            price=level.price,
                            amount=round(btc_per_sell, 8),
                            order_id=order["id"],
                        )
                        continue

                # No hay BTC disponible o la orden falló → marcar como sell_pending
                levels_data.append(self._level_dict(level, "sell", "sell_pending", None))
                logger.debug(
                    "nivel_reservado_para_sell",
                    price=level.price,
                    index=level.index,
                )
                continue

            ok, reason = self._risk.pre_order_check(
                order_size_usdt=level.order_size_quote,
                available_balance=None,
                open_order_count=open_count,
            )
            if not ok:
                logger.warning(
                    "orden_bloqueada_por_riesgo",
                    reason=reason,
                    level=level.index,
                )
                levels_data.append(self._level_dict(level, "buy", "blocked", None))
                continue

            order = self._client.create_limit_order(
                symbol=_sym,
                side="buy",
                amount=level.order_size_base,
                price=level.price,
            )

            if order:
                levels_data.append(
                    self._level_dict(level, "buy", "buy_open", order["id"])
                )
                placed_buys += 1
                open_count += 1
                logger.info(
                    "orden_inicial_colocada",
                    symbol=_sym,
                    side="buy",
                    price=level.price,
                    amount=round(level.order_size_base, 8),
                    order_id=order["id"],
                )
            else:
                levels_data.append(self._level_dict(level, "buy", "error", None))

        grid_config = {
            "price_min": grid_levels[0].price if grid_levels else 0.0,
            "price_max": grid_levels[-1].price if grid_levels else 0.0,
            "num_levels": len(grid_levels),
        }
        self._state.initialize(
            symbol=_sym,
            grid_config=grid_config,
            levels=levels_data,
        )

        logger.info(
            "ordenes_iniciales_completas",
            buy_colocadas=placed_buys,
            sell_colocadas=placed_sells,
            sell_pendientes=sum(1 for l in levels_data if l["status"] == "sell_pending"),
            total=len(grid_levels),
        )
        return placed_buys + placed_sells

    # ------------------------------------------------------------------
    # Monitoreo de órdenes ejecutadas
    # ------------------------------------------------------------------

    def check_filled_orders(
        self,
        current_price: float | None = None,
        symbol: str | None = None,
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
        _sym = symbol or self._symbol

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
                        "cruce_detectado",
                        side="buy",
                        order_price=order_price,
                        current_price=current_price,
                        resultado="FILLED",
                    )
                    filled.append(level)
                elif side == "sell" and current_price >= order_price:
                    logger.info(
                        "cruce_detectado",
                        side="sell",
                        order_price=order_price,
                        current_price=current_price,
                        resultado="FILLED",
                    )
                    filled.append(level)
                else:
                    logger.debug(
                        "sin_cruce",
                        side=side,
                        order_price=order_price,
                        current_price=current_price,
                    )

            if filled:
                logger.info(
                    "ordenes_simuladas_ejecutadas",
                    count=len(filled),
                    current_price=current_price,
                )
            return filled

        open_order_ids = {
            o["id"] for o in self._client.fetch_open_orders(_sym)
        }

        filled = []
        for level in self._state.levels:
            order_id = level.get("order_id")
            status = level.get("status", "")
            if order_id and status.endswith("_open") and order_id not in open_order_ids:
                filled.append(level)

        if filled:
            logger.info("ordenes_ejecutadas_detectadas", count=len(filled))
        return filled

    # ------------------------------------------------------------------
    # Reciclado de órdenes
    # ------------------------------------------------------------------

    def recycle_order(self, filled_level: dict[str, Any]) -> dict[str, Any] | None:
        """
        Recicla una orden ejecutada colocando la orden opuesta en el nivel adyacente.

        BUY en N ejecutada  → SELL en N+1
        SELL en N ejecutada → BUY en N-1 (y registra el ciclo en PnLTracker)

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
            logger.warning("reciclar_orden_side_invalido", side=side, index=index)
            return None

        target_levels = [l for l in self._state.levels if l["index"] == target_index]
        if not target_levels:
            logger.warning(
                "reciclar_orden_sin_nivel_objetivo",
                index=index,
                side=side,
                target=target_index,
            )
            return None

        target_level = target_levels[0]
        new_price: float = target_level["price"]
        previous_target_order_id: str | None = target_level.get("order_id")

        # Verificar risk antes de colocar la nueva orden
        open_count = sum(
            1 for l in self._state.levels
            if l.get("status", "").endswith("_open") and l["index"] != index
        )
        ok, reason = self._risk.pre_order_check(
            order_size_usdt=amount * new_price,
            available_balance=None,
            open_order_count=open_count,
        )
        if not ok:
            logger.warning(
                "orden_reciclada_bloqueada_por_riesgo",
                reason=reason,
                side=new_side,
                target_index=target_index,
            )
            return None

        order = self._client.create_limit_order(
            symbol=self._symbol,
            side=new_side,
            amount=amount,
            price=new_price,
        )

        if order:
            self._state.update_level(target_index, f"{new_side}_open", order["id"])
            self._state.update_level(index, f"{side}_filled")

            # Cuando SELL se ejecuta → ciclo completo (BUY anterior + esta SELL)
            # Se registra PnL real: intenta usar fills del exchange y, si falla,
            # cae a un cálculo conservador con precio esperado.
            if side == "sell" and self._pnl_tracker is not None:
                buy_fill = self._build_fill_record(
                    order_id=previous_target_order_id,
                    side="buy",
                    fallback_order_id=f"buy_{target_index}",
                    fallback_amount=amount,
                    fallback_price=new_price,
                )
                sell_fill = self._build_fill_record(
                    order_id=filled_level.get("order_id"),
                    side="sell",
                    fallback_order_id=f"sell_{index}",
                    fallback_amount=amount,
                    fallback_price=filled_level["price"],
                )
                cycle = self._pnl_tracker.record_cycle(buy_fill, sell_fill)
                self._state.record_profit(cycle.net_profit)
                logger.info(
                    "ciclo_completado_pnl",
                    symbol=self._symbol,
                    buy_price=buy_fill.price,
                    sell_price=sell_fill.price,
                    net_profit=cycle.net_profit,
                )

            logger.info(
                "orden_reciclada",
                filled_side=side,
                filled_index=index,
                new_side=new_side,
                new_price=new_price,
                new_order_id=order["id"],
            )
            return order

        logger.error(
            "reciclar_orden_fallida",
            side=new_side,
            target_index=target_index,
        )
        return None

    def _build_fill_record(
        self,
        order_id: str | None,
        side: str,
        fallback_order_id: str,
        fallback_amount: float,
        fallback_price: float,
    ) -> Any:
        """
        Construye un FillRecord usando datos reales del exchange cuando estén disponibles.

        En modo real intenta leer trades por order_id; si no puede, hace fallback
        al precio/cantidad esperados para no romper el flujo de reciclado.
        """
        price = fallback_price
        amount = fallback_amount
        fee = 0.0
        effective_order_id = order_id or fallback_order_id

        if not self._client.dry_run and order_id:
            try:
                trade_data = self._client.fetch_order_trades(order_id, self._symbol)
                if isinstance(trade_data, dict):
                    trade_price = float(trade_data.get("price", 0.0) or 0.0)
                    trade_amount = float(trade_data.get("amount", 0.0) or 0.0)
                    trade_fee = float(trade_data.get("fee", 0.0) or 0.0)
                    if trade_price > 0:
                        price = trade_price
                    if trade_amount > 0:
                        amount = trade_amount
                    if trade_fee >= 0:
                        fee = trade_fee
            except Exception as exc:
                logger.warning(
                    "error_obteniendo_fill_real",
                    side=side,
                    order_id=order_id,
                    symbol=self._symbol,
                    error=str(exc),
                )

        return self._pnl_tracker.record_fill(
            order_id=effective_order_id,
            symbol=self._symbol,
            side=side,
            amount=amount,
            price=price,
            fee=fee,
        )

    # ------------------------------------------------------------------
    # Limpieza / shutdown
    # ------------------------------------------------------------------

    def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancela todas las órdenes abiertas. Llamado en shutdown o al cambiar de par.

        Args:
            symbol: Símbolo a usar al cancelar. Si None usa self._symbol.

        Returns:
            Número de órdenes canceladas.
        """
        _sym = symbol or self._symbol
        cancelled = 0
        for level in self._state.levels:
            order_id = level.get("order_id")
            status = level.get("status", "")
            if order_id and status.endswith("_open"):
                self._client.cancel_order(order_id, _sym)
                self._state.update_level(level["index"], "cancelled")
                cancelled += 1

        if cancelled > 0:
            self._state.save()

        logger.info("ordenes_canceladas", total=cancelled)
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
