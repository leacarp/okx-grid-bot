"""
Orquestador principal del bot de grid trading.

Ejecuta ciclos de: lectura de precio → verificación de órdenes → reciclado
→ persistencia → sleep. Incluye circuit breaker y graceful shutdown.
"""
from __future__ import annotations

import logging
import signal
import time
from typing import TYPE_CHECKING, Any

from src.core.order_manager import OrderManager
from src.core.price_reader import PriceReader
from src.risk.risk_manager import RiskManager
from src.strategy.grid_calculator import GridCalculator
from src.strategy.grid_state import GridState
from src.utils.logger import get_logger
from src.utils.notifier import TelegramNotifier

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient

logger = get_logger("bot_loop")


class BotLoop:
    """
    Loop principal del bot. Acepta dependencias inyectadas para facilitar
    el testing; si no se proveen, las crea desde el entorno/config.
    """

    def __init__(
        self,
        config: dict[str, Any],
        client: "OKXClient | None" = None,
        price_reader: PriceReader | None = None,
        risk_manager: RiskManager | None = None,
        grid_state: GridState | None = None,
        order_manager: OrderManager | None = None,
        notifier: TelegramNotifier | None = None,
    ) -> None:
        self._config = config
        self._grid_cfg = config["grid"]
        self._risk_cfg = config["risk"]
        self._loop_cfg = config["loop"]

        self._symbol: str = self._grid_cfg["symbol"]
        self._interval: float = self._loop_cfg["interval_seconds"]
        self._price_min: float = self._grid_cfg["price_min"]
        self._price_max: float = self._grid_cfg["price_max"]
        self._max_order_usdt: float = self._grid_cfg["max_order_usdt"]
        self._dry_run: bool = config.get("dry_run", True)

        # Trailing grid — parámetros fijos de referencia (nunca cambian)
        self._trailing_cfg = config.get("trailing", {})
        self._original_range: float = self._grid_cfg["price_max"] - self._grid_cfg["price_min"]
        self._original_num_levels: int = self._grid_cfg["num_levels"]
        self._recenter_every_cycles: int = self._trailing_cfg.get("recenter_every_cycles", 200)
        self._min_step_usdt: float = float(self._trailing_cfg.get("min_step_usdt", 750))
        self._cycles_since_recenter: int = 0

        self._running = False
        self._total_cycles = 0

        # Construcción de dependencias (inyectables para tests)
        if client is None:
            from src.connectors.okx_client import OKXClient as _OKXClient
            client = _OKXClient.from_env(
                sandbox=config["exchange"].get("sandbox", False)
            )
        self._client = client

        self._price_reader = price_reader or PriceReader(self._client, self._symbol)
        self._risk_manager = risk_manager or RiskManager(self._risk_cfg, self._loop_cfg)
        self._grid_state = grid_state or GridState()

        if order_manager is None:
            order_manager = OrderManager(
                client=self._client,
                risk_manager=self._risk_manager,
                grid_state=self._grid_state,
                symbol=self._symbol,
                max_order_usdt=self._max_order_usdt,
            )
        self._order_manager = order_manager
        self._notifier: TelegramNotifier = notifier or TelegramNotifier()

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Señales del sistema operativo
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Graceful shutdown al recibir SIGINT (Ctrl+C) o SIGTERM."""
        logger.info("señal_recibida", signum=signum)
        self._running = False

    # ------------------------------------------------------------------
    # Inicialización de la grilla
    # ------------------------------------------------------------------

    def _initialize_grid(self) -> bool:
        """
        Calcula niveles y coloca órdenes iniciales.

        Returns:
            True si la inicialización fue exitosa.
        """
        try:
            calculator = GridCalculator(
                price_min=self._price_min,
                price_max=self._price_max,
                num_levels=self._grid_cfg["num_levels"],
                total_capital_usdt=self._grid_cfg["total_capital_usdt"],
                max_order_usdt=self._max_order_usdt,
            )
            grid_levels = calculator.calculate()

            current_price = self._price_reader.get_current_price()
            placed = self._order_manager.place_initial_orders(grid_levels, current_price)

            logger.info(
                "grid_inicializada",
                levels=len(grid_levels),
                orders_placed=placed,
                dry_run=self._dry_run,
            )
            return True

        except Exception as exc:
            logger.error("error_inicializando_grid", error=str(exc), exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Trailing grid — recentrado
    # ------------------------------------------------------------------

    def _recenter_grid(self, current_price: float, reason: str) -> bool:
        """
        Recentra la grilla alrededor del precio actual.

        Cancela todas las órdenes abiertas, calcula un nuevo rango manteniendo
        el mismo ancho total que el config original, ajusta num_levels para
        respetar min_step_usdt, y coloca las nuevas órdenes.

        Args:
            current_price: Precio actual del mercado.
            reason: "out_of_range" | "cycles"

        Returns:
            True si el recentrado fue exitoso.
        """
        try:
            rango = self._original_range
            nuevo_min = current_price - rango / 2
            nuevo_max = current_price + rango / 2

            # Ajustar num_levels para que step >= min_step_usdt
            # step = rango / (num_levels - 1) → num_levels <= rango/min_step + 1
            num_levels = self._original_num_levels
            if self._min_step_usdt > 0 and num_levels > 1:
                max_levels = max(2, int(rango / self._min_step_usdt) + 1)
                num_levels = min(num_levels, max_levels)

            step = rango / (num_levels - 1) if num_levels > 1 else rango

            self._order_manager.cancel_all_orders()
            self._grid_state.clear_levels()

            calculator = GridCalculator(
                price_min=nuevo_min,
                price_max=nuevo_max,
                num_levels=num_levels,
                total_capital_usdt=self._grid_cfg["total_capital_usdt"],
                max_order_usdt=self._max_order_usdt,
            )
            new_levels = calculator.calculate()

            self._price_min = nuevo_min
            self._price_max = nuevo_max

            placed = self._order_manager.place_initial_orders(new_levels, current_price)
            self._cycles_since_recenter = 0

            logger.info(
                "grilla_recentrada",
                motivo=reason,
                precio=round(current_price, 2),
                nuevo_min=round(nuevo_min, 2),
                nuevo_max=round(nuevo_max, 2),
                step=round(step, 2),
                num_levels=num_levels,
                ordenes_colocadas=placed,
            )
            self._notifier.notify_grid_recentered(
                reason=reason,
                current_price=round(current_price, 2),
                new_min=round(nuevo_min, 2),
                new_max=round(nuevo_max, 2),
                step=round(step, 2),
            )
            return True

        except Exception as exc:
            logger.error("error_recentrando_grilla", error=str(exc), exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Ciclo principal
    # ------------------------------------------------------------------

    def _run_cycle(self) -> bool:
        """
        Ejecuta un ciclo completo del bot:

        1. Lee precio actual.
        2. Verifica si está en rango.
        3. Detecta órdenes ejecutadas.
        4. Recicla órdenes ejecutadas.
        5. Persiste estado.

        Returns:
            True si el ciclo fue exitoso, False si hubo error.
        """
        try:
            current_price = self._price_reader.get_current_price()
            self._cycles_since_recenter += 1

            # TRIGGER 1 — precio fuera de rango: recentrar inmediatamente
            in_range = self._price_reader.is_price_in_range(
                current_price, self._price_min, self._price_max
            )
            if not in_range:
                logger.warning(
                    "precio_fuera_de_rango",
                    price=round(current_price, 2),
                    price_min=round(self._price_min, 2),
                    price_max=round(self._price_max, 2),
                )
                self._recenter_grid(current_price, "out_of_range")
                self._total_cycles += 1
                return True

            # TRIGGER 2 — por ciclos: recentrar aunque el precio esté en rango
            if self._cycles_since_recenter >= self._recenter_every_cycles:
                logger.info(
                    "recentrado_por_ciclos",
                    ciclos_desde_recentrado=self._cycles_since_recenter,
                    recenter_every=self._recenter_every_cycles,
                    price=round(current_price, 2),
                )
                self._recenter_grid(current_price, "cycles")
                self._total_cycles += 1
                return True

            filled_orders = self._order_manager.check_filled_orders(current_price)
            for filled in filled_orders:
                self._order_manager.recycle_order(filled)

            self._grid_state.save()
            self._risk_manager.reset_errors()
            self._total_cycles += 1

            logger.info(
                "ciclo_completado",
                ciclo=self._total_cycles,
                price=round(current_price, 2),
                filled=len(filled_orders),
                dry_run=self._dry_run,
            )

            if filled_orders:
                self._notifier.notify_orders_filled(
                    symbol=self._symbol,
                    filled_count=len(filled_orders),
                    current_price=round(current_price, 2),
                    cycle=self._total_cycles,
                    dry_run=self._dry_run,
                )

            return True

        except Exception as exc:
            error_msg = str(exc)
            logger.error("error_en_ciclo", error=error_msg, exc_info=True)
            self._notifier.notify_critical_error(error_msg, self._total_cycles)

            circuit_triggered = self._risk_manager.register_error()
            if circuit_triggered:
                logger.critical("circuit_breaker_activado")
                self._notifier.notify_circuit_breaker(
                    self._risk_manager.consecutive_errors
                )
                self._running = False
            return False

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------

    def run(self, max_cycles: int = 0) -> None:
        """
        Ejecuta el loop principal hasta que:
        - Se recibe SIGINT/SIGTERM.
        - Se activa el circuit breaker.
        - Se alcanza max_cycles (si max_cycles > 0).

        Args:
            max_cycles: Límite de ciclos. 0 = sin límite (producción).
        """
        logger.info(
            "bot_loop_iniciado",
            symbol=self._symbol,
            dry_run=self._dry_run,
            interval_seconds=int(self._interval),
            max_cycles=max_cycles,
        )
        self._notifier.notify_bot_started(self._symbol, self._dry_run)

        loaded = self._grid_state.load()
        if not loaded:
            if not self._initialize_grid():
                logger.critical("bot_loop_abortado | fallo_en_inicializacion")
                return

        self._running = True

        while self._running:
            self._run_cycle()

            if self._risk_manager.circuit_open:
                break

            if max_cycles > 0 and self._total_cycles >= max_cycles:
                logger.info("max_ciclos_alcanzado", ciclos=self._total_cycles)
                break

            if self._running:
                time.sleep(self._interval)

        self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Cancela órdenes abiertas, persiste el estado final y marca el loop como detenido."""
        self._running = False
        logger.info("bot_loop_deteniendo", ciclos_completados=self._total_cycles)
        cancelled = self._order_manager.cancel_all_orders()
        logger.info(
            "bot_loop_detenido",
            ordenes_canceladas=cancelled,
            profit_total_usdt=round(self._grid_state.total_profit, 8),
        )

    # ------------------------------------------------------------------
    # Propiedades para inspección externa
    # ------------------------------------------------------------------

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    @property
    def is_running(self) -> bool:
        return self._running
