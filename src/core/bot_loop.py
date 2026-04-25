"""
Orquestador principal del bot de grid trading.

Ejecuta ciclos de: lectura de precio → verificación de órdenes → reciclado
→ persistencia → sleep. Incluye circuit breaker y graceful shutdown.
"""
from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.core.order_manager import OrderManager
from src.core.pnl_tracker import PnLTracker
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
        pnl_tracker: PnLTracker | None = None,
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
        self._min_step_usdt: float = float(self._trailing_cfg.get("min_step_usdt", 600))
        # Ciclos de espera tras un recentrado fallido antes de volver a intentarlo.
        # Evita el loop infinito de reintentos cuando no hay USDT o hay error de red.
        self._cooldown_cycles_after_error: int = int(
            self._trailing_cfg.get("cooldown_cycles_after_error", 10)
        )
        self._recenter_cooldown_remaining: int = 0
        self._cycles_since_recenter: int = 0

        self._running = False
        self._total_cycles = 0
        self._last_no_usdt_notification: float | None = None
        self._last_daily_report_date: str | None = None

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

        self._pnl_tracker: PnLTracker = pnl_tracker or PnLTracker()

        if order_manager is None:
            order_manager = OrderManager(
                client=self._client,
                risk_manager=self._risk_manager,
                grid_state=self._grid_state,
                symbol=self._symbol,
                max_order_usdt=self._max_order_usdt,
                pnl_tracker=self._pnl_tracker,
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

    def _cancel_orphan_orders(self) -> int:
        """
        Cancela todas las órdenes abiertas en el exchange para el símbolo.

        Se llama antes de inicializar la grilla cuando no hay estado guardado,
        para liberar el USDT que pudiera estar bloqueado en órdenes de una
        sesión anterior cuyo estado se perdió (ej. redeploy sin volumen persistente).

        Returns:
            Número de órdenes canceladas.
        """
        if self._dry_run:
            return 0

        try:
            open_orders = self._client.fetch_open_orders(self._symbol)
            cancelled = 0
            for order in open_orders:
                order_id = order.get("id")
                if order_id:
                    self._client.cancel_order(order_id, self._symbol)
                    cancelled += 1

            if cancelled > 0:
                logger.warning(
                    "ordenes_huerfanas_canceladas",
                    symbol=self._symbol,
                    total=cancelled,
                )
                self._notifier.send(
                    f"\u26a0\ufe0f {cancelled} orden(es) huérfana(s) canceladas al reiniciar\n"
                    f"Símbolo: {self._symbol}\n"
                    f"El estado previo se perdió — iniciando grilla desde cero."
                )
            return cancelled

        except Exception as exc:
            logger.warning(
                "error_cancelando_ordenes_huerfanas",
                error=str(exc),
            )
            return 0

    def _initialize_grid(self) -> bool:
        """
        Calcula niveles y coloca órdenes iniciales.

        Secuencia:
        1. Cancela órdenes huérfanas abiertas en el exchange (libera USDT).
        2. Consulta el balance: USDT para BUYs, moneda base (ej. BTC) para SELLs.
        3. Si hay moneda base disponible (BTC acumulado de sesiones anteriores),
           coloca SELLs reales en niveles sobre el precio actual para retomar
           la venta del capital ya invertido.
        4. Coloca BUYs con el USDT disponible en niveles bajo el precio actual.

        Returns:
            True si la inicialización fue exitosa.
        """
        try:
            self._cancel_orphan_orders()

            # Consultar balance para detectar BTC disponible (SELLs) y USDT disponible (BUYs).
            # Se hace siempre —incluso en DRY_RUN— porque fetch_balance es lectura pura
            # y es la única forma de saber si hay BTC acumulado de sesiones anteriores.
            base_available = 0.0
            try:
                balance = self._client.fetch_balance()
                base_currency = self._symbol.split("/")[0]  # "BTC" de "BTC/USDT"
                base_available = float(balance.get("free", {}).get(base_currency, 0.0))
                if base_available > 0:
                    logger.warning(
                        "balance_base_detectado_al_iniciar",
                        moneda=base_currency,
                        disponible=round(base_available, 8),
                    )
                    self._notifier.send(
                        f"\u26a0\ufe0f {base_currency} detectado al iniciar\n"
                        f"Disponible: {base_available:.8f} {base_currency}\n"
                        f"Se colocarán órdenes SELL encima del precio actual."
                    )
            except Exception as exc:
                logger.warning("error_consultando_balance_al_iniciar", error=str(exc))

            calculator = GridCalculator(
                price_min=self._price_min,
                price_max=self._price_max,
                num_levels=self._grid_cfg["num_levels"],
                total_capital_usdt=self._grid_cfg["total_capital_usdt"],
                max_order_usdt=self._max_order_usdt,
            )
            grid_levels = calculator.calculate()

            current_price = self._price_reader.get_current_price()
            placed = self._order_manager.place_initial_orders(
                grid_levels, current_price, base_available=base_available
            )

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

        Antes de ejecutar el recentrado verifica que haya USDT disponible para
        colocar al menos 1 orden nueva. Si todo el capital está en BTC comprado
        (SELLs pendientes), postpone el recentrado para evitar gastar capital
        inexistente.

        Args:
            current_price: Precio actual del mercado.
            reason: "out_of_range" | "cycles"

        Returns:
            True si el recentrado fue exitoso.
        """
        try:
            min_order_usdt: float = self._risk_cfg.get("min_order_usdt", 0.5)
            base_currency_rc = self._symbol.split("/")[0]

            try:
                balance = self._client.fetch_balance()
                usdt_available: float = float(
                    balance.get("free", {}).get("USDT", 0.0)
                )
                base_available_rc: float = float(
                    balance.get("free", {}).get(base_currency_rc, 0.0)
                )
            except Exception as exc:
                logger.warning(
                    "error_consultando_balance_para_recentrado",
                    error=str(exc),
                )
                usdt_available = 0.0
                base_available_rc = 0.0

            if base_available_rc > 0:
                logger.info(
                    "btc_detectado_en_recentrado",
                    moneda=base_currency_rc,
                    disponible=round(base_available_rc, 8),
                )

            if usdt_available < min_order_usdt and base_available_rc == 0.0:
                logger.warning(
                    "recentrado_bloqueado_sin_usdt",
                    usdt_disponible=round(usdt_available, 4),
                    minimo_requerido=min_order_usdt,
                    motivo=reason,
                    precio=round(current_price, 2),
                )
                _now = time.time()
                _throttle_seconds = 21600  # 6 horas
                if (
                    self._last_no_usdt_notification is None
                    or _now - self._last_no_usdt_notification >= _throttle_seconds
                ):
                    self._notifier.notify_no_usdt_available(
                        usdt_available=round(usdt_available, 4),
                        required=min_order_usdt,
                    )
                    self._last_no_usdt_notification = _now
                return False

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

            placed = self._order_manager.place_initial_orders(
                new_levels, current_price, base_available=base_available_rc
            )
            self._cycles_since_recenter = 0
            self._last_no_usdt_notification = None

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
    # Reporte diario
    # ------------------------------------------------------------------

    def _send_daily_report(self) -> None:
        """
        Envía el resumen diario de PnL a Telegram una vez por día (00:00 UTC).

        Se llama en cada ciclo; internamente verifica si ya se envió hoy.
        El reporte incluye trades ganadores/perdedores, profit bruto, fees y neto.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if self._last_daily_report_date == today:
            return

        # Primer ciclo del día: calcular resumen del día ANTERIOR
        # (today es el nuevo día; los ciclos de hoy aún no terminaron)
        self._last_daily_report_date = today

        summary = self._pnl_tracker.get_summary()
        if summary.total_cycles == 0:
            logger.info("reporte_diario_omitido", motivo="sin_ciclos_completados")
            return

        # Determinar trades ganadores y perdedores del día anterior
        yesterday = None
        try:
            from datetime import timedelta
            yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        except Exception:
            pass

        winning = 0
        losing = 0
        daily_gross = 0.0
        daily_fees = 0.0
        daily_net = 0.0

        for cycle in self._pnl_tracker._cycles:
            cycle_date = cycle.completed_at[:10]  # YYYY-MM-DD
            if cycle_date != yesterday:
                continue
            daily_gross += cycle.gross_profit
            daily_fees += cycle.total_fees
            daily_net += cycle.net_profit
            if cycle.net_profit >= 0:
                winning += 1
            else:
                losing += 1

        if winning == 0 and losing == 0:
            logger.info(
                "reporte_diario_omitido",
                motivo="sin_ciclos_completados_ayer",
                fecha=yesterday,
            )
            return

        logger.info(
            "reporte_diario_enviado",
            fecha=yesterday,
            winning=winning,
            losing=losing,
            gross=round(daily_gross, 4),
            fees=round(daily_fees, 4),
            net=round(daily_net, 4),
        )
        self._notifier.notify_daily_summary(
            winning_trades=winning,
            losing_trades=losing,
            gross_profit=round(daily_gross, 4),
            total_fees=round(daily_fees, 4),
            net_profit=round(daily_net, 4),
        )

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
            self._send_daily_report()

            current_price = self._price_reader.get_current_price()
            self._cycles_since_recenter += 1

            # El cooldown cuenta hacia atrás en cada ciclo, sin importar si el precio
            # está en rango o no. Cuando llega a 0, se vuelve a intentar el recentrado.
            if self._recenter_cooldown_remaining > 0:
                self._recenter_cooldown_remaining -= 1

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
                if self._recenter_cooldown_remaining > 0:
                    logger.info(
                        "recentrado_pospuesto_por_cooldown",
                        ciclos_restantes=self._recenter_cooldown_remaining,
                        motivo="out_of_range",
                    )
                else:
                    success = self._recenter_grid(current_price, "out_of_range")
                    if not success:
                        self._recenter_cooldown_remaining = self._cooldown_cycles_after_error
                        logger.warning(
                            "recentrado_fallido_activando_cooldown",
                            cooldown_ciclos=self._cooldown_cycles_after_error,
                        )
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
                if self._recenter_cooldown_remaining > 0:
                    logger.info(
                        "recentrado_pospuesto_por_cooldown",
                        ciclos_restantes=self._recenter_cooldown_remaining,
                        motivo="cycles",
                    )
                else:
                    success = self._recenter_grid(current_price, "cycles")
                    if not success:
                        self._recenter_cooldown_remaining = self._cooldown_cycles_after_error
                        logger.warning(
                            "recentrado_fallido_activando_cooldown",
                            cooldown_ciclos=self._cooldown_cycles_after_error,
                        )
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
