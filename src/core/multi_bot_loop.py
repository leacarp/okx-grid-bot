"""
Orquestador multi-token con rotación round-robin.

Cada token se opera durante PAIR_COOLDOWN_SECONDS (por defecto 2 horas) antes de
rotar al siguiente. Al cambiar de par:
  1. Verifica que hay precio disponible para el siguiente token (pre-flight).
  2. Cancela la grilla actual.
  3. Inicializa la grilla del nuevo token (desde estado guardado o desde cero).
  4. Si la inicialización falla → intenta recuperar el token anterior.

No incluye MarketAnalyzer ni PairSelector (Fase 3).
"""
from __future__ import annotations

import signal
import time
from typing import TYPE_CHECKING, Any

from src.core.order_manager import OrderManager
from src.core.pnl_tracker import PnLTracker
from src.core.price_reader import PriceReader
from src.risk.risk_manager import RiskManager
from src.strategy.grid_calculator import GridCalculator
from src.strategy.grid_state import GridState, state_file_for_symbol
from src.utils.logger import get_logger
from src.utils.notifier import TelegramNotifier

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient

logger = get_logger("multi_bot_loop")

PAIR_COOLDOWN_SECONDS: int = 7200  # 2 horas mínimo entre cambios de par


class MultiBotLoop:
    """
    Loop multi-token que itera en round-robin sobre config["tokens"].

    Cooldown mínimo de 2 horas entre cambios de par. Atomicidad garantizada:
    si la nueva grilla falla, intenta recuperar el token anterior.
    """

    def __init__(
        self,
        config: dict[str, Any],
        client: "OKXClient | None" = None,
        notifier: TelegramNotifier | None = None,
        pnl_tracker: PnLTracker | None = None,
    ) -> None:
        self._config = config
        self._tokens: list[dict[str, Any]] = config["tokens"]
        self._grid_cfg: dict[str, Any] = config["grid"]
        self._risk_cfg: dict[str, Any] = config["risk"]
        self._loop_cfg: dict[str, Any] = config["loop"]
        self._dry_run: bool = config.get("dry_run", True)
        self._interval: float = float(self._loop_cfg["interval_seconds"])
        self._pair_cooldown: float = float(
            config.get("multi", {}).get("pair_cooldown_seconds", PAIR_COOLDOWN_SECONDS)
        )

        self._current_idx: int = 0
        self._last_switch_time: float = 0.0
        self._running: bool = False
        self._total_cycles: int = 0

        # Componentes del token activo (asignados en _initialize_token)
        self._current_symbol: str | None = None
        self._current_grid_state: GridState | None = None
        self._current_order_manager: OrderManager | None = None
        self._current_price_reader: PriceReader | None = None
        self._price_min: float = 0.0
        self._price_max: float = 0.0

        if client is None:
            from src.connectors.okx_client import OKXClient as _OKXClient

            client = _OKXClient.from_env(
                sandbox=config["exchange"].get("sandbox", False)
            )
        self._client = client

        self._notifier: TelegramNotifier = notifier or TelegramNotifier()
        self._pnl_tracker: PnLTracker = pnl_tracker or PnLTracker()
        self._risk_manager = RiskManager(self._risk_cfg, self._loop_cfg)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Señales del SO
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("señal_recibida", signum=signum)
        self._running = False

    # ------------------------------------------------------------------
    # Construcción de componentes por símbolo
    # ------------------------------------------------------------------

    def _build_components(
        self, symbol: str
    ) -> tuple[GridState, OrderManager, PriceReader]:
        """Instancia GridState, OrderManager y PriceReader para el símbolo dado."""
        gs = GridState(state_file=state_file_for_symbol(symbol))
        pr = PriceReader(self._client, symbol)
        om = OrderManager(
            client=self._client,
            risk_manager=self._risk_manager,
            grid_state=gs,
            symbol=symbol,
            max_order_usdt=self._grid_cfg["max_order_usdt"],
            pnl_tracker=self._pnl_tracker,
        )
        return gs, om, pr

    def _compute_range(self, price: float) -> tuple[float, float]:
        """Calcula price_min y price_max aplicando range_width_pct al precio actual."""
        pct = self._grid_cfg["range_width_pct"]
        half = price * pct / 200.0
        return price - half, price + half

    # ------------------------------------------------------------------
    # Inicialización de token
    # ------------------------------------------------------------------

    def _initialize_token(self, token_cfg: dict[str, Any]) -> bool:
        """
        Carga o crea la grilla para el token especificado.

        Si existe estado guardado, lo carga sin re-colocar órdenes.
        Si no, calcula el rango dinámicamente y coloca órdenes iniciales.

        Returns:
            True si la inicialización fue exitosa.
        """
        symbol: str = token_cfg["symbol"]
        gs, om, pr = self._build_components(symbol)

        try:
            current_price = pr.get_current_price()
        except Exception as exc:
            logger.error(
                "error_precio_al_inicializar_token",
                symbol=symbol,
                error=str(exc),
            )
            return False

        price_min, price_max = self._compute_range(current_price)
        num_levels: int = token_cfg["num_levels"]

        calculator = GridCalculator(
            price_min=price_min,
            price_max=price_max,
            num_levels=num_levels,
            total_capital_usdt=self._grid_cfg["total_capital_usdt"],
            max_order_usdt=self._grid_cfg["max_order_usdt"],
        )
        grid_levels = calculator.calculate()

        loaded = gs.load()
        if not loaded:
            try:
                om.place_initial_orders(grid_levels, current_price)
            except Exception as exc:
                logger.error(
                    "error_colocando_ordenes_iniciales_multi",
                    symbol=symbol,
                    error=str(exc),
                )
                return False

        # Actualizar estado activo solo si todo salió bien
        self._current_symbol = symbol
        self._current_grid_state = gs
        self._current_order_manager = om
        self._current_price_reader = pr
        self._price_min = price_min
        self._price_max = price_max

        logger.info(
            "token_inicializado",
            symbol=symbol,
            price_min=round(price_min, 2),
            price_max=round(price_max, 2),
            num_levels=num_levels,
            loaded_from_state=loaded,
            dry_run=self._dry_run,
        )
        return True

    # ------------------------------------------------------------------
    # Cambio de par (round-robin)
    # ------------------------------------------------------------------

    def _switch_to_next(self) -> bool:
        """
        Rota al siguiente token en round-robin con garantía de atomicidad.

        Secuencia:
          1. Pre-flight: verifica precio del siguiente token ANTES de cancelar el actual.
          2. Si no hay precio → aborta, mantiene el token actual intacto.
          3. Cancela la grilla actual.
          4. Inicializa la nueva grilla.
          5. Si falla → intenta recuperar el token anterior (re-inicializa desde estado).

        Returns:
            True si el cambio fue exitoso.
        """
        next_idx = (self._current_idx + 1) % len(self._tokens)
        next_token = self._tokens[next_idx]
        next_symbol = next_token["symbol"]
        prev_symbol = self._current_symbol
        prev_idx = self._current_idx

        # Pre-flight: comprobar precio del siguiente símbolo sin cancelar el actual
        _, _, next_pr = self._build_components(next_symbol)
        try:
            next_pr.get_current_price()
        except Exception as exc:
            logger.error(
                "cambio_de_par_abortado_sin_precio",
                siguiente=next_symbol,
                error=str(exc),
            )
            return False

        logger.info(
            "cambio_de_par_iniciando",
            actual=prev_symbol,
            siguiente=next_symbol,
        )

        # Cancelar grilla actual (órdenes abiertas)
        if self._current_order_manager:
            self._current_order_manager.cancel_all_orders()
        if self._current_grid_state:
            self._current_grid_state.save()

        # Intentar inicializar el nuevo token
        self._current_idx = next_idx
        success = self._initialize_token(next_token)

        if not success:
            logger.error(
                "cambio_de_par_fallido_intentando_recuperar",
                fallido=next_symbol,
                recuperando=prev_symbol,
            )
            # Volver al token anterior (órdenes ya canceladas, re-inicializar desde estado)
            self._current_idx = prev_idx
            recovery = self._initialize_token(self._tokens[prev_idx])
            if not recovery:
                logger.critical(
                    "recuperacion_fallida_tras_cambio_de_par",
                    token=prev_symbol,
                )
            return False

        self._last_switch_time = time.time()
        logger.info(
            "cambio_de_par_exitoso",
            anterior=prev_symbol,
            actual=self._current_symbol,
        )
        self._notifier.send(
            f"Cambio de par: {prev_symbol} -> {self._current_symbol}\n"
            f"Rango: {round(self._price_min, 2)} - {round(self._price_max, 2)}"
        )
        return True

    def _should_switch(self) -> bool:
        """Retorna True si transcurrió el cooldown mínimo desde el último cambio de par."""
        return (time.time() - self._last_switch_time) >= self._pair_cooldown

    # ------------------------------------------------------------------
    # Ciclo de operación
    # ------------------------------------------------------------------

    def _run_cycle(self) -> bool:
        """
        Ejecuta un ciclo de monitoreo para el token activo.

        Returns:
            True si el ciclo fue exitoso.
        """
        try:
            current_price = self._current_price_reader.get_current_price()  # type: ignore[union-attr]

            in_range = self._current_price_reader.is_price_in_range(  # type: ignore[union-attr]
                current_price, self._price_min, self._price_max
            )
            if not in_range:
                logger.warning(
                    "precio_fuera_de_rango_multi",
                    symbol=self._current_symbol,
                    price=round(current_price, 2),
                    price_min=round(self._price_min, 2),
                    price_max=round(self._price_max, 2),
                )
                self._total_cycles += 1
                return True

            filled_orders = self._current_order_manager.check_filled_orders(current_price)  # type: ignore[union-attr]
            for filled in filled_orders:
                self._current_order_manager.recycle_order(filled)  # type: ignore[union-attr]

            self._current_grid_state.save()  # type: ignore[union-attr]
            self._risk_manager.reset_errors()
            self._total_cycles += 1

            logger.info(
                "ciclo_multi_completado",
                symbol=self._current_symbol,
                ciclo=self._total_cycles,
                price=round(current_price, 2),
                filled=len(filled_orders),
                dry_run=self._dry_run,
            )

            if filled_orders:
                self._notifier.notify_orders_filled(
                    symbol=self._current_symbol,
                    filled_count=len(filled_orders),
                    current_price=round(current_price, 2),
                    cycle=self._total_cycles,
                    dry_run=self._dry_run,
                )

            return True

        except Exception as exc:
            error_msg = str(exc)
            logger.error("error_en_ciclo_multi", error=error_msg, exc_info=True)
            self._notifier.notify_critical_error(error_msg, self._total_cycles)
            triggered = self._risk_manager.register_error()
            if triggered:
                logger.critical("circuit_breaker_activado_multi")
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
        Ejecuta el loop multi-token hasta Ctrl+C, circuit breaker o max_cycles.

        Args:
            max_cycles: Límite de ciclos (0 = sin límite).
        """
        logger.info(
            "multi_bot_loop_iniciado",
            tokens=[t["symbol"] for t in self._tokens],
            dry_run=self._dry_run,
            pair_cooldown_seconds=int(self._pair_cooldown),
            interval_seconds=int(self._interval),
        )

        first_token = self._tokens[self._current_idx]
        if not self._initialize_token(first_token):
            logger.critical(
                "multi_bot_loop_abortado | fallo_en_primer_token",
                token=first_token["symbol"],
            )
            return

        self._last_switch_time = time.time()
        self._running = True

        while self._running:
            if self._should_switch() and len(self._tokens) > 1:
                self._switch_to_next()

            self._run_cycle()

            if self._risk_manager.circuit_open:
                break

            if max_cycles > 0 and self._total_cycles >= max_cycles:
                logger.info("max_ciclos_alcanzado_multi", ciclos=self._total_cycles)
                break

            if self._running:
                time.sleep(self._interval)

        self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Cancela órdenes activas y persiste el estado final."""
        self._running = False
        cancelled = 0
        if self._current_order_manager:
            cancelled = self._current_order_manager.cancel_all_orders()
        logger.info(
            "multi_bot_loop_detenido",
            ciclos_completados=self._total_cycles,
            ordenes_canceladas=cancelled,
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

    @property
    def current_symbol(self) -> str | None:
        return self._current_symbol
