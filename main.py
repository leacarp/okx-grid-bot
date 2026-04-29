"""
Entry point del OKX Grid Trading Bot.

Uso:
    python main.py --mode=scan   # Lee precio y balance, sin órdenes
"""
from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

load_dotenv()

from src.connectors.okx_client import OKXClient
from src.core.price_reader import PriceReader
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging


def _start_health_server(port: int = 8080) -> None:
    """
    Inicia un servidor HTTP mínimo en un thread daemon.

    Responde 200 OK a cualquier request para que Fly.io detecte
    la app como viva. Corre en background y nunca bloquea el bot.
    """

    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args) -> None:  # silencia el log de acceso
            pass

    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, name="health-server", daemon=True)
    thread.start()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OKX Grid Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["scan", "live", "multi"],
        default="scan",
        help=(
            "Modo de operación: "
            "'scan' lee precio y balance; "
            "'live' ejecuta el grid loop single-token; "
            "'multi' itera en round-robin sobre todos los tokens del config."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo de configuración (default: config.yaml).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Número máximo de ciclos en modo live (0 = sin límite).",
    )
    return parser.parse_args()


def run_scan(config: dict, log) -> None:
    """
    Modo scan: conecta a OKX, lee precio de BTC/USDT y balance USDT.
    No coloca órdenes. DRY_RUN siempre activo en este modo.
    """
    symbol: str = config["grid"]["symbol"]
    dry_run: bool = config["dry_run"]

    log.info("modo_scan_inicio", symbol=symbol, dry_run=dry_run)

    client = OKXClient.from_env(sandbox=config["exchange"].get("sandbox", False))

    # Validar conexión
    balance_data = client.connect()

    usdt_free: float = (
        balance_data.get("USDT", {}).get("free", 0.0)
        or balance_data.get("free", {}).get("USDT", 0.0)
        or 0.0
    )

    log.info(
        "balance_leido",
        usdt_disponible=round(usdt_free, 4),
        dry_run=dry_run,
    )

    price_reader = PriceReader(client=client, symbol=symbol)
    current_price = price_reader.get_current_price()

    price_min: float = config["grid"]["price_min"]
    price_max: float = config["grid"]["price_max"]
    in_range = price_reader.is_price_in_range(current_price, price_min, price_max)

    log.info(
        "precio_leido",
        symbol=symbol,
        precio_actual=round(current_price, 2),
        rango=f"{price_min} - {price_max}",
        en_rango=in_range,
        dry_run=dry_run,
    )

    if not in_range:
        log.warning(
            "precio_fuera_de_rango",
            symbol=symbol,
            precio_actual=round(current_price, 2),
            rango_min=price_min,
            rango_max=price_max,
        )

    log.info("modo_scan_completo", symbol=symbol)


def run_live(config: dict, log, cycles: int) -> None:
    """
    Modo live: ejecuta el BotLoop completo con DRY_RUN activo.

    Coloca órdenes simuladas en la grilla, monitorea ejecuciones y recicla.
    Usa max_cycles para limitar la ejecución (útil para pruebas).
    """
    from src.core.bot_loop import BotLoop

    dry_run: bool = config["dry_run"]
    symbol: str = config["grid"]["symbol"]

    log.info("modo_live_inicio", symbol=symbol, dry_run=dry_run, max_cycles=cycles)

    bot = BotLoop(config=config)
    bot.run(max_cycles=cycles)

    log.info("modo_live_completo", ciclos=bot.total_cycles, dry_run=dry_run)


def run_multi(config: dict, log, cycles: int) -> None:
    """
    Modo multi: itera en round-robin sobre todos los tokens de config["tokens"].

    Cada token se opera durante pair_cooldown_seconds (por defecto 2 horas) antes
    de rotar al siguiente. Si la nueva grilla falla, mantiene el token anterior.
    """
    from src.core.multi_bot_loop import MultiBotLoop

    dry_run: bool = config["dry_run"]
    symbols = [t["symbol"] for t in config.get("tokens", [])]

    log.info("modo_multi_inicio", symbols=symbols, dry_run=dry_run, max_cycles=cycles)

    bot = MultiBotLoop(config=config)
    bot.run(max_cycles=cycles)

    log.info(
        "modo_multi_completo",
        ciclos=bot.total_cycles,
        ultimo_symbol=bot.current_symbol,
        dry_run=dry_run,
    )


def main() -> None:
    _start_health_server()

    args = _parse_args()
    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/bot.log"),
        max_bytes=log_cfg.get("max_bytes", 10_485_760),
        backup_count=log_cfg.get("backup_count", 5),
    )

    log = get_logger("main")

    log.info(
        "bot_iniciado",
        modo=args.mode,
        dry_run=config["dry_run"],
        symbol=config["tokens"][0]["symbol"] if config.get("tokens") else "multi",
    )

    try:
        if args.mode == "scan":
            run_scan(config, log)
        elif args.mode == "live":
            run_live(config, log, cycles=args.cycles)
        elif args.mode == "multi":
            run_multi(config, log, cycles=args.cycles)
    except KeyboardInterrupt:
        log.info("bot_detenido_por_usuario")
        sys.exit(0)
    except Exception as exc:
        log.error("error_critico", error=str(exc), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
