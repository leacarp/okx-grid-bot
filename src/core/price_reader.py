"""
Lectura de precio actual desde OKX y validación de rango.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.connectors.okx_client import OKXClient

logger = logging.getLogger(__name__)


class PriceReader:
    """
    Obtiene el precio mid (bid+ask)/2 de un símbolo y valida que esté en rango.
    """

    def __init__(self, client: "OKXClient", symbol: str) -> None:
        self._client = client
        self._symbol = symbol

    def get_current_price(self) -> float:
        """
        Retorna el precio mid del símbolo configurado.

        El precio mid es (bid + ask) / 2, más robusto que solo 'last'
        porque refleja el libro de órdenes en tiempo real.

        Raises:
            RuntimeError: Si el ticker no contiene bid/ask válidos.
        """
        ticker = self._client.fetch_ticker(self._symbol)

        bid = ticker.get("bid")
        ask = ticker.get("ask")
        last = ticker.get("last")

        if bid is not None and ask is not None:
            mid_price = (bid + ask) / 2.0
        elif last is not None:
            mid_price = float(last)
            logger.debug(
                "precio_mid_fallback | symbol=%s usando_last=%.2f",
                self._symbol,
                mid_price,
            )
        else:
            raise RuntimeError(
                f"Ticker de {self._symbol} no contiene bid/ask/last válidos: {ticker}"
            )

        logger.debug(
            "precio_leido | symbol=%s bid=%s ask=%s mid=%.2f",
            self._symbol,
            bid,
            ask,
            mid_price,
        )
        return mid_price

    def is_price_in_range(self, price: float, price_min: float, price_max: float) -> bool:
        """
        Retorna True si el precio está dentro del rango [price_min, price_max].
        """
        in_range = price_min <= price <= price_max
        if not in_range:
            logger.warning(
                "precio_fuera_de_rango | symbol=%s price=%.2f min=%.2f max=%.2f",
                self._symbol,
                price,
                price_min,
                price_max,
            )
        return in_range
