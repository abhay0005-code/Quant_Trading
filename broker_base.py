"""
Broker abstraction layer for the Quant Trading system.

Every supported broker implements the :class:`BrokerClient` interface so
the pipeline (data fetch + order placement) is broker-agnostic. Concrete
implementations live in their own modules:

  - ``dhan_client``  : Dhan (NSE/BSE/MCX/F&O) — DhanHQ REST API
  - ``zerodha_client``: Zerodha (NSE/BSE/MCX/F&O) — Kite Connect REST API
  - ``binance_client``: Binance (crypto) — Binance spot/margin API
  - ``tradingview_client``: TradingView — webhook receiver that translates
    TradingView alert payloads into quotes / order requests

Brokers that only support a subset of the interface raise
:class:`NotImplementedError` for unsupported methods so callers can guard
themselves with capability flags.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

# Standardised OHLCV column set every broker must produce (lowercase).
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class BrokerError(RuntimeError):
    """Raised when a broker operation fails (connection, data, or order)."""


class BrokerClient(ABC):
    """
    Common interface every broker client implements.

    Capability flags let the UI / pipeline degrade gracefully:
      - ``supports_orders``   : broker can place live orders
      - ``supports_stream``   : broker can push intraday ticks (webhook)
      - ``assets``            : asset class handled (\"equity\"/\"crypto\")
    """

    #: Human-readable broker name for the UI.
    name: str = "broker"

    #: True when this broker can place real orders.
    supports_orders: bool = True

    #: True when this broker accepts push notifications (e.g. TradingView).
    supports_stream: bool = False

    #: Asset class: "equity", "commodity", "crypto", "index".
    assets: tuple[str, ...] = ("equity",)

    def __init__(self, **creds: Any):
        self._creds = creds
        self._connected = False

    # ── Connection ───────────────────────────────────────────────
    @abstractmethod
    def connect(self) -> bool:
        """Validate/setup the broker session. Return True on success."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Data ─────────────────────────────────────────────────────
    @abstractmethod
    def fetch_intraday(self, symbol: str, exchange: str = "", days: int = 10,
                       interval_minutes: int = 5, **kw: Any) -> pd.DataFrame:
        """Return standardised OHLCV with a DatetimeIndex."""

    def get_ltp(self, symbol: str, exchange: str = "", **kw: Any) -> float:
        """Return the last traded price (0.0 when unavailable)."""
        try:
            df = self.fetch_intraday(symbol, exchange, days=1,
                                     interval_minutes=5, **kw)
            return float(df["close"].iloc[-1])
        except Exception:
            return 0.0

    # ── Orders ───────────────────────────────────────────────────
    def place_order(self, symbol: str, quantity: int, side: str = "BUY",
                    price: Optional[float] = None, exchange: str = "",
                    **kw: Any) -> dict:
        if not self.supports_orders:
            raise NotImplementedError(f"{self.name} does not support live orders")
        return self._place_order(symbol, quantity, side, price, exchange, **kw)

    def _place_order(self, symbol: str, quantity: int, side: str,
                     price: Optional[float], exchange: str, **kw: Any) -> dict:
        raise NotImplementedError


def registry() -> dict[str, type]:
    """Return the {broker_id: client_class} registry.

    Imported lazily to avoid heavy client imports at module load.
    """
    from dhan_client import DhanDataClient
    from zerodha_client import ZerodhaClient
    from binance_client import BinanceClient
    from tradingview_client import TradingViewClient

    return {
        "dhan": DhanDataClient,
        "zerodha": ZerodhaClient,
        "binance": BinanceClient,
        "tradingview": TradingViewClient,
    }


def create_broker(broker_id: str, **creds: Any) -> BrokerClient:
    """Instantiate a broker client by id."""
    cls = registry().get((broker_id or "dhan").strip().lower())
    if cls is None:
        raise BrokerError(f"Unknown broker: {broker_id!r}")
    if issubclass(cls, BrokerClient):
        return cls(**creds)
    # DhanDataClient predates the interface — wrap it in an adapter below.
    return _DhanAdapter(cls, **creds)


class _DhanAdapter(BrokerClient):
    """Adapter exposing the pre-existing :class:`DhanDataClient` through the
    common :class:`BrokerClient` interface without rewriting its internals."""

    name = "Dhan"

    def __init__(self, client_cls: type, **creds: Any):
        super().__init__(**creds)
        self._client = client_cls(
            client_id=creds.get("client_id", ""),
            access_token=creds.get("access_token", ""),
            sandbox=bool(creds.get("sandbox", True)),
        )

    def connect(self) -> bool:
        return self._client.connect()

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    @property
    def sandbox(self) -> bool:
        return self._client.sandbox

    def fetch_intraday(self, symbol: str, exchange: str = "", days: int = 10,
                       interval_minutes: int = 5, **kw: Any) -> pd.DataFrame:
        return self._client.fetch_intraday(
            symbol, exchange or "NSE",
            security_id=kw.get("security_id", ""),
            instrument_type=kw.get("instrument_type", "EQUITY"),
            days=days, interval_minutes=interval_minutes)

    def get_ltp(self, symbol: str, exchange: str = "", **kw: Any) -> float:
        return self._client.get_ltp(symbol, exchange or "NSE",
                                    security_id=kw.get("security_id", ""))

    def _place_order(self, symbol: str, quantity: int, side: str,
                     price: Optional[float], exchange: str, **kw: Any) -> dict:
        return self._client.place_order(
            symbol, quantity, side=side, price=price,
            exchange=exchange or "NSE",
            security_id=kw.get("security_id", ""))
