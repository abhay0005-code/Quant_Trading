"""
Binance broker client.

Supports crypto spot intraday data and order placement using the official
``binance`` (python-binance) client. Works in testnet mode by setting
``testnet=True`` (recommended for demos).

Symbols are expressed as base+quote, e.g. ``BTCUSDT``, ``ETHUSDT``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from broker_base import BrokerClient, BrokerError, OHLCV_COLUMNS

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except Exception:  # pragma: no cover
    Client = None
    BinanceAPIException = None

log = logging.getLogger("binance_client")

# Interval (minutes) -> python-binance KLINE_INTERVAL string.
_BINANCE_INTERVAL = {
    1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h",
}

# Limit each klines request to this many candles.
_BINANCE_DEFAULT_LIMIT = 1000


class BinanceClient(BrokerClient):
    """python-binance client for crypto spot / margin."""

    name = "Binance"
    assets = ("crypto",)

    def __init__(self, api_key: str = "", api_secret: str = "",
                 testnet: bool = True, **kw: Any):
        super().__init__(api_key=api_key, api_secret=api_secret,
                         testnet=bool(testnet))
        self.api_key = api_key
        self.api_secret = api_secret
        # Binance testnet mode uses the testnet endpoints (paper trading).
        self.testnet = bool(testnet)
        self._client: Optional[Client] = None

    def connect(self) -> bool:
        if Client is None:
            raise BrokerError("python-binance package is not installed")
        try:
            self._client = Client(self.api_key or "", self.api_secret or "",
                                  testnet=self.testnet)
            # Public ping works even without credentials; needed to validate
            # the testnet endpoint is reachable.
            self._client.ping()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise BrokerError(f"Binance connection failed: {e}")
        return self._connected

    def _exchanges(self, symbol: str, **kw: Any) -> tuple[str, str]:
        base, quote = kw.get("base"), kw.get("quote")
        sym = (symbol or "").upper().strip()
        # Allow "BTC/USDT", "BTC:USDT", "BTCUSDT", "BTC_USDT"
        for sep in ("/", ":", "_"):
            if sep in sym:
                base, quote = sym.split(sep, 1)
                break
        if not base or not quote:
            # Concatenated form: "BTCUSDT" -> split at a known quote suffix.
            for q in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "BTC",
                      "ETH", "BNB", "EUR", "TRY", "BRL"):
                if sym.endswith(q) and len(sym) > len(q):
                    base, quote = sym[: -len(q)], q
                    break
        if not base or not quote:
            raise BrokerError(
                f"Binance symbol must include base/quote (e.g. BTCUSDT or "
                f"BTC/USDT) got {symbol!r}")
        return base.upper(), quote.upper()

    def fetch_intraday(self, symbol: str, exchange: str = "", days: int = 10,
                       interval_minutes: int = 5, **kw: Any) -> pd.DataFrame:
        if self._client is None:
            raise BrokerError("Binance client not connected (call connect first)")
        base, quote = self._exchanges(symbol, **kw)
        pair = f"{base}{quote}"
        interval = _BINANCE_INTERVAL.get(int(interval_minutes), "5m")
        # Binance klines accept a startTime/endTime in ms; we ask for enough
        # candles to cover `days` at the chosen resolution.
        limit = kw.get("limit", _BINANCE_DEFAULT_LIMIT)
        try:
            klines = self._client.get_klines(symbol=pair, interval=interval,
                                             limit=limit)
        except BinanceAPIException as e:
            raise BrokerError(f"Binance klines failed: {e}")
        except Exception as e:
            raise BrokerError(f"Binance klines failed: {e}")

        if not klines:
            raise BrokerError(f"No Binance data returned for {pair}")

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.set_index("datetime").sort_index()
        return df[[c for c in OHLCV_COLUMNS if c in df.columns]]

    def _place_order(self, symbol: str, quantity: int, side: str,
                     price: Optional[float], exchange: str, **kw: Any) -> dict:
        if self._client is None:
            return {"status": "error",
                    "message": "Binance client not connected", "broker": "binance"}
        base, quote = self._exchanges(symbol, **kw)
        pair = f"{base}{quote}"
        order_side = self._client.SIDE_BUY if side.upper() == "BUY" \
            else self._client.SIDE_SELL
        if price is None:
            order_type = self._client.ORDER_TYPE_MARKET
            params = {}
        else:
            order_type = self._client.ORDER_TYPE_LIMIT
            params = {"timeInForce": self._client.TIME_IN_FORCE_GTC}
        try:
            resp = self._client.create_order(
                symbol=pair, side=order_side, type=order_type,
                quantity=float(quantity), price=float(price), **params)
            return {"status": "success", "order_id": resp.get("orderId"),
                    "broker": "binance", "raw": resp.get("clientOrderId")}
        except BinanceAPIException as e:
            return {"status": "error", "message": str(e), "broker": "binance"}
        except Exception as e:
            return {"status": "error", "message": str(e), "broker": "binance"}
