"""
Zerodha broker client built on Kite Connect.

Supports live intraday/daily data and order placement using the official
``kiteconnect`` Python library. Requires an API key plus an access token
(the ``request_token`` exchanged via Kite Connect login).

Data is standardised to the OHLCV schema expected by the pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from broker_base import BrokerClient, BrokerError, OHLCV_COLUMNS

try:
    from kiteconnect import KiteConnect
except Exception:  # pragma: no cover
    KiteConnect = None

log = logging.getLogger("zerodha_client")

# Kite Connect interval -> interval string
_KITE_INTERVAL = {1: "minute", 3: "3minute", 5: "5minute", 10: "10minute",
                  15: "15minute", 60: "60minute"}


class ZerodhaClient(BrokerClient):
    """Kite Connect client for Zerodha (Indian equities, F&O, commodities)."""

    name = "Zerodha"
    assets = ("equity", "commodity", "index")

    def __init__(self, api_key: str = "", access_token: str = "",
                 sandbox: bool = False, **kw: Any):
        super().__init__(api_key=api_key, access_token=access_token,
                         sandbox=bool(sandbox))
        self.api_key = api_key
        self.access_token = access_token
        self.sandbox = bool(sandbox) or not api_key or not access_token
        self._kite: Optional[KiteConnect] = None

    def connect(self) -> bool:
        if self.sandbox:
            self._connected = True
            return True
        if KiteConnect is None:
            raise BrokerError("kiteconnect package is not installed")
        if not self.api_key or not self.access_token:
            raise BrokerError("Zerodha api_key and access_token are required")
        self._kite = KiteConnect(api_key=self.api_key)
        self._kite.set_access_token(self.access_token)
        # Validate the session with a lightweight call.
        try:
            self._kite.profile()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise BrokerError(f"Zerodha authentication failed: {e}")
        return self._connected

    def _instrument_token(self, symbol: str, exchange: str,
                          instrument_type: str = "NSE") -> str:
        """Resolve a symbol to a Kite instrument token (best effort)."""
        # Kite formats: NSE:RELIANCE, BSE:TCS, MCX:CRUDEOIL, NFO:NIFTY24OCTFUT
        prefix = {
            "NSE": "NSE", "BSE": "BSE", "MCX": "MCX", "NSE_FNO": "NFO",
            "NFO": "NFO", "BSE_FNO": "BFO", "CUR": "CDS",
        }.get(exchange.upper(), "NSE")
        return f"{prefix}:{symbol.upper()}"

    def fetch_intraday(self, symbol: str, exchange: str = "NSE", days: int = 10,
                       interval_minutes: int = 5, **kw: Any) -> pd.DataFrame:
        if self.sandbox or self._kite is None:
            # Fall back to yfinance via the shared helper so the pipeline can
            # still run for demos without live Zeordha access.
            from dhan_client import DhanDataClient
            return DhanDataClient(sandbox=True).fetch_intraday(
                symbol, days=days, interval_minutes=interval_minutes)

        instrument = self._instrument_token(symbol, exchange,
                                            kw.get("instrument_type", "NSE"))
        to_date = datetime.now()
        # Kite returns intraday minute data for recent dates; cap at 60 days
        # for minute data (Kite enforces a shorter lookback).
        from_date = to_date - timedelta(days=min(days, 30))
        interval = _KITE_INTERVAL.get(int(interval_minutes), "5minute")
        try:
            resp = self._kite.historical_data(
                instrument, from_date, to_date, interval,
                continuous=True, oi=False)
        except Exception as e:
            raise BrokerError(f"Zerodha historical data failed: {e}")
        if not resp:
            raise BrokerError(f"No Zerodha data returned for {instrument}")

        df = pd.DataFrame(resp)
        # Kite returns a "date" column (timezone-naive) we convert to a
        # DatetimeIndex, dropping the duplicate "oi" if present.
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
        if "oi" in df.columns:
            df = df.drop(columns=["oi"])
        rename = {"date": "datetime"}
        for c in OHLCV_COLUMNS:
            if c in df.columns:
                rename.setdefault(c, c)
        df = df.rename(columns=rename)
        df = df.set_index("datetime").sort_index()
        return df[[c for c in OHLCV_COLUMNS if c in df.columns]]

    def _place_order(self, symbol: str, quantity: int, side: str,
                     price: Optional[float], exchange: str, **kw: Any) -> dict:
        if self.sandbox or self._kite is None:
            return {"status": "sandbox",
                    "message": f"Sandbox: {side} {quantity} {symbol} (no order sent)"}
        instrument = self._instrument_token(symbol, exchange,
                                            kw.get("instrument_type", "NSE"))
        txn = self._kite.TRANSACTION_TYPE_BUY if side.upper() == "BUY" \
            else self._kite.TRANSACTION_TYPE_SELL
        order_type = self._kite.ORDER_TYPE_MARKET if price is None \
            else self._kite.ORDER_TYPE_LIMIT
        try:
            resp = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=exchange.upper(),
                tradingsymbol=symbol.upper(),
                transaction_type=txn,
                quantity=int(quantity),
                product=self._kite.PRODUCT_MIS,
                order_type=order_type,
                price=float(price) if price is not None else 0.0,
            )
            return {"status": "success", "order_id": resp, "broker": "zerodha"}
        except Exception as e:
            return {"status": "error", "message": str(e), "broker": "zerodha"}
