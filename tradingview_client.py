"""
TradingView integration client.

TradingView is a charting platform rather than a broker with a public
REST API for order execution or historical intraday data. The supported
patterns are:

  1. **Alert → Webhook (orders).** TradingView alerts can POST a JSON
     payload to a configured webhook URL. This client instead *emits* a
     TradingView-compatible alert payload to your own webhook receiver
     (e.g. a local bot / relay) when the pipeline produces a signal, so
     orders can be forwarded to any broker.

     Set ``webhook_url`` to the endpoint that should receive these alerts.

  2. **Data.** TradingView does not expose a free historical REST API, so
     for pipeline data we transparently fall back to ``yfinance`` (the
     same fallback Dhan uses in sandbox mode). This keeps the full
     analysis pipeline usable for TradingView symbols.

      `CLOUD_TRADER_APP_KEY` / `sharing.genuine.page` secrets are not used;
     no official account-based data API is consumed.

Because order execution on TradingView is alert-driven, ``supports_orders``
is True but only when a ``webhook_url`` is configured; otherwise signals are
returned as a pending alert.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import pandas as pd
import requests

from broker_base import BrokerClient, BrokerError, OHLCV_COLUMNS

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

log = logging.getLogger("tradingview_client")


class TradingViewClient(BrokerClient):
    """Webhook-emitting TradingView integration."""

    name = "TradingView"
    assets = ("equity", "crypto", "index")
    supports_orders = True
    supports_stream = True

    def __init__(self, webhook_url: str = "", symbol_map: str = "",
                 sandbox: bool = True, **kw: Any):
        super().__init__(webhook_url=webhook_url, symbol_map=symbol_map,
                         sandbox=bool(sandbox))
        self.webhook_url = (webhook_url or "").strip()
        # Optional comma-separated "TV_SYMBOL:YF_SYMBOL" mappings.
        self.symbol_map: dict[str, str] = {}
        for pair in (symbol_map or "").split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                self.symbol_map[k.strip().upper()] = v.strip()
        self._connected = True

    def connect(self) -> bool:
        # TradingView needs no persistent session; connectivity is just the
        # presence of a webhook URL for order forwarding.
        self._connected = True
        return True

    def _yf_symbol(self, symbol: str) -> str:
        return self.symbol_map.get((symbol or "").strip().upper(), symbol or "")

    def fetch_intraday(self, symbol: str, exchange: str = "", days: int = 10,
                       interval_minutes: int = 5, **kw: Any) -> pd.DataFrame:
        if yf is None:
            raise BrokerError("yfinance is not installed (TradingView uses it "
                              "for pipeline data)")
        yf_symbol = self._yf_symbol(symbol)
        interval_map = {1: "1m", 5: "5m", 15: "15m", 25: "15m", 60: "60m"}
        yf_interval = interval_map.get(int(interval_minutes), "5m")
        yf_period = f"{min(days, 60)}d" if yf_interval in ("1m", "5m") \
            else f"{min(days, 720)}d"
        try:
            raw = yf.download(yf_symbol, interval=yf_interval, period=yf_period,
                              progress=False)
        except Exception as e:
            raise BrokerError(f"yfinance (TradingView data) failed: {e}")
        if raw is None or raw.empty:
            raise BrokerError(f"No data returned for {yf_symbol}")
        return _tv_to_df(raw)

    def _place_order(self, symbol: str, quantity: int, side: str,
                     price: Optional[float], exchange: str, **kw: Any) -> dict:
        """Forward a TradingView-compatible alert to the configured webhook."""
        if not self.webhook_url:
            return {
                "status": "pending",
                "message": "No webhook_url configured — alert NOT sent. "
                           "Set webhook_url to forward orders via TradingView.",
                "broker": "tradingview",
            }

        alert = {
            "symbol": (symbol or "").strip().upper(),
            "action": ("buy" if side.upper() == "BUY" else "sell").lower(),
            "quantity": int(quantity),
            "price": float(price) if price is not None else 0.0,
        }
        alert.update(kw.get("extra", {}) or {})
        try:
            resp = requests.post(self.webhook_url, json=alert, timeout=15)
            resp.raise_for_status()
            return {"status": "success", "broker": "tradingview",
                    "webhook_url": self.webhook_url, "alert": alert,
                    "http": resp.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e), "broker": "tradingview"}


def _tv_to_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance DataFrame for TradingView symbol data."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    rename = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ("open", "high", "low", "close", "volume"):
            rename[col] = cl
    df = df.rename(columns=rename)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[[c for c in OHLCV_COLUMNS if c in df.columns]]
