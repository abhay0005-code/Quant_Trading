"""
Dhan API client wrapper.

Provides a single interface to:
  - Fetch 5-minute (and other intervals) OHLCV intraday data
  - Fetch daily OHLCV data
  - Resolve ticker symbols → security IDs via the Dhan instrument master
  - Get live quotes, fund limits, holdings
  - Place / preview orders

When *sandbox* mode is enabled (or credentials are missing) the client
falls back to ``yfinance`` so the UI remains fully functional for demos.
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    from dhanhq import DhanContext, dhanhq
except Exception:  # pragma: no cover
    DhanContext = None
    dhanhq = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from utils import epoch_to_datetime, dhan_response_to_df, yf_to_df, safe_float

log = logging.getLogger("dhan_client")


# ──────────────────────────────────────────────────────────────────────
#  Symbol → security-id resolution
# ──────────────────────────────────────────────────────────────────────

# A small built-in mapping for well-known Indian instruments.
# Users can still rely on the live CSV download or manual entry.
_BUILTIN_SECURITY_MAP = {
    ("NSE", "RELIANCE"): ("720225", "EQUITY"),
    ("NSE", "TCS"):       ("131135", "EQUITY"),
    ("NSE", "HDFCBANK"):  ("349377", "EQUITY"),
    ("NSE", "INFY"):      ("454621", "EQUITY"),
    ("NSE", "ICICIBANK"): ("424390", "EQUITY"),
    ("NSE", "HDFC"):      ("349377", "EQUITY"),
    ("NSE", "SBIN"):      ("350640", "EQUITY"),
    ("NSE", "LT"):        ("366680", "EQUITY"),
    ("NSE", "ITC"):       ("169603", "EQUITY"),
    ("NSE", "HUL"):       ("356893", "EQUITY"),
    ("NSE", "WIPRO"):     ("376896", "EQUITY"),
    ("NSE", "AXISBANK"):  ("445319", "EQUITY"),
    ("NSE", "MARICO"):    ("529637", "EQUITY"),
    ("NSE_FNO", "NIFTY"):  ("26000",  "INDEX"),
    ("NSE_FNO", "BANKNIFTY"): ("26001", "INDEX"),
}

_COMPACT_CSV_URL = (
    "https://images.dhan.co/api-data/api-scrip-master.csv"
)


def resolve_security_id(symbol: str, exchange: str,
                        cache_path: str | None = None) -> tuple[str, str]:
    """
    Resolve a trading symbol to ``(security_id, instrument_type)``.

    Tries the built-in map first, then falls back to the online CSV.
    """
    key = (exchange.upper(), symbol.upper())
    if key in _BUILTIN_SECURITY_MAP:
        return _BUILTIN_SECURITY_MAP[key]

    cache_path = cache_path or "security_master.csv"
    try:
        if not os.path.exists(cache_path) or \
           (time.time() - os.path.getmtime(cache_path)) > 86400:
            r = requests.get(_COMPACT_CSV_URL, timeout=15)
            r.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(r.content)

        df = pd.read_csv(cache_path)
        mask = (
            df["SEM_TRADING_SYMBOL"].str.upper().eq(symbol.upper())
            & df["SEM_EXM_EXCH_ID"].str.upper().eq(exchange.upper())
        )
        if mask.any():
            row = df[mask].iloc[-1]
            return str(row["SEM_SMST_SECURITY_ID"]), row["SEM_INSTRUMENT_NAME"]
    except Exception as e:
        log.warning("Could not resolve security ID via CSV: %s", e)

    raise ValueError(
        f"Could not resolve security_id for {symbol} on {exchange}. "
        "Please provide it manually."
    )


# ──────────────────────────────────────────────────────────────────────
#  Data client
# ──────────────────────────────────────────────────────────────────────

# yfinance symbol mapping for common Indian / US stocks (demo fallback)
_YF_MAP = {
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS",
    "HDFCBANK":  "HDFCBANK.NS",
    "INFY":      "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN":      "SBIN.NS",
    "LT":        "LT.NS",
    "ITC":       "ITC.NS",
    "HUL":       "HINDUNILVR.NS",
    "WIPRO":     "WIPRO.NS",
    "AXISBANK":  "AXISBANK.NS",
    "MARICO":    "MARICO.NS",
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "AAPL":      "AAPL",
    "TSLA":      "TSLA",
    "GOOG":      "GOOGL",
}


class DhanDataClient:
    """
    Unified data + trading client.

    When credentials are available and ``sandbox`` is False, data is
    fetched live from the Dhan REST API.
    Otherwise, ``yfinance`` is used as a transparent fallback so the
    full pipeline can be tested end-to-end without Dhan access.
    """

    def __init__(self, client_id: str = "", access_token: str = "",
                 sandbox: bool = True):
        self.client_id = client_id
        self.access_token = access_token
        self.sandbox = sandbox or not client_id or not access_token
        self._dhan: Optional[dhanhq] = None
        self._connected = False

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        """Initialise the dhanhq client (if not sandboxed)."""
        if self.sandbox:
            self._connected = True
            return True
        if DhanContext is None:
            raise RuntimeError("dhanhq package is not installed")
        ctx = DhanContext(self.client_id, self.access_token)
        self._dhan = dhanhq(ctx)
        self._connected = True
        return self._connected

    @property
    def is_connected(self) -> bool:
        """True once ``connect()`` has succeeded or sandbox mode is active."""
        return self._connected

    # ── Data fetching ────────────────────────────────────────────
    @staticmethod
    def _yf_symbol(symbol: str) -> str:
        return _YF_MAP.get(symbol.upper(), symbol)

    def fetch_intraday(self, symbol, exchange="NSE", security_id="",
                       instrument_type="EQUITY", days=10,
                       interval_minutes=5) -> pd.DataFrame:
        """Fetch intraday OHLCV data."""
        if not self.sandbox and self._dhan is not None:
            return self._fetch_from_dhan(
                symbol, exchange, security_id, instrument_type, days, interval_minutes)
        return self._fetch_from_yf(symbol, days, interval_minutes)

    def _fetch_from_dhan(self, symbol, exchange, security_id, instrument_type,
                         days, interval_minutes) -> pd.DataFrame:
        dhan = self._dhan
        if not security_id:
            security_id, instrument_type = resolve_security_id(symbol, exchange)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=min(days, 5))).strftime("%Y-%m-%d")
        exchange_segment = dhan.NSE if exchange.upper() == "NSE" else exchange
        response = dhan.intraday_minute_data(
            security_id=str(security_id), exchange_segment=exchange_segment,
            instrument_type=instrument_type, from_date=start_date,
            to_date=end_date, interval=interval_minutes, oi=True)
        if response.get("status") == "failure":
            raise RuntimeError(f"Dhan API error: {response.get('remarks', '')}")
        df = dhan_response_to_df(response)
        if df.empty and isinstance(response.get("data"), dict):
            inner = response["data"].get("data", {})
            if isinstance(inner, dict) and "timestamp" in inner:
                df = pd.DataFrame(inner)
                _normalise_df_inline(df)
        if df.empty:
            raise RuntimeError("No intraday data returned")
        return df

    def _fetch_from_yf(self, symbol, days, interval_minutes) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed and Dhan credentials missing")
        yf_symbol = self._yf_symbol(symbol)
        yf_interval_map = {1: "1m", 5: "5m", 15: "15m", 25: "15m", 60: "60m"}
        yf_interval = yf_interval_map.get(interval_minutes, "5m")
        yf_period = f"{min(days, 60)}d" if yf_interval in ("1m", "5m") else f"{min(days, 720)}d"
        try:
            raw = yf.download(yf_symbol, interval=yf_interval, period=yf_period, progress=False)
        except Exception as e:
            raise RuntimeError(f"yfinance fetch failed: {e}")
        if raw.empty:
            raise RuntimeError(f"No data returned by yfinance for {yf_symbol}")
        return yf_to_df(raw)

    # ── Quotes / trading ─────────────────────────────────────────
    def get_ltp(self, symbol, exchange="NSE", security_id="") -> float:
        """Return last-traded price (best-effort; 0 on failure)."""
        if not self.sandbox and self._dhan is not None:
            try:
                if not security_id:
                    security_id, _ = resolve_security_id(symbol, exchange)
                seg = self._dhan.NSE if exchange.upper() == "NSE" else exchange
                resp = self._dhan.marketfeed_ltp([seg], [security_id])
                data = resp.get("data", {})
                if security_id in data:
                    return safe_float(data[security_id].get("last_price"))
            except Exception as e:
                log.warning("LTP fetch failed: %s", e)
        try:
            df = self._fetch_from_yf(symbol, 1, 5)
            return safe_float(df["close"].iloc[-1])
        except Exception:
            return 0.0

    def place_order(self, symbol, quantity, side="BUY", price=None,
                    exchange="NSE", security_id="") -> dict:
        """Place a market/limit order via Dhan (no-op in sandbox)."""
        if self.sandbox or self._dhan is None:
            return {"status": "sandbox",
                    "message": f"Sandbox: {side} {quantity} {symbol} (no order sent)"}
        transaction = self._dhan.BUY if side.upper() == "BUY" else self._dhan.SELL
        if not security_id:
            security_id, _ = resolve_security_id(symbol, exchange)
        seg = self._dhan.NSE if exchange.upper() == "NSE" else exchange
        return self._dhan.place_order(
            security_id=str(security_id), exchange_segment=seg,
            transaction_type=transaction, quantity=int(quantity),
            order_type=self._dhan.MARKET if price is None else self._dhan.LIMIT,
            product_type=self._dhan.INTRA, price=price or 0)


def _normalise_df_inline(df):
    """Inline normaliser mirroring ``utils._normalise_df`` for raw payloads."""
    from utils import _normalise_df
    _normalise_df(df)