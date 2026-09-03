"""
Configuration module for the Quant Trading application.

Manages credentials, trade parameters, and runtime settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Runtime configuration for the quant trading application."""

    # ── Broker ──
    broker: str = "dhan"           # dhan | zerodha | binance | tradingview
    sandbox: bool = True          # demo mode (no live orders)

    # ── Dhan API credentials ──
    client_id: str = ""
    access_token: str = ""

    # ── Zerodha (Kite Connect) credentials ──
    kite_api_key: str = ""
    kite_access_token: str = ""
    kite_sandbox: bool = True     # not used for zerodha; kept for parity

    # ── Binance credentials ──
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # ── TradingView integration ──
    tv_webhook_url: str = ""
    tv_symbol_map: str = ""

    # ── Default symbol / instrument ──
    symbol: str = "RELIANCE"
    exchange: str = "NSE"          # NSE, BSE, MCX, NSE_FNO, CUR / USDT crypto
    instrument_type: str = "EQUITY"
    security_id: str = ""          # numeric; resolved if blank

    # ── Data parameters ──
    timeframe: str = "5minute"     # 5-minute candles
    lookback_days: int = 10        # how many days of history to fetch

    # ── Risk parameters ──
    account_equity: float = 100000.0
    risk_per_trade: float = 0.01   # 1 % of equity
    max_risk_per_trade: float = 5000.0
    rr_ratio: float = 2.0          # reward : risk

    # ── ML training parameters ──
    train_window: int = 250        # rows used for training
    ema_span: int = 169
    rsi_period: int = 14
    atr_period: int = 14

    # ── Instrument master cache ──
    security_cache_path: str = "security_master.csv"
    security_cache_ttl: int = 86400  # seconds

    def validate(self) -> list[str]:
        """Return list of validation errors (empty when valid)."""
        errors: list[str] = []
        if not self.sandbox:
            if self.broker == "dhan":
                if not self.client_id:
                    errors.append("client_id is required for Dhan")
                if not self.access_token:
                    errors.append("access_token is required for Dhan")
            elif self.broker == "zerodha":
                if not self.kite_api_key:
                    errors.append("Zerodha API key is required")
                if not self.kite_access_token:
                    errors.append("Zerodha access token is required")
            elif self.broker == "binance":
                if self.binance_testnet is False:
                    if not self.binance_api_key or not self.binance_api_secret:
                        errors.append("Binance API key and secret are required "
                                      "when not in testnet mode")
            elif self.broker == "tradingview":
                if not self.tv_webhook_url:
                    # TradingView can run in data/analysis-only mode; orders
                    # just report as pending. Not a hard error.
                    pass
        return errors

    @property
    def exchange_segment_value(self) -> str:
        """Return the broker-agnostic exchange segment string."""
        mapping = {
            "NSE": "NSE_EQ",
            "BSE": "BSE_EQ",
            "MCX": "MCX_COMM",
            "NSE_FNO": "NSE_FNO",
            "BSE_FNO": "BSE_FNO",
            "CUR": "NSE_CURRENCY",
            "INDEX": "IDX_I",
        }
        return mapping.get(self.exchange, "NSE_EQ")

    def save_to_env(self) -> None:
        """Persist active broker credentials to environment variables."""
        os.environ["BROKER"] = self.broker
        os.environ["DHAN_CLIENT_ID"] = self.client_id
        os.environ["DHAN_ACCESS_TOKEN"] = self.access_token
        os.environ["KITE_API_KEY"] = self.kite_api_key
        os.environ["KITE_ACCESS_TOKEN"] = self.kite_access_token
        os.environ["BINANCE_API_KEY"] = self.binance_api_key
        os.environ["BINANCE_API_SECRET"] = self.binance_api_secret
        os.environ["TV_WEBHOOK_URL"] = self.tv_webhook_url
        os.environ["TV_SYMBOL_MAP"] = self.tv_symbol_map
