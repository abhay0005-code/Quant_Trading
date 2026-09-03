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

    # ── Dhan API credentials ──
    client_id: str = ""
    access_token: str = ""

    # ── Default symbol / instrument ──
    symbol: str = "RELIANCE"
    exchange: str = "NSE"          # NSE, BSE, MCX, NSE_FNO, CUR
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

    # ── Sandbox / demo mode ──
    sandbox: bool = True          # use yfinance when True

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
            if not self.client_id:
                errors.append("client_id is required when not in sandbox mode")
            if not self.access_token:
                errors.append("access_token is required when not in sandbox mode")
        return errors

    @property
    def exchange_segment_value(self) -> str:
        """Return the dhanhq exchange-segment constant string."""
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
        """Persist credentials to environment variables."""
        os.environ["DHAN_CLIENT_ID"] = self.client_id
        os.environ["DHAN_ACCESS_TOKEN"] = self.access_token
