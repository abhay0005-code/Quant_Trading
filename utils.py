"""
Utility functions shared across the quant trading application.

Includes:
  - Data normalisation (Dhan API ↔ yfinance ↔ standard DataFrame)
  - Epoch / datetime conversion
  - Matplotlib figure helpers
  - Safe numeric helpers
"""
from __future__ import annotations

import math
import warnings


# ──────────────────────────────────────────────────────────────────────
#  Data conversion
# ──────────────────────────────────────────────────────────────────────

def epoch_to_datetime(epoch):
    """Convert a Dhan epoch timestamp (seconds) to an IST datetime."""
    if isinstance(epoch, (list, np.ndarray)):
        return pd.Series([_epoch_to_dt(e) for e in epoch])
    return _epoch_to_dt(epoch)


def _epoch_to_dt(epoch):
    try:
        return datetime.fromtimestamp(int(epoch), IST)
    except (ValueError, OSError, OverflowError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def dhan_response_to_df(response):
    """
    Convert a ``dhan.intraday_minute_data`` / ``historical_daily_data``
    response dict into a standard OHLCV DataFrame.

    The Dhan API returns::
        {"status":"success","remarks":"","data":{"timestamp":[...],
         "open":[...],"high":[...],"low":[...],"close":[...],"volume":[...]}}

    The ``data`` value may itself be wrapped in another ``data`` key.
    """
    if not isinstance(response, dict):
        raise ValueError("Dhan response is not a dict")

    if response.get("status") == "failure":
        msg = response.get("remarks", "Unknown API error")
        raise RuntimeError(f"Dhan API failure: {msg}")

    data = response.get("data", {})
    # Handle double-nested data.data (Dhan wraps json body inside dhan_http)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    if not isinstance(data, dict) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    _normalise_df(df)
    return df


def yf_to_df(yf_df):
    """Convert a yfinance OHLCV DataFrame to the standard lowercase schema."""
    df = yf_df.copy()
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]

    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            rename_map[col] = cl
    df = df.rename(columns=rename_map)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]


def _normalise_df(df):
    """Mutate *df* in-place: rename columns, convert epoch to datetime,
    set a DatetimeIndex, sort, and coerce dtypes."""
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ("open", "high", "low", "close", "volume", "timestamp", "oi"):
            col_map[col] = cl
    df.rename(columns=col_map, inplace=True)

    if "timestamp" in df.columns:
        df["datetime"] = df["timestamp"].apply(
            lambda e: _epoch_to_dt(e) if not pd.isna(e) else pd.NaT
        )
        df.drop(columns=["timestamp"], inplace=True)
        df.set_index("datetime", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    if "oi" in df.columns:
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)

    df.sort_index(inplace=True)


# ──────────────────────────────────────────────────────────────────────
#  Safe numeric helpers
# ──────────────────────────────────────────────────────────────────────

def safe_float(val, default=0.0):
    """Convert *val* to float, returning *default* on failure."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────────────

def plot_candlestick(df, title="OHLC", ema_cols=None, vwap_col=None,
                     markers=None, figsize=(14, 7)):
    """Create a candlestick chart with optional overlay lines."""
    fig, ax = plt.subplots(figsize=figsize)

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    candle_width = 0.8 * (1 / max(len(df), 1))
    for _, row in df.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax.vlines(row.name, row["low"], row["high"],
                  color=color, linewidth=0.8)
        ax.vlines(
            row.name,
            min(row["open"], row["close"]),
            max(row["open"], row["close"]),
            color=color,
            linewidth=max(candle_width, 0.5),
        )

    palette = ["#2196F3", "#FF9800", "#9C27B0", "#00BCD4"]
    if ema_cols:
        for i, col in enumerate(ema_cols):
            if col in df.columns:
                ax.plot(df.index, df[col], label=col.replace("_", " "),
                        color=palette[i % len(palette)], linewidth=1.2, alpha=0.8)
    if vwap_col and vwap_col in df.columns:
        ax.plot(df.index, df[vwap_col], label="VWAP",
                color="#FFEB3B", linewidth=1.2, alpha=0.8)

    if markers:
        if "signals" in markers and markers["signals"] is not None:
            sigs = markers["signals"]
            if isinstance(sigs, pd.DataFrame) and "signal" in sigs.columns:
                buys = sigs[sigs["signal"] == 1]
                sells = sigs[sigs["signal"] == -1]
                if not buys.empty:
                    ax.scatter(buys.index, buys["close"] * 0.98,
                               marker="^", color="lime", s=60, zorder=5,
                               label="Buy")
                if not sells.empty:
                    ax.scatter(sells.index, sells["close"] * 1.02,
                               marker="v", color="orange", s=60, zorder=5,
                               label="Sell")
        if "points" in markers and markers["points"]:
            for label, pts in markers["points"].items():
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, label=label, s=40, zorder=5)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, alpha=0.3)
    if ema_cols or vwap_col or (markers and markers.get("points")):
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
    return fig


def plot_indicator_panel(df, indicators, figsize=(14, 8)):
    """Create a multi-panel chart showing price chart + indicator sub-panels."""
    n = max(len(indicators), 1)
    if not indicators:
        indicators = []

    fig, axes = plt.subplots(n + 1, 1, figsize=figsize, sharex=True,
                             height_ratios=[3] + [1] * n)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = list(axes)

    # Main price panel
    ax_main = axes[0]
    candle_width = 0.8 * (1 / max(len(df), 1))
    for _, row in df.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax_main.vlines(row.name, row["low"], row["high"],
                       color=color, linewidth=0.8)
        ax_main.vlines(
            row.name,
            min(row["open"], row["close"]),
            max(row["open"], row["close"]),
            color=color,
            linewidth=max(candle_width, 0.5),
        )
    for col in ["ema_169", "vwap"]:
        if col in df.columns:
            ax_main.plot(df.index, df[col], label=col.upper(), linewidth=1)
    ax_main.set_title("Price Chart", fontsize=12)
    ax_main.set_ylabel("Price")
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.grid(True, alpha=0.3)

    # Indicator panels
    for i, ind in enumerate(indicators):
        ax = axes[i + 1]
        if ind in df.columns:
            ax.plot(df.index, df[ind], color="blue", linewidth=1)
            if ind.startswith("rsi"):
                ax.axhline(70, color="red", linestyle="--", alpha=0.5)
                ax.axhline(30, color="green", linestyle="--", alpha=0.5)
                ax.set_ylim(0, 100)
            ax.set_title(ind, fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{ind} (unavailable)", fontsize=9)

    ax_main.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def dataframe_to_markdown(df, max_rows=50):
    """Convert a DataFrame to a Gradio-safe markdown string."""
    if df is None or df.empty:
        return "_No data available_"
    if len(df) > max_rows:
        df = df.tail(max_rows).copy()
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].astype(float).round(4)
    try:
        # Rich markdown via tabulate (Github style).
        return df.to_markdown(index=True)
    except ImportError:
        # Degrade gracefully if tabulate isn't installed.
        cols = list(df.columns)
        header = f"| | {' | '.join(map(str, cols))} |\n"
        sep = f"|---|{'---|' * len(cols)}\n"
        rows = []
        for idx, (label, row) in enumerate(df.iterrows()):
            cells = " | ".join(_fmt_cell(v) for v in row)
            rows.append(f"| {label} | {cells} |")
        return header + sep + "\n".join(rows)


def _fmt_cell(value) -> str:
    """Format a single value as a markdown table cell."""
    try:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, pd.Int64Dtype)):
            return str(value)
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_number(val, decimals=2):
    """Format a number for human display."""
    f = safe_float(val)
    if abs(f) >= 1000000:
        return f"{f / 1000000:.2f}M"
    if abs(f) >= 1000:
        return f"{f / 1000:.2f}K"
    return f"{f:.{decimals}f}"


from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for Gradio
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))
