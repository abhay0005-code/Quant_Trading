"""
Feature engineering for the 5-minute OHLCV pipeline.

Produces (per the architecture):
  - EMA(169) + EMA slope
  - ATR / volatility
  - VWAP (session-anchored)
  - RSI
  - Volume ratio
  - Swing High / Swing Low
  - BOS / CHoCH (market-structure flags)
  - Candle patterns (engulfing, doji, hammer)
  - Returns / momentum
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def add_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Return *df* with all engineered feature columns appended."""
    cfg = cfg or Config()
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── EMA 169 + slope ──
    df["ema_169"] = c.ewm(span=cfg.ema_span, adjust=False).mean()
    df["ema_slope"] = df["ema_169"].diff(3) / df["ema_169"].shift(3)

    # ── 9 EMA, 21 EMA, 200 EMA ──
    df["ema_9"] = c.ewm(span=9, adjust=False).mean()
    df["ema_21"] = c.ewm(span=21, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # ── ATR / volatility ──
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(cfg.atr_period).mean()
    df["volatility"] = df["atr"] / c

    # ── VWAP (session anchored) ──
    session = df.index.normalize()
    tp = (h + l + c) / 3
    cum_pv = (tp * v).groupby(session).cumsum()
    cum_v = v.groupby(session).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_v
    df["vwap_dist"] = (c - df["vwap"]) / df["vwap"]

    # ── RSI ──
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # ── Stochastic RSI ──
    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_rsi_k"] = ((df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # ── MACD (12, 26, 9) ──
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ── ROC (Rate of Change, 10-bar) ──
    df["roc"] = c.pct_change(10) * 100

    # ── ADX (14) ──
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    cond = plus_dm > minus_dm
    plus_dm = plus_dm.where(cond, 0)
    minus_dm = minus_dm.where(~cond, 0)
    atr_14 = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_14.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["adx"] = dx.ewm(span=14, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # ── Supertrend (10, 3) ──
    hl2 = (h + l) / 2
    up_band = hl2 - 3 * atr_14
    dn_band = hl2 + 3 * atr_14
    st = pd.Series(np.nan, index=df.index)
    st_dir = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > dn_band.iloc[i - 1]:
            st_dir.iloc[i] = 1
        elif c.iloc[i] < up_band.iloc[i - 1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i - 1]
        st.iloc[i] = up_band.iloc[i] if st_dir.iloc[i] == 1 else dn_band.iloc[i]
    df["supertrend"] = st
    df["supertrend_dir"] = st_dir

    # ── Volume ratio ──
    avg_vol = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = v / avg_vol

    # ── Swing high / low (fractal, ±3 bars) ──
    w = 3
    df["swing_high"] = (h == h.rolling(2 * w + 1, center=True).max()).astype(int)
    df["swing_low"] = (l == l.rolling(2 * w + 1, center=True).min()).astype(int)
    df["last_swing_high"] = h.where(df["swing_high"] == 1).ffill()
    df["last_swing_low"] = l.where(df["swing_low"] == 1).ffill()

    # ── BOS / CHoCH (simple structural breakout flags) ──
    df["bos_up"] = (c > df["last_swing_high"].shift()).astype(int)
    df["bos_down"] = (c < df["last_swing_low"].shift()).astype(int)
    prev_dir = np.sign(df["bos_up"] - df["bos_down"]).ffill().fillna(0)
    df["choch"] = (np.sign(prev_dir.diff().fillna(0)) != 0).astype(int)

    # ── Candle patterns ──
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    df["body_ratio"] = body / rng
    df["bull_engulf"] = ((c > o) & (c.shift() < o.shift())
                         & (c >= o.shift()) & (o <= c.shift())).astype(int)
    df["bear_engulf"] = ((c < o) & (c.shift() > o.shift())
                         & (c <= o.shift()) & (o >= c.shift())).astype(int)
    df["doji"] = (df["body_ratio"] < 0.1).astype(int)
    df["hammer"] = ((c > o) & ((min_col(o, c) - l) > 2 * body) & (df["body_ratio"] < 0.4)).astype(int)

    # ── Higher High / Higher Low, Lower High / Lower Low ──
    df["prev_swing_high"] = df["last_swing_high"].shift(1)
    df["prev_swing_low"] = df["last_swing_low"].shift(1)
    df["higher_high"] = ((df["last_swing_high"] > df["prev_swing_high"]).astype(int))
    df["higher_low"] = ((df["last_swing_low"] > df["prev_swing_low"]).astype(int))
    df["lower_high"] = ((df["last_swing_high"] < df["prev_swing_high"]).astype(int))
    df["lower_low"] = ((df["last_swing_low"] < df["prev_swing_low"]).astype(int))

    # ── Support / Resistance (rolling extremes, ±20 bars) ──
    df["support"] = l.rolling(20).min()
    df["resistance"] = h.rolling(20).max()

    # ── Previous Day High / Low ──
    daily = df.resample("D").agg({"high": "max", "low": "min"}).dropna()
    daily.columns = ["pd_high", "pd_low"]
    df = df.join(daily, how="left")
    df["pd_high"] = df["pd_high"].ffill()
    df["pd_low"] = df["pd_low"].ffill()

    # ── Opening Range High / Low (first 30 min = 6 bars of 5-min) ──
    session = df.index.normalize()
    or_group = df.groupby(session).cumcount()
    or_mask = or_group < 6
    or_high = h.where(or_mask).groupby(session).transform("max")
    or_low = l.where(or_mask).groupby(session).transform("min")
    df["or_high"] = or_high
    df["or_low"] = or_low

    # ── 5-minute High / Low (current bar) ──
    df["bar_high"] = h
    df["bar_low"] = l

    # ── Candle breakout (close above resistance or below support) ──
    df["breakout_up"] = (c > df["resistance"].shift()).astype(int)
    df["breakout_down"] = (c < df["support"].shift()).astype(int)

    # ── Retest (price returns to broken level) ──
    df["retest_support"] = ((df["breakout_down"].shift().rolling(5).max() == 1)
                            & (l <= df["support"]) & (c > df["support"])).astype(int)
    df["retest_resistance"] = ((df["breakout_up"].shift().rolling(5).max() == 1)
                               & (h >= df["resistance"]) & (c < df["resistance"])).astype(int)

    # ── Rejection candle (long wick, small body) ──
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    df["rejection_up"] = ((upper_wick > 2 * body) & (df["body_ratio"] < 0.3)).astype(int)
    df["rejection_down"] = ((lower_wick > 2 * body) & (df["body_ratio"] < 0.3)).astype(int)

    # ── Pin bar (long wick in one direction, small body) ──
    df["pin_bar_bull"] = ((lower_wick > 2.5 * body) & (upper_wick < body)).astype(int)
    df["pin_bar_bear"] = ((upper_wick > 2.5 * body) & (lower_wick < body)).astype(int)

    # ── Returns / momentum ──
    df["ret_1"] = c.pct_change()
    df["ret_5"] = c.pct_change(5)
    df["ret_15"] = c.pct_change(15)
    df["momentum"] = c - c.shift(10)
    df["ema_dist"] = (c - df["ema_169"]) / df["ema_169"]

    return df


def min_col(a, b):
    return pd.concat([a, b], axis=1).min(axis=1)


FEATURE_COLS = [
    "ema_slope", "volatility", "vwap_dist", "rsi", "volume_ratio",
    "body_ratio", "ret_1", "ret_5", "ret_15", "momentum", "ema_dist",
    "bos_up", "bos_down", "choch", "bull_engulf", "bear_engulf", "doji", "hammer",
    "higher_high", "higher_low", "lower_high", "lower_low",
    "breakout_up", "breakout_down", "retest_support", "retest_resistance",
    "rejection_up", "rejection_down", "pin_bar_bull", "pin_bar_bear",
    "ema_9", "ema_21", "ema_200", "stoch_rsi_k", "stoch_rsi_d",
    "macd_line", "macd_signal", "macd_hist", "roc", "adx", "plus_di", "minus_di",
    "supertrend", "supertrend_dir",
]
