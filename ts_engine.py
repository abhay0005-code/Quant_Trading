"""
Time-Series Engine: ARIMA (return forecast), GARCH (volatility forecast),
Kalman filter (trend/state).

Each component degrades gracefully if its library is unavailable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("ts_engine")


def arima_forecast(returns: pd.Series, steps: int = 1) -> float:
    """Forecast the next *steps*-period return using ARIMA(1,0,1)."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        r = returns.dropna().iloc[-300:]
        if len(r) < 50:
            return 0.0
        res = ARIMA(r.reset_index(drop=True), order=(1, 0, 1)).fit()
        return float(res.forecast(steps=steps).iloc[-1])
    except Exception as e:
        log.warning("ARIMA failed: %s", e)
        return 0.0


def garch_volatility(returns: pd.Series) -> float:
    """One-step-ahead volatility forecast from GARCH(1,1)."""
    try:
        from arch import arch_model
        r = returns.dropna().iloc[-300:] * 100  # percent returns
        if len(r) < 50:
            return float(r.std() / 100) if len(r) else 0.0
        res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
        var = float(res.forecast(horizon=1).variance.iloc[-1, 0])
        return (var ** 0.5) / 100
    except Exception as e:
        log.warning("GARCH failed: %s", e)
        return float(returns.dropna().tail(20).std()) if len(returns) else 0.0


def kalman_trend(closes: pd.Series):
    """Return (filtered_trend, slope) of the price series via a local-level
    Kalman filter with hidden slope (pykalman)."""
    try:
        from pykalman import KalmanFilter
        obs = closes.dropna().values
        if len(obs) < 30:
            return float(closes.iloc[-1]), 0.0
        kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],
            observation_matrices=[[1, 0]],
            initial_state_mean=[obs[0], 0],
            transition_covariance=[[1e-5, 0], [0, 1e-5]],
            observation_covariance=1.0,
        )
        means, _ = kf.filter(obs)
        return float(means[-1, 0]), float(means[-1, 1])
    except Exception as e:
        log.warning("Kalman failed: %s", e)
        last = float(closes.iloc[-1])
        return last, float(closes.diff().tail(5).mean())


def run_time_series_engine(df: pd.DataFrame) -> dict:
    """Run all three components on the feature frame; return a summary dict."""
    closes = df["close"]
    returns = closes.pct_change()
    trend, slope = kalman_trend(closes)
    return {
        "arima_return": arima_forecast(returns),
        "garch_vol": garch_volatility(returns),
        "kalman_trend": trend,
        "kalman_slope": slope,
    }
