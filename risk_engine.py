"""
Risk Engine: combines the signal score with volatility to produce
Stop-Loss, Target, and Position Size with R:R and max-loss constraints.
"""
from __future__ import annotations

import math


def signal_score(p_up: float, arima_return: float, kalman_slope: float,
                 choch: int = 0) -> float:
    """
    Composite score in [-1, +1]:
      50% ML probability, 25% ARIMA forecast, 20% Kalman slope, 5% CHoCH.
    """
    s = (0.50 * (2 * p_up - 1)
         + 0.25 * math.copysign(min(abs(arima_return) / 0.005, 1.0), arima_return)
         + 0.20 * math.copysign(min(abs(kalman_slope) / 0.05, 1.0), kalman_slope)
         + 0.05 * (1 if choch else 0))
    return max(-1.0, min(1.0, s))


def compute_trade_plan(price: float, atr: float, score: float,
                       equity: float, risk_pct: float = 0.01,
                       max_loss: float = 5000.0, rr: float = 2.0) -> dict:
    """
    Build a trade plan from ATR-based stops.

    Long when score > +0.15, short when score < -0.15, else no-trade.
    """
    if price <= 0 or atr <= 0 or abs(score) <= 0.15:
        return {"action": "NO TRADE", "score": score, "reason": "weak/no signal"}

    side = "LONG" if score > 0 else "SHORT"
    stop_dist = max(1.5 * atr, 0.001 * price)

    # Position size = risk budget / stop distance, capped by max-loss
    risk_budget = min(equity * risk_pct, max_loss)
    qty = max(int(risk_budget / stop_dist), 1)

    if side == "LONG":
        sl = price - stop_dist
        target = price + rr * stop_dist
    else:
        sl = price + stop_dist
        target = price - rr * stop_dist

    actual_risk = qty * stop_dist
    return {
        "action": side, "score": round(score, 3),
        "entry": round(price, 2), "stop_loss": round(sl, 2),
        "target": round(target, 2), "quantity": qty,
        "risk_per_unit": round(stop_dist, 2),
        "total_risk": round(actual_risk, 2),
        "reward_risk": f"1:{rr:g}",
        "atr": round(atr, 2),
    }
