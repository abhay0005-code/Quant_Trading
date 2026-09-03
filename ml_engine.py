"""
Quant ML Engine: trains XGBoost / LightGBM on engineered features to
produce P(UP) / P(DOWN) for the next bar(s).

Falls back to sklearn GradientBoosting when neither booster is available.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from features import FEATURE_COLS

log = logging.getLogger("ml_engine")

_MODEL_CACHE: dict = {}


def _build_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1 if the close *horizon* bars ahead is higher, else 0."""
    fwd = df["close"].shift(-horizon)
    return (fwd > df["close"]).astype(int)


def _make_model(prefer: str = "xgboost"):
    try:
        if prefer == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0)
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1)
    except Exception:
        log.info("%s unavailable, falling back to sklearn", prefer)
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=150, max_depth=3)


def train_and_predict(df: pd.DataFrame, engine: str = "xgboost",
                      train_window: int = 250, horizon: int = 1) -> dict:
    """
    Train on the last *train_window* rows and predict direction on the
    final row. Returns probabilities + feature importances.
    """
    data = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns]).copy()
    if len(data) < 60:
        return {"p_up": 0.5, "p_down": 0.5, "signal": "NEUTRAL",
                "model": "insufficient-data", "importance": {}}

    data["target"] = _build_target(data, horizon)
    data = data.dropna(subset=["target"])
    data = data[data["target"].notna()]

    train = data.iloc[-train_window:-horizon] if len(data) > train_window else data.iloc[:-horizon]
    latest = data.iloc[[-1]]

    X_train, y_train = train[FEATURE_COLS], train["target"].astype(int)
    X_pred = latest[FEATURE_COLS]

    key = (engine, len(X_train))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = _make_model(engine)
        model.fit(X_train, y_train)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model

    prob = model.predict_proba(X_pred)[0]
    classes = list(model.classes_)
    p_up = float(prob[classes.index(1)]) if 1 in classes else 0.5
    p_down = 1.0 - p_up

    imp = {}
    try:
        imp = dict(sorted(zip(FEATURE_COLS, map(float, model.feature_importances_)),
                          key=lambda kv: -kv[1])[:8])
    except Exception:
        pass

    signal = "BULLISH" if p_up >= 0.55 else ("BEARISH" if p_down >= 0.55 else "NEUTRAL")
    return {"p_up": p_up, "p_down": p_down, "signal": signal,
            "model": engine, "importance": imp}
