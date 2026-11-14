"""
Rolling-origin (expanding window) backtest with Diebold-Mariano tests.

For each forecast origin t (the last `n_origins` months), every model is fitted on
y[:t] only and forecasts h = 1..max(horizons) months ahead. Errors are collected per
horizon, so no model ever sees the data it is scored on.

Diebold-Mariano (squared-error loss, Harvey-Leybourne-Newbold small-sample
correction) tests each model against the naive random walk: a negative statistic
with p < 0.05 means the model is significantly MORE accurate than naive.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
from scipy import stats

from .exceptions import TimeSeriesValidationError
from .models import ALL_MODELS, Forecaster, Naive


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> Dict[str, float]:
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    if n < 8:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    d_mean = d.mean()
    # long-run variance with h-1 autocovariances (forecast errors overlap for h > 1)
    gamma = [np.mean((d[k:] - d_mean) * (d[:n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {"dm_stat": round(stat, 3), "p_value": round(p, 4)}


def rolling_backtest(y: Sequence[float], models: Optional[Iterable[Type[Forecaster]]] = None,
                     horizons: Sequence[int] = (1, 3, 6, 12), n_origins: int = 60,
                     min_train: int = 120) -> Dict:
    y = np.asarray(y, dtype=float)
    models = list(models or ALL_MODELS)
    max_h = max(horizons)
    first = len(y) - n_origins
    if first < min_train:
        raise TimeSeriesValidationError(
            f"Need at least {min_train + n_origins} observations for this backtest, got {len(y)}.")

    errors: Dict[str, Dict[int, List[float]]] = {m.name: {h: [] for h in horizons} for m in models}
    pct: Dict[str, Dict[int, List[float]]] = {m.name: {h: [] for h in horizons} for m in models}
    failures: Dict[str, int] = {m.name: 0 for m in models}
    targets: Dict[int, List[int]] = {h: [] for h in horizons}

    for origin in range(first, len(y)):
        train = y[:origin]
        for h in horizons:
            if origin + h - 1 < len(y):
                targets[h].append(origin + h - 1)
        for m in models:
            try:
                fc = m().fit(train).forecast(max_h)
            except Exception:
                failures[m.name] += 1
                fc = Naive().fit(train).forecast(max_h)   # count failure, score as naive
            for h in horizons:
                t = origin + h - 1
                if t < len(y):
                    e = y[t] - fc[h - 1]
                    errors[m.name][h].append(e)
                    pct[m.name][h].append(abs(e) / y[t] * 100)

    naive = Naive.name
    out: Dict = {"n_obs": len(y), "n_origins": n_origins, "horizons": list(horizons), "models": {}}
    for m in models:
        rows = {}
        for h in horizons:
            e = np.array(errors[m.name][h])
            en = np.array(errors[naive][h]) if naive in errors else None
            rmse = float(np.sqrt(np.mean(e ** 2)))
            row = {"n": len(e), "rmse": round(rmse, 3), "mae": round(float(np.mean(np.abs(e))), 3),
                   "mape": round(float(np.mean(pct[m.name][h])), 2)}
            if en is not None and m.name != naive:
                row["rel_rmse_vs_naive"] = round(rmse / float(np.sqrt(np.mean(en ** 2))), 3)
                row.update(diebold_mariano(e, en, h))
            rows[str(h)] = row
        out["models"][m.name] = {"horizons": rows, "fit_failures": failures[m.name]}
    return out
