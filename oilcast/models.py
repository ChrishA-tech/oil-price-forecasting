"""
Forecasting models with a common interface: fit(y) -> self, forecast(h) -> np.ndarray.

Baselines (hard to beat for commodity prices, so every model is compared to them):
  Naive          next value = last value (random walk)
  Drift          random walk with average historical drift
  SeasonalNaive  value 12 months earlier
Statistical models:
  AR12-levels    the original ENVS456 model: AutoReg(12) on price levels
  AR-returns     AutoReg on monthly log-returns, lag order chosen by AIC (0..12)
  ARIMA(1,1,1)   on log prices
  ETS-damped     Holt damped-trend exponential smoothing on log prices
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Type

import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .exceptions import TimeSeriesValidationError


class Forecaster:
    name = "base"

    def fit(self, y: np.ndarray) -> "Forecaster":
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) < self.min_obs:
            raise TimeSeriesValidationError(f"{self.name} needs at least {self.min_obs} observations, got {len(y)}.")
        if not np.isfinite(y).all() or (y <= 0).any():
            raise TimeSeriesValidationError(f"{self.name}: series must be finite and strictly positive.")
        self.y = y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit(y)
        return self

    def forecast(self, h: int) -> np.ndarray:
        if h < 1:
            raise TimeSeriesValidationError("Horizon must be >= 1.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(self._forecast(h), dtype=float)
        if out.shape != (h,) or not np.isfinite(out).all():
            raise TimeSeriesValidationError(f"{self.name} produced an invalid forecast.")
        return out

    min_obs = 2

    def _fit(self, y):
        pass

    def _forecast(self, h):
        raise NotImplementedError


class Naive(Forecaster):
    name = "Naive (random walk)"

    def _forecast(self, h):
        return np.repeat(self.y[-1], h)


class Drift(Forecaster):
    name = "Drift"

    def _forecast(self, h):
        slope = (self.y[-1] - self.y[0]) / (len(self.y) - 1)
        return self.y[-1] + slope * np.arange(1, h + 1)


class SeasonalNaive(Forecaster):
    name = "Seasonal naive (12m)"
    min_obs = 12

    def _forecast(self, h):
        return np.array([self.y[-12 + (k % 12)] for k in range(h)])


class AR12Levels(Forecaster):
    name = "AR(12) on levels (original)"
    min_obs = 30

    def _fit(self, y):
        self.res = AutoReg(y, lags=12, trend="c").fit()

    def _forecast(self, h):
        n = len(self.y)
        return self.res.predict(start=n, end=n + h - 1)


class ARReturnsAIC(Forecaster):
    name = "AR on log-returns (AIC)"
    min_obs = 40

    def _fit(self, y):
        r = np.diff(np.log(y))
        sel = ar_select_order(r, maxlag=12, ic="aic", trend="c")
        self.p = len(sel.ar_lags) if sel.ar_lags else 0
        self.res = AutoReg(r, lags=sel.ar_lags or 0, trend="c").fit()
        self.r = r

    def _forecast(self, h):
        n = len(self.r)
        r_hat = self.res.predict(start=n, end=n + h - 1)
        return self.y[-1] * np.exp(np.cumsum(r_hat))


class ARIMA111(Forecaster):
    name = "ARIMA(1,1,1) on log"
    min_obs = 30

    def _fit(self, y):
        self.res = ARIMA(np.log(y), order=(1, 1, 1)).fit()

    def _forecast(self, h):
        return np.exp(self.res.forecast(h))


class ETSDamped(Forecaster):
    name = "ETS damped trend on log"
    min_obs = 24

    def _fit(self, y):
        self.res = ExponentialSmoothing(np.log(y), trend="add", damped_trend=True).fit()

    def _forecast(self, h):
        return np.exp(self.res.forecast(h))


ALL_MODELS: List[Type[Forecaster]] = [Naive, Drift, SeasonalNaive, AR12Levels, ARReturnsAIC, ARIMA111, ETSDamped]
MODEL_BY_NAME: Dict[str, Type[Forecaster]] = {m.name: m for m in ALL_MODELS}
