import numpy as np
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from .exceptions import TimeSeriesValidationError


def generate_brent_oil_series(n_months: int = 180, seed: int = 42) -> pd.DataFrame:
    """Generates realistic monthly Brent Crude Oil prices (USD/bbl) with 12-month seasonal and cyclical structure."""
    if n_months < 36:
        raise TimeSeriesValidationError("Requires at least 36 monthly observations for AR(12) modeling.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2011-01-01", periods=n_months, freq="MS")

    prices = np.zeros(n_months, dtype=float)
    prices[:12] = 75.0 + rng.normal(0, 3.0, size=12)
    ar_weights = np.array([0.72, 0.14, -0.06, 0.03, 0.02, -0.02, 0.01, 0.01, -0.01, 0.02, 0.03, 0.06])

    for t in range(12, n_months):
        lags = prices[t - 12:t][::-1]
        seasonal = 2.5 * np.sin(2 * np.pi * t / 12.0)
        prices[t] = np.clip(3.8 + np.dot(ar_weights, lags) + seasonal + rng.normal(0, 2.8), 25.0, 135.0)

    return pd.DataFrame({"date": dates, "brent_usd": np.round(prices, 2)})


class OilPriceAR12Forecaster:
    """Encapsulates AR(12) fitting, ADF diagnostics, rolling evaluation, and confidence interval forecasting."""

    def __init__(self, lags: int = 12):
        if lags < 1:
            raise TimeSeriesValidationError("Lag order must be a positive integer.")
        self.lags = lags
        self.fitted_model = None
        self.train_series: pd.Series | None = None

    def validate_series(self, df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame) or "brent_usd" not in df.columns or "date" not in df.columns:
            raise TimeSeriesValidationError("Input DataFrame must contain 'date' and 'brent_usd' columns.")
        if df["brent_usd"].isna().any():
            raise TimeSeriesValidationError("Input 'brent_usd' series contains NaN values.")
        if (df["brent_usd"] <= 0).any():
            raise TimeSeriesValidationError("Oil prices must be strictly positive.")
        min_required = self.lags * 2 + 6
        if len(df) < min_required:
            raise TimeSeriesValidationError(
                f"Series length ({len(df)}) is shorter than minimum required ({min_required}) for AR({self.lags})."
            )
        try:
            idx = pd.to_datetime(df["date"])
        except (ValueError, TypeError) as exc:
            raise TimeSeriesValidationError(f"Unparseable dates: {exc}") from exc
        s = pd.Series(df["brent_usd"].values, index=idx, name="brent_usd").sort_index()
        if s.index.duplicated().any():
            raise TimeSeriesValidationError("Duplicate dates in series.")
        s = s.asfreq("MS")
        if s.isna().any():
            raise TimeSeriesValidationError(
                f"Series is not a gap-free monthly (month-start) series: {int(s.isna().sum())} missing month(s).")
        return s

    def run_adf_test(self, series: pd.Series) -> Dict[str, Any]:
        adf_stat, p_val, used_lag, nobs, crit_vals, _ = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": round(float(adf_stat), 4),
            "p_value": round(float(p_val), 4),
            "used_lag": int(used_lag),
            "n_obs": int(nobs),
            "is_stationary_5pct": bool(p_val < 0.05),
            "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
        }

    def fit_and_forecast(self, df: pd.DataFrame, horizon: int = 12, test_size: int = 12) -> Dict[str, Any]:
        if horizon < 1 or test_size < 1:
            raise TimeSeriesValidationError("Forecast horizon and test_size must be >= 1.")
        series = self.validate_series(df)
        if len(series) - test_size < self.lags * 2 + 2:
            raise TimeSeriesValidationError("Insufficient training samples after train/test split.")

        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # Fit on training split for out-of-sample evaluation
        eval_model = AutoReg(train, lags=self.lags).fit()
        test_pred = eval_model.predict(start=len(train), end=len(series) - 1, dynamic=False)

        rmse = float(np.sqrt(np.mean((test.values - test_pred.values) ** 2)))
        mae = float(np.mean(np.abs(test.values - test_pred.values)))
        mape = float(np.mean(np.abs((test.values - test_pred.values) / test.values)) * 100.0)

        # Refit on full series for future horizon forecast + 95% CI
        full_model = AutoReg(series, lags=self.lags).fit()
        self.fitted_model = full_model
        self.train_series = series

        forecast_res = full_model.get_prediction(start=len(series), end=len(series) + horizon - 1)
        mean_forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        future_dates = pd.date_range(
            start=series.index[-1] + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast_usd": np.round(mean_forecast.values, 2),
            "ci_lower_95": np.round(conf_int.iloc[:, 0].values, 2),
            "ci_upper_95": np.round(conf_int.iloc[:, 1].values, 2),
        })

        ar_params = {
            k: round(float(v), 4)
            for k, v in full_model.params.items()
        }

        return {
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mape_pct": round(mape, 4),
                "aic": round(float(full_model.aic), 3),
                "bic": round(float(full_model.bic), 3),
            },
            "adf_diagnostics": self.run_adf_test(series),
            "ar_coefficients": ar_params,
            "historical_df": df.copy(),
            "in_sample_fitted": full_model.fittedvalues,
            "test_actual": test,
            "test_predicted": test_pred,
            "forecast_df": forecast_df,
        }
