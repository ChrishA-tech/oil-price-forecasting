import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.ar12_model import generate_brent_oil_series, OilPriceAR12Forecaster
from src.dashboard import export_interactive_plotly_dashboard
from src.exceptions import TimeSeriesValidationError
from src.api import app

client = TestClient(app)


def test_ar12_model_fitting_metrics_and_html_export(tmp_path):
    df = generate_brent_oil_series(n_months=120, seed=42)
    forecaster = OilPriceAR12Forecaster(lags=12)
    res = forecaster.fit_and_forecast(df, horizon=12, test_size=12)

    assert "rmse" in res["metrics"] and res["metrics"]["rmse"] > 0
    assert len(res["forecast_df"]) == 12
    assert len(res["ar_coefficients"]) == 13  # const + 12 lags

    # Verify confidence intervals bracket the point forecast
    f_df = res["forecast_df"]
    assert (f_df["ci_lower_95"] < f_df["forecast_usd"]).all()
    assert (f_df["ci_upper_95"] > f_df["forecast_usd"]).all()

    # Verify HTML Plotly export
    html_file = tmp_path / "brent_ar12_dashboard.html"
    out_path = export_interactive_plotly_dashboard(res, str(html_file))
    assert os.path.exists(out_path)
    content = html_file.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    assert "ENVS456" in content


def test_ar12_validation_errors_on_short_or_corrupt_series():
    forecaster = OilPriceAR12Forecaster(lags=12)
    short_df = generate_brent_oil_series(n_months=36).iloc[:20]
    with pytest.raises(TimeSeriesValidationError, match="shorter than minimum required"):
        forecaster.fit_and_forecast(short_df)

    nan_df = generate_brent_oil_series(n_months=60)
    nan_df.loc[5, "brent_usd"] = float("nan")
    with pytest.raises(TimeSeriesValidationError, match="contains NaN"):
        forecaster.fit_and_forecast(nan_df)


def test_fastapi_oil_forecast_endpoint():
    df = generate_brent_oil_series(n_months=60, seed=7)
    records = df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    resp = client.post("/api/v1/forecast-oil", json={"observations": records, "horizon": 6})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecasts"]) == 6
    assert "brent_usd.L12" in body["ar_coefficients"]
