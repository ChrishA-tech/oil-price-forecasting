import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .ar12_model import OilPriceAR12Forecaster
from .exceptions import TimeSeriesValidationError

app = FastAPI(
    title="ENVS456 Oil Price AR(12) Forecasting API",
    version="1.0.0",
    description="REST API service for autoregressive AR(12) oil price forecasting with 95% confidence intervals.",
)


class OilObservation(BaseModel):
    date: str = Field(..., description="ISO date string YYYY-MM-01")
    brent_usd: float = Field(..., gt=0.0, description="Monthly Brent crude oil price in USD/bbl")


class OilForecastRequest(BaseModel):
    observations: List[OilObservation] = Field(..., min_length=32)
    horizon: int = Field(default=12, ge=1, le=36)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "oil-price-ar12-envs456"}


@app.post("/api/v1/forecast-oil")
def forecast_oil_endpoint(req: OilForecastRequest):
    df = pd.DataFrame([o.model_dump() for o in req.observations])
    forecaster = OilPriceAR12Forecaster(lags=12)
    try:
        results = forecaster.fit_and_forecast(df, horizon=req.horizon, test_size=6)
    except TimeSeriesValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, np.linalg.LinAlgError) as exc:   # e.g. bad dates, singular fit
        raise HTTPException(status_code=422, detail=f"Model could not be fitted: {exc}") from exc

    fcast_records = results["forecast_df"].assign(
        date=results["forecast_df"]["date"].dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")

    return {
        "metrics": results["metrics"],
        "adf_diagnostics": results["adf_diagnostics"],
        "ar_coefficients": results["ar_coefficients"],
        "forecasts": fcast_records,
    }
