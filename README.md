# Project 3: Oil Price Time-Series Forecasting & Interactive Dashboard (ENVS456)

## Overview
- **Autoregressive `AR(12)` Model (`ar12_model.py`)**: Fits a 12-lag Autoregressive model (`statsmodels.tsa.ar_model.AutoReg(lags=12)`) on monthly Brent Crude Oil prices, performs Augmented Dickey-Fuller (ADF) stationarity diagnostics, computes out-of-sample RMSE, MAE, and MAPE, and produces multi-step ahead forecasts with 95% confidence intervals.
- **Interactive Plotly HTML Dashboard (`dashboard.py`)**: Builds a multi-panel interactive Plotly dashboard (Historical Series + AR(12) In-Sample Fit + Out-of-Sample Forecast + 95% Confidence Fan + 12-Lag Autoregressive Coefficient Bar Chart) and exports it directly to standalone HTML (`output/brent_ar12_dashboard.html`).
- **Jupyter Notebook (`notebooks/ENVS456_AR12_Oil_Forecasting.ipynb`)**: Self-contained reproducible academic notebook for ENVS456.
- **FastAPI Backend Service (`api.py`)**: Exposes `POST /api/v1/forecast-oil` for programmatic integration.
