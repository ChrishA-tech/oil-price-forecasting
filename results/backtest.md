**Data:** IMF PCPS POILBRE via DBnomics (same IMF series as FRED POILBREUSDM; monthly average USD/bbl), 2000-01 to 2025-06 (306 months). **Backtest:** 60 rolling origins, expanding window, run 2026-11-14.

Out-of-sample RMSE (USD/bbl) by horizon; in brackets: RMSE relative to the naive random walk (<1 = better). ✓ = significantly better than naive (Diebold-Mariano, p<0.05).

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Naive (random walk) | 5.62 | 10.69 | 16.14 | 23.58 |
| Drift | 5.62 (1.00) | 10.72 (1.00) | 16.20 (1.00) | 23.82 (1.01) |
| Seasonal naive (12m) | 23.50 (4.18) | 23.67 (2.21) | 23.89 (1.48) | 23.58 (1.00) |
| AR(12) on levels (original) | 5.73 (1.02) | 10.89 (1.02) | 15.24 (0.94) | 19.93 (0.84) ✓ |
| AR on log-returns (AIC) | 5.62 (1.00) | 10.79 (1.01) | 16.46 (1.02) | 24.45 (1.04) |
| ARIMA(1,1,1) on log | 5.60 (1.00) | 10.66 (1.00) | 16.15 (1.00) | 23.53 (1.00) |
| ETS damped trend on log | 5.63 (1.00) | 10.73 (1.00) | 16.26 (1.01) | 23.90 (1.01) |

**Best 1-month model out of sample:** ARIMA(1,1,1) on log. Next-month forecast (2025-07): $71.85/bbl.
