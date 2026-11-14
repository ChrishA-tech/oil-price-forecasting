**Data:** US EIA Europe Brent Spot Price FOB (RBRTE), monthly average USD/bbl, 2000-01 to 2026-10 (322 months). **Backtest:** 60 rolling origins, expanding window, run 2026-11-14.

Out-of-sample RMSE (USD/bbl) by horizon; in brackets: RMSE relative to the naive random walk (<1 = better). ✓ = significantly better than naive (Diebold-Mariano, p<0.05).

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Naive (random walk) | 8.92 | 16.76 | 18.09 | 21.25 |
| Drift | 8.92 (1.00) | 16.80 (1.00) | 18.25 (1.01) | 22.11 (1.04) |
| Seasonal naive (12m) | 25.04 (2.81) | 24.76 (1.48) | 23.61 (1.30) | 21.25 (1.00) |
| AR(12) on levels (original) | 8.70 (0.97) | 17.20 (1.03) | 17.28 (0.95) | 18.48 (0.87) ✓ |
| AR on log-returns (AIC) | 8.41 (0.94) | 16.67 (0.99) | 18.51 (1.02) | 23.14 (1.09) |
| ARIMA(1,1,1) on log | 8.61 (0.96) | 16.93 (1.01) | 18.19 (1.00) | 21.57 (1.01) |
| ETS damped trend on log | 8.93 (1.00) | 16.82 (1.00) | 18.29 (1.01) | 21.94 (1.03) |

**Best 1-month model out of sample:** AR on log-returns (AIC). Next-month forecast (2026-11): $122.83/bbl.
