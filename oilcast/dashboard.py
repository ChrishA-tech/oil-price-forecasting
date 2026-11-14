import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any


def export_interactive_plotly_dashboard(results: Dict[str, Any], output_html_path: str) -> str:
    """
    Builds an interactive 2-panel Plotly HTML dashboard displaying:
    1. Historical Brent Crude prices, In-Sample AR(12) Fit, Out-of-Sample Test Predictions,
       and 12-Month Future Forecast with 95% Confidence Interval fan.
    2. AR(12) Lag Coefficient Profile (L1 through L12) & Model Diagnostics summary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)

    hist_df = results["historical_df"]
    fitted = results["in_sample_fitted"]
    fcast_df = results["forecast_df"]
    metrics = results["metrics"]
    ar_coeffs = {k: v for k, v in results["ar_coefficients"].items() if k != "const"}

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.12,
        subplot_titles=(
            f"Brent Crude Oil AR(12) Time-Series Forecast (RMSE: ${metrics['rmse']:.2f} | MAPE: {metrics['mape_pct']:.2f}% | AIC: {metrics['aic']:.1f})",
            "Estimated Autoregressive Lag Coefficients (AR Lags 1 to 12)",
        ),
    )

    # Historical Actuals
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["brent_usd"],
            mode="lines",
            name="Historical Brent Crude (USD/bbl)",
            line=dict(color="#1f2937", width=2.2),
        ),
        row=1,
        col=1,
    )

    # In-sample AR(12) fitted values
    fig.add_trace(
        go.Scatter(
            x=fitted.index,
            y=fitted.values,
            mode="lines",
            name="AR(12) In-Sample Fit",
            line=dict(color="#2563eb", width=1.6, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # 95% Confidence Interval Band
    fig.add_trace(
        go.Scatter(
            x=list(fcast_df["date"]) + list(fcast_df["date"])[::-1],
            y=list(fcast_df["ci_upper_95"]) + list(fcast_df["ci_lower_95"])[::-1],
            fill="toself",
            fillcolor="rgba(220, 38, 38, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Interval",
        ),
        row=1,
        col=1,
    )

    # Future AR(12) Forecast
    fig.add_trace(
        go.Scatter(
            x=fcast_df["date"],
            y=fcast_df["forecast_usd"],
            mode="lines+markers",
            name="AR(12) Out-of-Sample Forecast",
            line=dict(color="#dc2626", width=2.8),
        ),
        row=1,
        col=1,
    )

    # Lag Coefficients Bar Chart
    fig.add_trace(
        go.Bar(
            x=list(ar_coeffs.keys()),
            y=list(ar_coeffs.values()),
            name="AR(12) Lag Weights",
            marker_color="#059669",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="ENVS456: Brent Crude Oil Autoregressive AR(12) Forecasting Dashboard",
        template="plotly_white",
        height=820,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (USD / Barrel)", row=1, col=1)
    fig.update_yaxes(title_text="Coefficient Weight", row=2, col=1)

    fig.write_html(output_html_path, include_plotlyjs="cdn", full_html=True)
    return output_html_path


def export_comparison_dashboard(df, res: Dict[str, Any], output_html_path: str) -> str:
    """
    Two-panel dashboard for the model comparison run (oilcast.run):
      1. Brent history and the forecast of the best out-of-sample model
      2. Out-of-sample RMSE relative to the naive random walk, per model and horizon
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    bt, fc = res["backtest"], res["forecast"]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.14,
                        subplot_titles=(f"Brent crude, monthly (USD/bbl) and {fc['model']} forecast",
                                        f"Out-of-sample RMSE relative to naive random walk "
                                        f"({bt['n_origins']} rolling origins; below 1 = better)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_usd"], mode="lines", name="Brent (actual)",
                             line=dict(color="#1f2937", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc["dates"], y=fc["values"], mode="lines+markers", name=f"Forecast: {fc['model']}",
                             line=dict(color="#dc2626", width=2.5)), row=1, col=1)
    for m, v in bt["models"].items():
        if "rel_rmse_vs_naive" not in v["horizons"][str(bt["horizons"][0])]:
            continue
        fig.add_trace(go.Bar(x=[f"h={h}" for h in bt["horizons"]],
                             y=[v["horizons"][str(h)]["rel_rmse_vs_naive"] for h in bt["horizons"]], name=m),
                      row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#6b7280", row=2, col=1)
    title = "Brent crude oil forecasting: model comparison"
    if res["data"].get("synthetic"):
        title += " (SYNTHETIC DEMO DATA)"
    fig.update_layout(title=title, template="plotly_white", height=860, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="USD / barrel", row=1, col=1)
    fig.update_yaxes(title_text="RMSE / naive RMSE", row=2, col=1)
    fig.write_html(output_html_path, include_plotlyjs="cdn", full_html=True)
    return output_html_path
