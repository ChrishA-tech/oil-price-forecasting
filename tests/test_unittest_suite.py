import os
import tempfile
import unittest
from src.ar12_model import generate_brent_oil_series, OilPriceAR12Forecaster
from src.dashboard import export_interactive_plotly_dashboard
from src.exceptions import TimeSeriesValidationError


class TestOilPriceAR12(unittest.TestCase):
    def test_ar12_fit_forecast_and_plotly_html(self):
        df = generate_brent_oil_series(n_months=120, seed=42)
        forecaster = OilPriceAR12Forecaster(lags=12)
        res = forecaster.fit_and_forecast(df, horizon=12, test_size=12)
        self.assertGreater(res["metrics"]["rmse"], 0.0)
        self.assertEqual(len(res["forecast_df"]), 12)
        self.assertEqual(len(res["ar_coefficients"]), 13)

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "dashboard.html")
            export_interactive_plotly_dashboard(res, html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path, "r", encoding="utf-8") as f:
                html_text = f.read()
            self.assertIn("ENVS456", html_text)

    def test_validation_errors(self):
        forecaster = OilPriceAR12Forecaster(lags=12)
        short_df = generate_brent_oil_series(n_months=36).iloc[:20]
        with self.assertRaises(TimeSeriesValidationError):
            forecaster.fit_and_forecast(short_df)


if __name__ == "__main__":
    unittest.main()
