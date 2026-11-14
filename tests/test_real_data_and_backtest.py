"""Tests for the real-data loader, model zoo and rolling backtest (Nov 2026 upgrade)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oilcast.backtest import diebold_mariano, rolling_backtest
from oilcast.data import clean_monthly, load_brent, parse_fred_csv
from oilcast.exceptions import TimeSeriesValidationError
from oilcast.models import ALL_MODELS, Forecaster, Naive

FX = Path(__file__).parent / "fixtures"


def test_parse_fred_csv_drops_trailing_missing():
    df = parse_fred_csv((FX / "fred_sample.csv").read_text())
    assert len(df) == 6 and df["date"].iloc[-1] == pd.Timestamp("2023-06-01")
    assert df["brent_usd"].iloc[0] == pytest.approx(83.0864)


def test_parse_fred_csv_old_header():
    txt = (FX / "fred_sample.csv").read_text().replace("observation_date", "DATE")
    assert len(parse_fred_csv(txt)) == 6


@pytest.mark.parametrize("dates,values,msg", [
    (["2023-01-01", "2023-01-15"], [1, 2], "Duplicate"),
    (["2023-01-01", "2023-03-01"], [1, 2], "missing month"),
    (["2023-01-01", "2023-02-01", "2023-03-01"], [1, np.nan, 2], "missing month"),
    (["2023-01-01", "2023-02-01"], [1, -2], "positive"),
])
def test_clean_monthly_rejects_bad_series(dates, values, msg):
    with pytest.raises(TimeSeriesValidationError, match=msg):
        clean_monthly(pd.DataFrame({"date": dates, "brent_usd": values}))


def test_load_brent_from_cache(tmp_path):
    cache = tmp_path / "b.csv"
    parse_fred_csv((FX / "fred_sample.csv").read_text()).to_csv(cache, index=False)
    assert len(load_brent(cache=cache)) == 6


@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_model_forecasts_finite_positive(model):
    y = 70 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.05, 150)))
    fc = model().fit(y).forecast(12)
    assert fc.shape == (12,) and np.isfinite(fc).all() and (fc > 0).all()


def test_models_reject_invalid_input():
    with pytest.raises(TimeSeriesValidationError):
        Naive().fit([1.0, np.nan, 2.0])
    with pytest.raises(TimeSeriesValidationError):
        Naive().fit([1.0, 2.0]).forecast(0)


class _Spy(Forecaster):
    name = "spy"
    seen = []

    def _fit(self, y):
        _Spy.seen.append(len(y))

    def _forecast(self, h):
        return np.repeat(self.y[-1], h)


def test_backtest_never_trains_on_future_data():
    y = np.arange(1, 201, dtype=float)
    _Spy.seen = []
    rolling_backtest(y, models=[Naive, _Spy], n_origins=30, horizons=(1, 3))
    assert _Spy.seen == list(range(170, 200))           # origin t trains on y[:t] only


def test_backtest_naive_on_linear_trend_has_exact_errors():
    y = np.arange(1, 201, dtype=float)                  # naive h-step error = h exactly
    r = rolling_backtest(y, models=[Naive], n_origins=30, horizons=(1, 3))
    assert r["models"][Naive.name]["horizons"]["1"]["rmse"] == 1.0
    assert r["models"][Naive.name]["horizons"]["3"]["rmse"] == 3.0


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    bench = rng.normal(0, 2, 200)
    better = rng.normal(0, 1, 200)
    res = diebold_mariano(better, bench, h=1)
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01


def test_backtest_requires_enough_data():
    with pytest.raises(TimeSeriesValidationError):
        rolling_backtest(np.ones(50) * 70.0, n_origins=60)


def test_run_synthetic_end_to_end(tmp_path, monkeypatch):
    import oilcast.run as run
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    res = run.main(["--synthetic", "--origins", "24"])
    assert res["data"]["synthetic"] and len(res["forecast"]["values"]) == 12
    assert (tmp_path / "backtest_synthetic.md").exists() and (tmp_path / "output" / "dashboard_synthetic.html").exists()


def test_api_rejects_gappy_series():
    from fastapi.testclient import TestClient
    from oilcast.api import app
    obs = [{"date": d.strftime("%Y-%m-01"), "brent_usd": 70.0 + i}
           for i, d in enumerate(pd.date_range("2018-01-01", periods=40, freq="MS")) if i != 20]
    r = TestClient(app).post("/api/v1/forecast-oil", json={"observations": obs, "horizon": 3})
    assert r.status_code == 400 and "missing month" in r.json()["detail"]


def test_load_brent_uses_manual_file_when_no_cache(tmp_path, monkeypatch):
    import oilcast.data as d
    manual = tmp_path / "manual" / "POILBREUSDM.csv"
    manual.parent.mkdir()
    manual.write_text((FX / "fred_sample.csv").read_text())
    monkeypatch.setattr(d, "MANUAL", manual)
    monkeypatch.setattr(d, "download_fred", lambda: (_ for _ in ()).throw(AssertionError("should not download")))
    assert len(d.load_brent(cache=tmp_path / "cache.csv")) == 6


def test_download_failure_message_explains_manual_option(monkeypatch):
    import oilcast.data as d
    monkeypatch.setattr(d.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow")))
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    with pytest.raises(TimeSeriesValidationError, match="Manual option"):
        d.download_fred(attempts=2)


def test_parse_dbnomics_json():
    from pathlib import Path
    from oilcast.data import parse_dbnomics_json
    txt = (Path(__file__).parent / "fixtures" / "dbnomics_brent_sample.json").read_text()
    df = parse_dbnomics_json(txt)
    assert len(df) == 3 and df["brent_usd"].iloc[0] == 21.25          # trailing NA dropped
    assert str(df["date"].iloc[-1].date()) == "1990-03-01"


def test_parse_dbnomics_rejects_bad_payload():
    from oilcast.data import parse_dbnomics_json
    from oilcast.exceptions import TimeSeriesValidationError
    with pytest.raises(TimeSeriesValidationError):
        parse_dbnomics_json('{"series": {"docs": []}}')


def test_download_brent_falls_back_to_fred(monkeypatch):
    import oilcast.data as d
    fred = "observation_date,POILBREUSDM\n1990-01-01,21.25\n1990-02-01,19.81\n"
    def fake_get(url, timeout, attempts):
        raise OSError("blocked")
    monkeypatch.setattr(d, "_http_get", fake_get)
    monkeypatch.setattr(d, "download_fred", lambda **k: fred)
    df, src = d.download_brent()
    assert len(df) == 2 and src == d.SOURCE_FRED
