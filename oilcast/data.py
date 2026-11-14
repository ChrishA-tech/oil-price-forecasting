"""
Real Brent crude data: FRED series POILBREUSDM (IMF, global price of Brent crude,
monthly average, USD per barrel). Downloaded once and cached to data/brent_monthly.csv.

    from oilcast.data import load_brent
    df = load_brent()                 # cached CSV, else download from FRED
    df = load_brent(refresh=True)     # force re-download

Source: https://fred.stlouisfed.org/series/POILBREUSDM (public domain / citation requested)
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from .exceptions import TimeSeriesValidationError

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "brent_monthly.csv"


def parse_fred_csv(text: str, series: str = "POILBREUSDM") -> pd.DataFrame:
    """Parse FRED's CSV ('observation_date' or 'DATE' column; '.' marks missing values)."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), None)
    if date_col is None or series not in df.columns:
        raise TimeSeriesValidationError(f"Unexpected FRED CSV columns: {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]), "brent_usd": pd.to_numeric(df[series], errors="coerce")})
    return clean_monthly(out)


def clean_monthly(df: pd.DataFrame, value_col: str = "brent_usd") -> pd.DataFrame:
    """
    Validate and regularise a monthly series:
      - dates normalised to month start, sorted, duplicates rejected
      - leading/trailing missing values dropped; interior gaps rejected (not silently filled)
      - non-positive prices rejected
    """
    if df.empty:
        raise TimeSeriesValidationError("Empty series.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date")
    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m").tolist()[:5]
        raise TimeSeriesValidationError(f"Duplicate months in series: {dups}")
    valid = df[value_col].notna()
    df = df.loc[valid.idxmax(): valid[::-1].idxmax()] if valid.any() else df.iloc[0:0]
    if df.empty:
        raise TimeSeriesValidationError("Series has no valid observations.")
    full = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="MS")
    if len(full) != len(df) or df[value_col].isna().any():
        missing = sorted(set(full) - set(df.loc[df[value_col].notna(), "date"]))
        raise TimeSeriesValidationError(
            f"{len(missing)} missing month(s), e.g. {[d.strftime('%Y-%m') for d in missing[:5]]}")
    if (df[value_col] <= 0).any():
        raise TimeSeriesValidationError("Prices must be strictly positive.")
    return df.reset_index(drop=True)


MANUAL = DATA_DIR / "manual" / "POILBREUSDM.csv"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
              "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")


def download_fred(series: str = "POILBREUSDM", timeout: int = 90, attempts: int = 3) -> str:
    """Download with retries and back-off (FRED can be slow to respond to scripts)."""
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(FRED_URL.format(series=series), headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # timeout, DNS, HTTP errors
            last = e
            time.sleep(3 * (i + 1))
    raise TimeSeriesValidationError(
        f"Could not download {series} from FRED after {attempts} attempts ({type(last).__name__}: {last}). "
        f"Manual option: open {FRED_URL.format(series=series)} in a browser and save the file as {MANUAL}")


def load_brent(refresh: bool = False, start: Optional[str] = None, cache: Path = CACHE) -> pd.DataFrame:
    """Monthly Brent prices (date, brent_usd). Uses the cached CSV unless refresh=True or missing."""
    if refresh or not cache.exists():
        if MANUAL.exists() and not refresh:
            df = parse_fred_csv(MANUAL.read_text(encoding="utf-8-sig"))      # browser-downloaded copy
        else:
            df = parse_fred_csv(download_fred())
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, date_format="%Y-%m-%d")
    df = clean_monthly(pd.read_csv(cache, parse_dates=["date"]))
    if start:
        df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return df
