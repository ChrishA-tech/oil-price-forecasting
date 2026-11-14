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
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from .exceptions import TimeSeriesValidationError

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "brent_monthly.csv"
SOURCE_FILE = DATA_DIR / "brent_source.txt"
# DBnomics mirrors the IMF Primary Commodity Prices dataset, the same series FRED republishes as POILBREUSDM.
DBNOMICS_URL = ("https://api.db.nomics.world/v22/series/IMF/PCPS/M.W00.POILBRE.USD"
                "?observations=1&format=json")
EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls"
SOURCE_EIA = "US EIA Europe Brent Spot Price FOB (RBRTE), monthly average USD/bbl"
MAX_STALE_DAYS = 270      # a source whose last month is older than this is skipped if a fresher one exists
SOURCE_FRED = "FRED POILBREUSDM (IMF Brent crude, monthly average USD/bbl)"
SOURCE_DBNOMICS = "IMF PCPS POILBRE via DBnomics (same IMF series as FRED POILBREUSDM; monthly average USD/bbl)"


def parse_fred_csv(text: str, series: str = "POILBREUSDM") -> pd.DataFrame:
    """Parse FRED's CSV ('observation_date' or 'DATE' column; '.' marks missing values)."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), None)
    if date_col is None or series not in df.columns:
        raise TimeSeriesValidationError(f"Unexpected FRED CSV columns: {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]), "brent_usd": pd.to_numeric(df[series], errors="coerce")})
    return clean_monthly(out)


def parse_dbnomics_json(text: str, value_col: str = "brent_usd") -> pd.DataFrame:
    """Parse a DBnomics v22 series response (series.docs[0].period / .value; 'NA' marks missing)."""
    try:
        doc = json.loads(text)["series"]["docs"][0]
        periods, values = doc["period"], doc["value"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise TimeSeriesValidationError(f"Unexpected DBnomics response ({type(e).__name__}: {e})")
    if len(periods) != len(values):
        raise TimeSeriesValidationError("DBnomics periods and values differ in length.")
    out = pd.DataFrame({"date": pd.to_datetime(pd.Series(periods).astype(str).str[:7], format="%Y-%m"),
                        value_col: pd.to_numeric(pd.Series(values).replace("NA", None), errors="coerce")})
    return clean_monthly(out, value_col)


def parse_eia_xls(content: bytes, value_col: str = "brent_usd") -> pd.DataFrame:
    """
    Parse EIA's history workbook (e.g. RBRTEm.xls). Layout-tolerant: on every sheet, keep rows whose first
    cell is a date and second cell a number, and use the sheet with the most such rows ('Data 1').
    """
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, engine="xlrd")
    except Exception as e:
        raise TimeSeriesValidationError(f"Could not read EIA workbook ({type(e).__name__}: {e})")
    best = None
    for sh in sheets.values():
        if sh.shape[1] < 2:
            continue
        d = pd.to_datetime(sh.iloc[:, 0], errors="coerce")
        v = pd.to_numeric(sh.iloc[:, 1], errors="coerce")
        ok = d.notna() & v.notna()
        if ok.sum() and (best is None or ok.sum() > len(best)):
            best = pd.DataFrame({"date": d[ok].values, value_col: v[ok].values})
    if best is None or len(best) < 24:
        raise TimeSeriesValidationError("EIA workbook has no usable monthly series.")
    return clean_monthly(best, value_col)


def is_stale(df: pd.DataFrame, max_days: int = MAX_STALE_DAYS, today: Optional[pd.Timestamp] = None) -> bool:
    today = today or pd.Timestamp.today().normalize()
    return (today - pd.Timestamp(df["date"].iloc[-1])).days > max_days


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


def _http_get(url: str, timeout: int, attempts: int, binary: bool = False):
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return body if binary else body.decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


def _sources(timeout: int, attempts: int):
    return [
        (SOURCE_EIA, lambda: parse_eia_xls(_http_get(EIA_URL, timeout, attempts, binary=True))),
        (SOURCE_FRED, lambda: parse_fred_csv(download_fred(timeout=timeout, attempts=attempts))),
        (SOURCE_DBNOMICS, lambda: parse_dbnomics_json(_http_get(DBNOMICS_URL, timeout, attempts))),
    ]


def download_brent(timeout: int = 60, attempts: int = 2):
    """
    Try EIA, then FRED, then DBnomics. A source whose data ends more than MAX_STALE_DAYS ago is only
    used if nothing fresher can be downloaded. Returns (DataFrame, source label).
    """
    errors, stale = [], []
    for label, fn in _sources(timeout, attempts):
        try:
            df = fn()
        except Exception as e:
            errors.append(f"{label.split(' (')[0]}: {type(e).__name__}: {e}")
            continue
        if not is_stale(df):
            return df, label
        stale.append((df, label))
        errors.append(f"{label.split(' (')[0]}: stale, ends {df['date'].iloc[-1]:%Y-%m}")
    if stale:
        df, label = max(stale, key=lambda t: t[0]["date"].iloc[-1])
        print(f"!! Using the freshest available Brent data ({label}, ends {df['date'].iloc[-1]:%Y-%m}). "
              + " | ".join(errors))
        return df, label
    raise TimeSeriesValidationError("Could not download Brent prices. " + " | ".join(errors)
                                    + f". Manual option: save {EIA_URL} or {FRED_URL.format(series='POILBREUSDM')} "
                                      f"from a browser as {MANUAL}")


def data_source() -> str:
    return SOURCE_FILE.read_text().strip() if SOURCE_FILE.exists() else SOURCE_FRED


def load_brent(refresh: bool = False, start: Optional[str] = None, cache: Path = CACHE) -> pd.DataFrame:
    """Monthly Brent prices (date, brent_usd). Uses the cached CSV unless refresh=True or missing."""
    if cache.exists() and not refresh:
        cached = clean_monthly(pd.read_csv(cache, parse_dates=["date"]))
        if is_stale(cached) and cache == CACHE and not MANUAL.exists():
            try:                                    # cached data is old: try to update, keep the cache on failure
                fresh, src = download_brent()
                if fresh["date"].iloc[-1] > cached["date"].iloc[-1]:
                    fresh.to_csv(cache, index=False, date_format="%Y-%m-%d")
                    SOURCE_FILE.write_text(src + "\n")
            except Exception as e:
                print(f"!! Cached Brent data ends {cached['date'].iloc[-1]:%Y-%m}; update failed ({e}). Using cache.")
    if refresh or not cache.exists():
        if MANUAL.exists() and not refresh:
            df, src = parse_fred_csv(MANUAL.read_text(encoding="utf-8-sig")), SOURCE_FRED   # browser copy
        else:
            df, src = download_brent()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, date_format="%Y-%m-%d")
        if cache == CACHE:
            SOURCE_FILE.write_text(src + "\n")
    df = clean_monthly(pd.read_csv(cache, parse_dates=["date"]))
    if start:
        df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return df
