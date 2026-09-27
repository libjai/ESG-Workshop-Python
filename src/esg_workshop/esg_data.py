"""ESG data sourcing (investment universe, ESG scores) and ESG screening.

Note on Yahoo Finance: Yahoo displayed Sustainalytics ESG Risk Ratings from 2018 until late 2025.
Those endpoints now return HTTP 404 (verified 2026-09-27). The `yahoo_*` functions are kept as a
documented reference of the method; use `yahoo_esg_available()` before relying on them.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YAHOO_ESG_CHART = "https://query2.finance.yahoo.com/v1/finance/esgChart"
YAHOO_QUOTE_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}


# ------------------------------------------------------------------ universe
def sp500_constituents(session: requests.Session | None = None) -> pd.DataFrame:
    """Current S&P 500 constituents (ticker, name, GICS sector/sub-industry) from Wikipedia."""
    s = session or requests.Session()
    html = s.get(SP500_URL, headers=HEADERS, timeout=30).text
    df = pd.read_html(StringIO(html))[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector", "sub_industry"]]


def load_sample_universe(path: str | Path | None = None) -> pd.DataFrame:
    """Anonymized 10-stock universe (Mexican large caps, 2012-2013) with an illustrative 0-10 ESG score."""
    path = Path(path) if path else DATA_DIR / "sample_universe.csv"
    return pd.read_csv(path)


# ------------------------------------------------------ Yahoo (historical)
def yahoo_esg_available(ticker: str = "MSFT", session: requests.Session | None = None) -> dict:
    """Check whether Yahoo still serves ESG scores, using a non-ESG module as positive control."""
    s = session or requests.Session()
    s.headers.update(HEADERS)
    s.get("https://fc.yahoo.com", timeout=20)
    crumb = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=20).text
    status = {}
    for module in ("assetProfile", "esgScores"):
        r = s.get(YAHOO_QUOTE_SUMMARY.format(ticker=ticker), params={"modules": module, "crumb": crumb}, timeout=20)
        status[module] = r.status_code
    status["esgChart"] = s.get(YAHOO_ESG_CHART, params={"symbol": ticker, "crumb": crumb}, timeout=20).status_code
    status["esg_available"] = status["esgScores"] == 200
    return status


def yahoo_esg_chart(tickers, session: requests.Session | None = None) -> pd.DataFrame:
    """Method 1 (historical): Yahoo `esgChart` endpoint -> ESG score time series per ticker."""
    s = session or requests.Session()
    frames = []
    for t in tickers:
        r = s.get(YAHOO_ESG_CHART, params={"symbol": t}, headers=HEADERS, timeout=20)
        if not r.ok:
            continue
        result = (r.json().get("esgChart") or {}).get("result") or []
        series = result[0].get("symbolSeries") if result else None
        if series:
            df = pd.DataFrame(series)
            df["ticker"] = t
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


YAHOOQUERY_FIELDS = ["totalEsg", "environmentScore", "socialScore", "governanceScore", "ratingYear", "ratingMonth",
                     "highestControversy", "peerGroup", "percentile", "adult", "alcoholic", "animalTesting",
                     "controversialWeapons", "smallArms", "furLeather", "gambling", "gmo", "militaryContract",
                     "nuclear", "pesticides", "palmOil", "coal", "tobacco"]


def yahooquery_esg(tickers) -> pd.DataFrame:
    """Method 2 (historical): `yahooquery.Ticker(...).esg_scores` -> latest scores + product involvement flags."""
    from yahooquery import Ticker

    rows = {}
    for t in tickers:
        data = Ticker(t).esg_scores.get(t)
        if isinstance(data, dict) and "totalEsg" in data:
            rows[t] = {f: data.get(f) for f in YAHOOQUERY_FIELDS}
    return pd.DataFrame.from_dict(rows, orient="index")


def yfinance_esg(tickers) -> pd.DataFrame:
    """Method 3 (historical): `yfinance.Ticker(...).sustainability`."""
    import yfinance as yf

    frames = []
    for t in tickers:
        sus = yf.Ticker(t).sustainability
        if sus is not None and not sus.empty:
            row = sus.T
            row["ticker"] = t
            frames.append(row)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------- screening
def exclusion_screen(universe: pd.DataFrame, min_score: float | None = None, excluded_sectors=(),
                     score_col: str = "esg_score", sector_col: str = "sector") -> pd.DataFrame:
    """Negative screening: drop excluded sectors and names below a minimum ESG score."""
    out = universe[~universe[sector_col].isin(excluded_sectors)]
    if min_score is not None:
        out = out[out[score_col] >= min_score]
    return out


def best_in_class(universe: pd.DataFrame, top_fraction: float = 0.5, score_col: str = "esg_score",
                  sector_col: str = "sector") -> pd.DataFrame:
    """Positive screening: keep the top `top_fraction` of each sector by ESG score (at least one per sector)."""
    ranked = universe.copy()
    ranked["sector_pct_rank"] = ranked.groupby(sector_col)[score_col].rank(pct=True, ascending=True)
    keep = ranked["sector_pct_rank"] > 1 - top_fraction
    best = ranked.groupby(sector_col)[score_col].transform("max") == ranked[score_col]
    return ranked[keep | best]


def esg_tilt_weights(universe: pd.DataFrame, base_col: str = "market_cap_mn", score_col: str = "esg_score",
                     strength: float = 1.0) -> pd.Series:
    """ESG integration by tilting: w_i ∝ base_i * (score_i / mean score)^strength."""
    tilt = (universe[score_col] / universe[score_col].mean()).clip(lower=0) ** strength
    w = universe[base_col] * tilt
    return (w / w.sum()).rename("weight")
