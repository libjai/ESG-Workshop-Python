"""Minimum-variance portfolios with an ESG-score constraint and the risk-ESG frontier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRADING_DAYS = 252


def load_sample_prices(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "sample_prices.csv"
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def portfolio_stats(weights, returns: pd.DataFrame, esg: pd.Series, ppy: int = TRADING_DAYS) -> dict:
    """Annualized volatility and mean return, plus weighted ESG score: ESG_p = sum_i w_i * ESG_i."""
    w = np.asarray(weights)
    cov = returns.cov().values
    return {
        "volatility": float(np.sqrt(w @ cov @ w * ppy)),
        "expected_return": float(w @ returns.mean().values * ppy),
        "esg_score": float(w @ esg.values),
        "n_assets": int((w > 1e-4).sum()),
        "max_weight": float(w.max()),
    }


def min_variance(returns: pd.DataFrame, esg: pd.Series | None = None, min_esg: float | None = None,
                 esg_equality: bool = False, max_weight: float = 1.0) -> pd.Series:
    """Long-only minimum-variance portfolio.

    min_w  w' S w   s.t.  sum(w) = 1,  0 <= w_i <= max_weight,
                          [ESG_p >= min_esg]  or  [ESG_p == min_esg] if esg_equality.
    """
    cov = returns.cov().values
    n = cov.shape[0]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if min_esg is not None:
        e = esg.values
        kind = "eq" if esg_equality else "ineq"
        cons.append({"type": kind, "fun": lambda w, e=e: w @ e - min_esg})
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, max_weight)] * n, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise ValueError(f"Optimization failed (min_esg={min_esg}): {res.message}")
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=returns.columns, name="weight")


def esg_frontier(returns: pd.DataFrame, esg: pd.Series, n_points: int = 25, max_weight: float = 1.0) -> pd.DataFrame:
    """Risk-ESG efficient frontier: minimum volatility for each target ESG score,
    from the unconstrained MVP's ESG up to the highest attainable ESG."""
    mvp = min_variance(returns, max_weight=max_weight)
    lo = float(mvp @ esg.values)
    hi = float(esg.max()) if max_weight >= 1 else float(np.sort(esg.values)[::-1][: int(np.ceil(1 / max_weight))].mean())
    rows = []
    for target in np.linspace(lo, hi, n_points):
        try:
            w = min_variance(returns, esg, target, esg_equality=True, max_weight=max_weight)
        except ValueError:
            continue
        rows.append({"target_esg": target, **portfolio_stats(w, returns, esg), **w.add_prefix("w_").to_dict()})
    return pd.DataFrame(rows)
