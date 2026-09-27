"""Monthly temperature analysis and ARIMA/SARIMA forecasting.

Data: Berkeley Earth, `GlobalLandTemperaturesByMajorCity` (monthly averages, degrees Celsius).
"""
from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_monthly_temperature(path: str | Path | None = None) -> pd.DataFrame:
    """Monthly series indexed by month start (freq 'MS') with columns avg_temp_c, uncertainty_c."""
    path = Path(path) if path else DATA_DIR / "cdmx_monthly_temperature.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.asfreq("MS")


def load_berkeley_city(csv_path: str | Path, city: str) -> pd.DataFrame:
    """Extract one city from the full Berkeley Earth file (GlobalLandTemperaturesByMajorCity.csv)."""
    raw = pd.read_csv(csv_path, parse_dates=["dt"])
    df = raw.loc[raw["City"] == city, ["dt", "AverageTemperature", "AverageTemperatureUncertainty"]]
    df.columns = ["date", "avg_temp_c", "uncertainty_c"]
    return df.set_index("date").sort_index().asfreq("MS")


def annual_means(series: pd.Series, min_months: int = 12) -> pd.Series:
    """Calendar-year means, keeping only complete years."""
    g = series.groupby(series.index.year)
    return g.mean()[g.count() >= min_months]


def warming_trend(series: pd.Series, start: int | None = None, end: int | None = None, hac_lags: int = 3) -> dict:
    """Linear trend of annual means in degrees C per decade, with Newey-West 95% CI.

    Model: T_y = a + b * year + e_y ;  trend per decade = 10 * b.
    """
    ann = annual_means(series)
    ann = ann.loc[(start or ann.index.min()):(end or ann.index.max())]
    X = sm.add_constant(ann.index.values.astype(float))
    res = sm.OLS(ann.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    b, se = res.params[1], res.bse[1]
    return {
        "period": f"{ann.index.min()}-{ann.index.max()}",
        "trend_c_per_decade": 10 * b,
        "ci95_low": 10 * (b - 1.96 * se),
        "ci95_high": 10 * (b + 1.96 * se),
        "p_value": res.pvalues[1],
        "n_years": len(ann),
    }


def anomaly_vs_baseline(series: pd.Series, baseline=(1951, 1980)) -> pd.Series:
    """Annual anomaly relative to a climatological baseline (Berkeley Earth uses 1951-1980)."""
    ann = annual_means(series)
    return (ann - ann.loc[baseline[0]:baseline[1]].mean()).rename("anomaly_c")


def stationarity_report(series: pd.Series) -> pd.Series:
    """ADF (H0: unit root) and KPSS (H0: stationary). Read them together."""
    x = series.dropna()
    adf = adfuller(x, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(x, regression="c", nlags="auto")
    return pd.Series({"n_obs": len(x), "adf_stat": adf[0], "adf_p": adf[1], "kpss_stat": kp[0], "kpss_p": kp[1]})


def dominant_periods(series: pd.Series, top: int = 3) -> pd.DataFrame:
    """Periodogram (FFT) of the demeaned series; returns the strongest periods in months."""
    x = series.dropna().values
    x = x - x.mean()
    amp = np.abs(np.fft.rfft(x)) * 2 / len(x)
    freq = np.fft.rfftfreq(len(x), d=1.0)
    amp, freq = amp[1:], freq[1:]
    idx = np.argsort(amp)[::-1][:top]
    return pd.DataFrame({"period_months": 1 / freq[idx], "amplitude_c": amp[idx]})


def train_test_split_last(series: pd.Series, test_months: int):
    """Chronological split: the last `test_months` observations form the test set."""
    return series.iloc[:-test_months], series.iloc[-test_months:]


def _fit(train, order, seasonal_order=(0, 0, 0, 0)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def grid_search_sarimax(train: pd.Series, p=range(3), d=range(2), q=range(3),
                        P=(0,), D=(0,), Q=(0,), s: int = 12) -> pd.DataFrame:
    """Fit every (p,d,q)(P,D,Q,s) combination on the TRAIN set and rank by AIC."""
    rows = []
    for order in product(p, d, q):
        for sorder in product(P, D, Q):
            seasonal = (*sorder, s) if any(sorder) else (0, 0, 0, 0)
            try:
                res = _fit(train, order, seasonal)
                rows.append({"order": order, "seasonal_order": seasonal, "aic": res.aic, "bic": res.bic})
            except Exception:
                continue
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def forecast(train: pd.Series, horizon: int, order, seasonal_order=(0, 0, 0, 0), alpha: float = 0.05) -> pd.DataFrame:
    """Out-of-sample forecast with confidence interval from the same fitted model."""
    res = _fit(train, order, seasonal_order)
    fc = res.get_forecast(steps=horizon)
    ci = fc.conf_int(alpha=alpha)
    return pd.DataFrame({"forecast": fc.predicted_mean, "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]})


def seasonal_naive(train: pd.Series, horizon: int, s: int = 12) -> pd.Series:
    """Benchmark: each month equals the same month of the last observed year."""
    last = train.iloc[-s:].values
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    return pd.Series(np.resize(last, horizon), index=idx, name="seasonal_naive")


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    e = actual.values - predicted.values
    return {"RMSE": float(np.sqrt(np.mean(e**2))), "MAE": float(np.mean(np.abs(e))), "Bias": float(np.mean(-e))}


def monthly_degree_days(series: pd.Series, base_c: float = 18.0) -> pd.DataFrame:
    """Approximate HDD/CDD from MONTHLY means: days_in_month * max(base - T, 0).

    Limitation: true degree days use daily temperatures; monthly means understate HDD/CDD
    whenever daily values cross the base temperature within the month.
    """
    days = series.index.days_in_month
    return pd.DataFrame({
        "HDD": days * np.maximum(base_c - series, 0),
        "CDD": days * np.maximum(series - base_c, 0),
    }, index=series.index)
