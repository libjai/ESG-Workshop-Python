import numpy as np
import pandas as pd
import pytest

from esg_workshop import esg_data as ed
from esg_workshop import portfolio as pf
from esg_workshop import temperature as tp


@pytest.fixture(scope="module")
def temps():
    return tp.load_monthly_temperature()["avg_temp_c"]


def test_temperature_data(temps):
    assert len(temps) == 2145
    assert temps.isna().sum() == 0
    assert temps.index.freqstr == "MS"


def test_warming_trend_positive(temps):
    trend = tp.warming_trend(temps)
    assert trend["trend_c_per_decade"] > 0
    assert trend["ci95_low"] > 0


def test_annual_cycle_detected(temps):
    assert abs(tp.dominant_periods(temps).period_months.iloc[0] - 12) < 0.5


def test_split_is_chronological(temps):
    train, test = tp.train_test_split_last(temps, 24)
    assert len(test) == 24 and train.index.max() < test.index.min()


def test_sarima_beats_seasonal_naive(temps):
    train, test = tp.train_test_split_last(temps["1992":], 24)
    fc = tp.forecast(train, 24, (1, 0, 1), (1, 1, 1, 12))
    assert fc.index.equals(test.index)
    assert tp.forecast_metrics(test, fc.forecast)["RMSE"] < tp.forecast_metrics(test, tp.seasonal_naive(train, 24))["RMSE"]


def test_degree_days():
    s = pd.Series([10.0, 25.0], index=pd.date_range("2020-01-01", periods=2, freq="MS"))
    dd = tp.monthly_degree_days(s, base_c=18)
    assert dd.HDD.tolist() == [31 * 8, 0] and dd.CDD.tolist() == [0, 29 * 7]


@pytest.fixture(scope="module")
def market():
    r = pf.daily_returns(pf.load_sample_prices())
    esg = ed.load_sample_universe().set_index("ticker").loc[r.columns, "esg_score"]
    return r, esg


def test_min_variance_weights(market):
    r, _ = market
    w = pf.min_variance(r)
    assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_esg_constraint_binds_and_costs_risk(market):
    r, esg = market
    mvp = pf.portfolio_stats(pf.min_variance(r), r, esg)
    constrained = pf.portfolio_stats(pf.min_variance(r, esg, min_esg=7), r, esg)
    assert constrained["esg_score"] >= 7 - 1e-6
    assert constrained["volatility"] >= mvp["volatility"]


def test_frontier_monotone(market):
    r, esg = market
    fr = pf.esg_frontier(r, esg, n_points=8)
    assert fr.volatility.is_monotonic_increasing


def test_screens():
    u = ed.load_sample_universe()
    assert "Basic Materials" not in ed.exclusion_screen(u, excluded_sectors=["Basic Materials"]).sector.values
    assert set(ed.best_in_class(u).sector) == set(u.sector)
    w = ed.esg_tilt_weights(u.set_index("ticker"), strength=0)
    assert np.allclose(w, u.market_cap_mn.values / u.market_cap_mn.sum())
