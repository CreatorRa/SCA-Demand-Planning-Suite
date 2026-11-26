"""Demand models: promotion/price regression, baseline estimation, forecasts, backtest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from core.data import calendar_flags, clean_price, regular_price

HORIZON = 13
MODEL_NAMES = [
    "Price & promotion regression",
    "Exponential smoothing",
    "Moving median (13 weeks)",
    "Seasonal naive (same week last year)",
]


def mape(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    m = a > 0
    return float(np.mean(np.abs(a[m] - f[m]) / a[m]) * 100) if m.any() else np.nan


# ------------------------------------------------------------ regression model
@dataclass
class PromoModel:
    coef: pd.Series            # coefficients on log1p(units)
    r2: float
    regp_level: float          # regular price assumed for the future
    n_obs: int

    @property
    def uplift_per_10pct(self) -> float:
        """% change in units for a 10 % price cut below regular price."""
        return (np.exp(self.coef["discount"] * 0.10) - 1) * 100


def _design(weeks, disc, regp, t) -> pd.DataFrame:
    flags = calendar_flags(weeks)
    return pd.DataFrame(
        {
            "const": 1.0,
            "discount": np.asarray(disc, float),
            "log_regular_price": np.log(np.asarray(regp, float)),
            "trend": np.asarray(t, float),
            "black_friday": flags["black_friday"].values,
            "pre_xmas": flags["pre_xmas"].values,
        },
        index=pd.DatetimeIndex(weeks),
    )


def prepare(units: pd.Series, price: pd.Series) -> pd.DataFrame:
    """Per-week frame with price, regular price, discount depth and event flags.

    Gap handling:
      - weeks before launch (no sales and no price yet) are dropped
      - missing sales stay empty in 'units' (excluded from model fit and MAPE);
        'units_filled' interpolates them for the time-series models
      - missing prices are filled from neighbouring weeks (see clean_price)
    """
    active = (units.fillna(0) > 0) | price.notna()
    if active.any():
        units, price = units.loc[active.idxmax():], price.loc[active.idxmax():]
    p = clean_price(price)
    rp = regular_price(price)
    df = pd.DataFrame({"units": units, "price": price, "price_clean": p, "regular_price": rp})
    df["units_filled"] = units.interpolate(limit_direction="both")
    df["discount"] = (1 - p / rp).clip(lower=0)
    return df.join(calendar_flags(df.index))


def fit_promo_model(df: pd.DataFrame) -> PromoModel:
    X = _design(df.index, df["discount"], df["regular_price"], np.arange(len(df)))
    y = np.log1p(df["units"].values)
    ok = np.isfinite(y)                      # skip weeks with missing sales
    X, y = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    resid = y - X.values @ beta
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return PromoModel(
        coef=pd.Series(beta, index=X.columns),
        r2=float(r2),
        regp_level=float(df["regular_price"].iloc[-HORIZON:].median()),
        n_obs=len(df),
    )


def baseline_units(df: pd.DataFrame, model: PromoModel) -> pd.Series:
    """Units we would have expected at the regular price in each week.

    Model fit with discount set to 0: keeps regular price level, trend and
    holiday effects of that week, removes only the promotion. Comparing actual
    units with this baseline shows how much each individual promotion added.
    """
    X = _design(df.index, np.zeros(len(df)), df["regular_price"], np.arange(len(df)))
    return pd.Series(np.clip(np.expm1(X.values @ model.coef.values), 0, None), index=df.index)


def predict_promo_model(model: PromoModel, n_hist: int, future_weeks, future_discount) -> np.ndarray:
    t = np.arange(n_hist, n_hist + len(future_weeks))
    X = _design(future_weeks, future_discount, np.repeat(model.regp_level, len(future_weeks)), t)
    return np.clip(np.expm1(X.values @ model.coef.values), 0, None)


# ------------------------------------------------------------ all forecasts
def forecast(name: str, df: pd.DataFrame, future_weeks, future_discount=None) -> np.ndarray:
    """Forecast len(future_weeks) weeks after the end of df with model `name`."""
    h = len(future_weeks)
    y = df["units_filled"].values            # gaps interpolated
    if future_discount is None:
        future_discount = np.zeros(h)

    if name == "Price & promotion regression":
        return predict_promo_model(fit_promo_model(df), len(df), future_weeks, future_discount)
    if name == "Exponential smoothing":
        fit = SimpleExpSmoothing(np.log1p(y), initialization_method="estimated").fit()
        return np.clip(np.expm1(fit.forecast(h)), 0, None)
    if name == "Moving median (13 weeks)":
        return np.repeat(np.median(y[-13:]), h)
    if name == "Seasonal naive (same week last year)":
        hist = pd.Series(y, index=df.index)
        out = []
        for w in future_weeks:
            lag = hist.index[np.argmin(np.abs(hist.index - (pd.Timestamp(w) - pd.Timedelta(weeks=52))))]
            out.append(hist[lag])
        return np.array(out, float)
    raise ValueError(name)


def backtest(df: pd.DataFrame, holdout: int = HORIZON) -> pd.DataFrame:
    """Fit on all but the last `holdout` weeks, forecast them assuming no
    planned promotions (as in a real buying meeting), and score with MAPE."""
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    rows = []
    for name in MODEL_NAMES:
        fc = forecast(name, train, test.index)
        rows.append({"Model": name, "MAPE %": mape(test["units"], fc), "forecast": fc})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ promotion economics
def promo_table(df: pd.DataFrame, model: PromoModel, min_discount: float, cost_share: float) -> pd.DataFrame:
    """One row per promotion week with incremental units, revenue and gross profit.

    cost_share: unit cost as a share of the regular price (e.g. 0.55).
    """
    base = baseline_units(df, model)
    d = df.assign(baseline=base)
    d = d[(d["discount"] >= min_discount) & d["units"].notna()].copy()
    unit_cost = cost_share * d["regular_price"]
    d["extra_units"] = d["units"] - d["baseline"]
    d["revenue"] = d["units"] * d["price_clean"]
    d["baseline_revenue"] = d["baseline"] * d["regular_price"]
    d["extra_revenue"] = d["revenue"] - d["baseline_revenue"]
    d["gross_profit"] = d["units"] * (d["price_clean"] - unit_cost)
    d["baseline_gross_profit"] = d["baseline"] * (d["regular_price"] - unit_cost)
    d["extra_gross_profit"] = d["gross_profit"] - d["baseline_gross_profit"]
    margin = 1 - cost_share
    d["breakeven_uplift_%"] = np.where(
        margin > d["discount"], d["discount"] / (margin - d["discount"]) * 100, np.inf
    )
    d["actual_uplift_%"] = np.where(d["baseline"] > 0, (d["units"] / d["baseline"] - 1) * 100, np.nan)
    d["pays_off"] = d["extra_gross_profit"] > 0
    return d
