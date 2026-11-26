"""Data import and preparation.

Expected input: an Excel file with two sheets
  - sales sheet  (default name "Sales in unit")  : column "Week" + one column per series
  - price sheet  (default name "Prices in Euro") : same layout, prices in EUR
Series columns are named "<Region> <Product>", e.g. "North 1".
Rows at the end with empty sales are treated as the forecast horizon.
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

SALES_SHEET = "Sales in unit"
PRICE_SHEET = "Prices in Euro"


def _pick_sheet(sheets: dict[str, pd.DataFrame], name: str, position: int) -> pd.DataFrame:
    if name in sheets:
        return sheets[name]
    return list(sheets.values())[position]


def load_workbook(file) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    """Return (sales, prices, future_weeks).

    sales / prices: DataFrame indexed by week (history only), one column per series.
    future_weeks: list of week dates to forecast (empty rows in the file, or the
    next 13 weeks if the file has none).
    """
    if hasattr(file, "read"):
        file = io.BytesIO(file.read())
    sheets = pd.read_excel(file, sheet_name=None)
    sales = _pick_sheet(sheets, SALES_SHEET, 0).copy()
    prices = _pick_sheet(sheets, PRICE_SHEET, 1).copy()

    cleaned = []
    for df in (sales, prices):
        df["Week"] = pd.to_datetime(df["Week"], errors="coerce")
        df = df.dropna(subset=["Week"]).drop_duplicates("Week").set_index("Week").sort_index()
        df.columns = [str(c).strip() for c in df.columns]
        # text such as "n/a" or "-" and negative values become gaps
        df = df.apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
        cleaned.append(df)
    sales, prices = cleaned

    has_sales = sales.notna().any(axis=1)
    last_hist = has_sales[has_sales].index.max()
    future_weeks = list(sales.index[sales.index > last_hist])
    if not future_weeks:
        future_weeks = [last_hist + pd.Timedelta(weeks=i) for i in range(1, 14)]

    # complete weekly calendar: weeks missing from the file become empty rows (gaps)
    full = pd.date_range(sales.index.min(), last_hist, freq="7D")
    sales = sales.reindex(full)
    prices = prices.reindex(full)
    sales.index.name = prices.index.name = "Week"
    return sales, prices, future_weeks


def data_gaps(sales: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per series: missing sales weeks, missing price weeks, pre-launch weeks."""
    rows = []
    for c in sales.columns:
        u, p = sales[c], prices[c]
        active = (u.fillna(0) > 0) | p.notna()
        start = active.idxmax() if active.any() else u.index[0]
        rows.append({
            "Series": c,
            "Missing sales weeks": int(u.loc[start:].isna().sum()),
            "Missing price weeks": int(p.loc[start:].isna().sum()),
            "Weeks before launch": int((u.index < start).sum()),
        })
    return pd.DataFrame(rows)


def split_series_name(col: str) -> tuple[str, str]:
    """'North 3' -> ('North', '3')."""
    parts = col.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (col, "")


def clean_price(price: pd.Series) -> pd.Series:
    """Fill missing prices with the local median (missing price = no info, not free)."""
    filled = price.fillna(price.rolling(5, min_periods=1, center=True).median())
    return filled.ffill().bfill()


def regular_price(price: pd.Series, window: int = 13) -> pd.Series:
    """Regular (non-promoted) price: rolling 75th percentile of the observed price.

    Promotions are short and deep, so a high quantile over ~a quarter tracks the
    shelf price while ignoring promo weeks.
    """
    p = clean_price(price)
    return p.rolling(window, min_periods=3, center=True).quantile(0.75).bfill().ffill()


# ---------------------------------------------------------------- calendar events
def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _black_friday(year: int) -> date:
    nov1 = date(year, 11, 1)
    first_fri = nov1 + timedelta(days=(4 - nov1.weekday()) % 7)
    return first_fri + timedelta(weeks=3)


def calendar_flags(weeks) -> pd.DataFrame:
    """Event flags per week (week = Monday start date).

    black_friday : week containing Black Friday and the following (Cyber) week
    pre_xmas     : weeks starting 1-21 December
    easter       : week containing Good Friday
    """
    idx = pd.DatetimeIndex(weeks)
    out = pd.DataFrame(index=idx)
    bf, xm, ea = [], [], []
    for w in idx:
        d = w.date()
        end = d + timedelta(days=6)
        b = _black_friday(d.year)
        bf.append(d <= b <= end or d <= b + timedelta(days=7) <= end)
        xm.append(d.month == 12 and d.day <= 21)
        gf = _easter(d.year) - timedelta(days=2)
        ea.append(d <= gf <= end)
    out["black_friday"] = np.array(bf, dtype=float)
    out["pre_xmas"] = np.array(xm, dtype=float)
    out["easter"] = np.array(ea, dtype=float)
    return out


EVENT_LABELS = {"black_friday": "Black Friday / Cyber week", "pre_xmas": "Pre-Christmas", "easter": "Easter"}
