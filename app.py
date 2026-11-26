"""Hafen & Home - Demand Planning Suite.

Run locally:  streamlit run app.py
Each tab answers one planner question. Calculations live in core/; this file is UI only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data import EVENT_LABELS, data_gaps, load_workbook, split_series_name
from core.models import (
    HORIZON,
    MODEL_NAMES,
    backtest,
    baseline_units,
    fit_promo_model,
    forecast,
    prepare,
    promo_table,
)

SAMPLE_FILE = Path(__file__).parent / "data" / "weekly_sales_price_data.xls"

# Validated categorical palette (dataviz reference palette, light mode)
C_ACTUAL = "#257de0"     # blue   - actual units
C_ALT = "#f47230"        # orange - baseline / forecast
C_3, C_4 = "#00b894", "#f2a600"
C_PRICE = "#666666"      # price line
C_GOOD, C_BAD = "#008a17", "#ec4647"
MODEL_COLORS = dict(zip(MODEL_NAMES, [C_ALT, C_3, C_4, "#6a4ce2"]))
EVENT_FILL = {"black_friday": "rgba(106,76,226,0.12)", "pre_xmas": "rgba(236,70,71,0.10)", "easter": "rgba(0,184,148,0.12)"}

st.set_page_config(page_title="Hafen & Home · Demand Planning", page_icon="📦", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load(file_bytes: bytes | None):
    import io
    src = io.BytesIO(file_bytes) if file_bytes else SAMPLE_FILE
    return load_workbook(src)


@st.cache_data(show_spinner=False)
def series_frame(units: pd.Series, price: pd.Series) -> pd.DataFrame:
    return prepare(units, price)


@st.cache_data(show_spinner=False)
def series_backtest(df: pd.DataFrame) -> pd.DataFrame:
    return backtest(df)


def base_layout(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        plot_bgcolor="white", legend=dict(orientation="h", y=1.08, x=0),
        font=dict(size=13, color="#191919"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cccccc")
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False, rangemode="tozero")
    return fig


def shade_events(fig, df, rows=(1,)):
    for flag, fill in EVENT_FILL.items():
        for w in df.index[df[flag] == 1]:
            for r in rows:
                fig.add_vrect(x0=w - pd.Timedelta(days=3), x1=w + pd.Timedelta(days=4),
                              fillcolor=fill, line_width=0, layer="below", row=r, col=1)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📦 Demand Planning")
    st.caption("Hafen & Home · weekly sales & prices")
    upload = st.file_uploader("Import sales & price data (.xls / .xlsx)", type=["xls", "xlsx"],
                              help="Workbook with a 'Sales in unit' sheet and a 'Prices in Euro' sheet, "
                                   "one column per series (e.g. 'North 1').")
    try:
        sales, prices, future_weeks = load(upload.getvalue() if upload else None)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        st.stop()
    st.caption(("Using uploaded file" if upload else "Using sample data") +
               f" · {len(sales)} weeks, {sales.index.min():%d %b %Y} – {sales.index.max():%d %b %Y}")

    gaps = data_gaps(sales, prices)
    n_gaps = int(gaps[["Missing sales weeks", "Missing price weeks"]].to_numpy().sum())
    with st.expander(f"Data check · {n_gaps} gaps found" if n_gaps else "Data check · complete"):
        st.dataframe(gaps.set_index("Series"), width="stretch")
        st.caption("Missing sales weeks are left out of model fitting and accuracy tests, and interpolated "
                   "for the smoothing models. Missing prices are filled from neighbouring weeks. "
                   "Weeks before a product's launch are ignored.")

    regions = sorted({split_series_name(c)[0] for c in sales.columns})
    products = sorted({split_series_name(c)[1] for c in sales.columns}, key=lambda x: (len(x), x))
    st.divider()
    product = st.selectbox("Product", products, format_func=lambda p: f"Product {p}")
    region = st.radio("Region", regions, horizontal=True)
    series = f"{region} {product}"
    if series not in sales.columns:
        st.error(f"No data for {series}")
        st.stop()

    st.divider()
    st.markdown("**Promotion settings**")
    min_disc = st.slider("Count a week as a promotion if price is at least … below regular", 5, 40, 15, 1,
                         format="%d%%") / 100
    cost_share = st.slider("Unit cost as % of regular price", 20, 90, 55, 5, format="%d%%",
                           help="Needed to judge profit. Replace with the real purchase cost if known.") / 100

df = series_frame(sales[series], prices[series])
model = fit_promo_model(df)
df["baseline"] = baseline_units(df, model)
df["is_promo"] = df["discount"] >= min_disc

st.session_state.setdefault("model_choice", {})
st.session_state.setdefault("promo_plan", {})

st.header(f"Product {product} · {region}")
tab_hist, tab_promo, tab_fc, tab_exp = st.tabs(
    ["📈 Sales history", "🏷️ Promotions", "🔮 Forecast", "⬇️ Export"])

# ================================================================== 1 HISTORY
with tab_hist:
    st.markdown("##### What happened, and when?")
    reg = df[~df["is_promo"]]
    k = st.columns(4)
    k[0].metric("Avg units / week", f"{df['units'].mean():,.0f}")
    k[1].metric("Avg units, regular-price weeks", f"{reg['units'].mean():,.0f}")
    k[2].metric("Avg units, promotion weeks", f"{df.loc[df['is_promo'], 'units'].mean():,.0f}"
                if df["is_promo"].any() else "–")
    k[3].metric("Promotion weeks", f"{df['is_promo'].sum()} of {len(df)}")

    compare = st.toggle("Compare with the other region", value=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Units sold per week", "Price (€)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["units"], name=f"{region} units", mode="lines",
                             line=dict(color=C_ACTUAL, width=2)), 1, 1)
    promo = df[df["is_promo"]]
    fig.add_trace(go.Scatter(x=promo.index, y=promo["units"], name="Promotion week", mode="markers",
                             marker=dict(color=C_ALT, size=9, line=dict(color="white", width=2)),
                             customdata=promo["discount"] * 100,
                             hovertemplate="%{y:,.0f} units · %{customdata:.0f}% off<extra></extra>"), 1, 1)
    if compare:
        other = [r for r in regions if r != region]
        for i, r in enumerate(other):
            col = f"{r} {product}"
            if col in sales.columns:
                fig.add_trace(go.Scatter(x=sales.index, y=sales[col], name=f"{r} units", mode="lines",
                                         line=dict(color=C_3, width=2)), 1, 1)
                fig.add_trace(go.Scatter(x=prices.index, y=prices[col], name=f"{r} price", mode="lines",
                                         line=dict(color=C_3, width=1.5, dash="dot")), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", mode="lines",
                             line=dict(color=C_PRICE, width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["regular_price"], name="Regular price", mode="lines",
                             line=dict(color=C_PRICE, width=1.5, dash="dash")), 2, 1)
    shade_events(fig, df, rows=(1, 2))
    base_layout(fig, 580).update_layout(margin=dict(t=80), legend=dict(y=1.14))
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded weeks: purple = Black Friday / Cyber week · red = 1–21 December · green = Easter. "
               "Regular price = rolling 13-week 75th percentile of the observed price.")

    st.markdown("##### Do we really sell more before the holidays?")
    c = model.coef
    ev = st.columns(3)
    ev[0].metric("Black Friday / Cyber week", f"{(np.exp(c['black_friday']) - 1) * 100:+.0f}%",
                 help="Effect after removing price cuts and trend (regression estimate).")
    ev[1].metric("Pre-Christmas (1–21 Dec)", f"{(np.exp(c['pre_xmas']) - 1) * 100:+.0f}%")
    ev[2].metric("Price cut of 10%", f"{model.uplift_per_10pct:+.0f}% units")
    st.caption(f"Effects relative to a normal regular-price week, estimated jointly from {model.n_obs} weeks "
               f"(R² = {model.r2:.2f}). Only two holiday seasons are in the data, so treat these as indicative.")

    with st.expander("Show data table"):
        st.dataframe(df[["units", "price", "regular_price", "discount", "is_promo"]]
                     .rename(columns={"discount": "discount %"})
                     .style.format({"price": "€{:.2f}", "regular_price": "€{:.2f}", "discount %": "{:.0%}",
                                    "units": "{:,.0f}"}), width="stretch")

# ================================================================== 2 PROMOTIONS
with tab_promo:
    st.markdown("##### Do price cuts pay off? What would we have sold at the regular price?")
    pt = promo_table(df, model, min_disc, cost_share)
    if pt.empty:
        st.info("No promotion weeks at this threshold. Lower the threshold in the sidebar.")
    else:
        k = st.columns(4)
        k[0].metric("Extra units per 10% price cut", f"{model.uplift_per_10pct:+.0f}%")
        k[1].metric("Promotion weeks that paid off", f"{pt['pays_off'].sum()} of {len(pt)}")
        k[2].metric("Extra units from promotions", f"{pt['extra_units'].sum():+,.0f}")
        k[3].metric("Extra gross profit from promotions", f"€{pt['extra_gross_profit'].sum():+,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df["units"], name="Actual units",
                             marker=dict(color=np.where(df["is_promo"], C_ACTUAL, "#b6d5f8"),
                                         cornerradius=4)))
        fig.add_trace(go.Scatter(x=df.index, y=df["baseline"], name="At regular price (estimated)",
                                 mode="lines", line=dict(color=C_ALT, width=2)))
        fig.update_layout(bargap=0.15)
        st.plotly_chart(base_layout(fig, 380), width="stretch")
        st.caption("Dark bars = promotion weeks. The orange line is the expected demand at the regular price "
                   "(trend and holiday effects included, promotion removed).")

        st.markdown("##### Look up one promotion week")
        pick = st.selectbox("Week", pt.index[::-1], format_func=lambda w: f"{w:%d %b %Y} · "
                            f"{pt.loc[w, 'discount']:.0%} off · {pt.loc[w, 'units']:,.0f} units")
        r = pt.loc[pick]
        k = st.columns(4)
        k[0].metric("Sold (actual)", f"{r['units']:,.0f}", help=f"at €{r['price_clean']:.2f}")
        k[1].metric("At regular price (est.)", f"{r['baseline']:,.0f}", help=f"at €{r['regular_price']:.2f}")
        be = r["breakeven_uplift_%"]
        k[2].metric("Uplift needed to break even", f"{be:.0f}%" if be < 1000 else "not reachable",
                    help="Extra units needed so the discount earns the same gross profit as the regular price.")
        k[3].metric("Actual uplift", f"{r['actual_uplift_%']:.0f}%")
        verdict = "paid off" if r["pays_off"] else "did not pay off"
        (st.success if r["pays_off"] else st.warning)(
            f"The {r['discount']:.0%} discount {verdict}: gross profit changed by "
            f"€{r['extra_gross_profit']:+,.0f} versus selling at the regular price.")

        with st.expander("All promotion weeks"):
            show = pt[["price_clean", "regular_price", "discount", "units", "baseline", "extra_units",
                       "actual_uplift_%", "breakeven_uplift_%", "extra_revenue", "extra_gross_profit", "pays_off"]]
            show.columns = ["Price", "Regular price", "Discount", "Units", "Units at regular price", "Extra units",
                            "Uplift %", "Break-even uplift %", "Extra revenue €", "Extra gross profit €", "Paid off"]
            st.dataframe(show.sort_values("Extra gross profit €", ascending=False).style.format(
                {"Price": "€{:.2f}", "Regular price": "€{:.2f}", "Discount": "{:.0%}", "Units": "{:,.0f}",
                 "Units at regular price": "{:,.0f}", "Extra units": "{:+,.0f}", "Uplift %": "{:.0f}",
                 "Break-even uplift %": "{:.0f}", "Extra revenue €": "{:+,.0f}",
                 "Extra gross profit €": "{:+,.0f}"}), width="stretch")
        st.caption(f"Break-even uplift = discount ÷ (margin − discount), with margin = {1 - cost_share:.0%}. "
                   "A promotion pays off when the actual uplift exceeds it.")

# ================================================================== 3 FORECAST
with tab_fc:
    st.markdown("##### Which forecast do we take to the buying meeting?")
    bt = series_backtest(df.drop(columns=["baseline", "is_promo"]))
    best = bt.loc[bt["MAPE %"].idxmin(), "Model"]

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"**Accuracy test** – each model forecasts the last {HORIZON} known weeks "
                    "without seeing them (lower MAPE = better).")
        st.dataframe(bt[["Model", "MAPE %"]].sort_values("MAPE %").style.format({"MAPE %": "{:.1f}"}),
                     hide_index=True, width="stretch")
        default = st.session_state["model_choice"].get(series, best)
        choice = st.radio("Forecast to use", MODEL_NAMES, index=MODEL_NAMES.index(default),
                          format_func=lambda m: f"{m}  ★" if m == best else m)
        st.session_state["model_choice"][series] = choice
    with right:
        test = df.iloc[-HORIZON:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-40:], y=df["units"].iloc[-40:], name="Actual", mode="lines",
                                 line=dict(color=C_ACTUAL, width=2)))
        for _, row in bt.iterrows():
            fig.add_trace(go.Scatter(x=test.index, y=row["forecast"], name=row["Model"], mode="lines",
                                     line=dict(color=MODEL_COLORS[row["Model"]],
                                               width=3 if row["Model"] == choice else 1.5)))
        st.plotly_chart(base_layout(fig, 340), width="stretch")

    st.markdown(f"##### Planned promotions in the next {HORIZON} weeks")
    plan_default = pd.DataFrame({"Week": [w.date() for w in future_weeks], "Planned discount %": 0.0})
    plan = st.session_state["promo_plan"].get(series, plan_default)
    plan = st.data_editor(plan, hide_index=True, key=f"plan_{series}", width="stretch",
                          disabled=["Week"],
                          column_config={"Planned discount %": st.column_config.NumberColumn(
                              min_value=0, max_value=70, step=5, format="%d%%")})
    st.session_state["promo_plan"][series] = plan
    disc = plan["Planned discount %"].values / 100
    if choice != "Price & promotion regression" and disc.any():
        st.info("Planned discounts only change the 'Price & promotion regression' forecast.")

    fc = forecast(choice, df, future_weeks, disc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["units"], name="Actual", mode="lines",
                             line=dict(color=C_ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=future_weeks, y=fc, name=f"Forecast · {choice}", mode="lines+markers",
                             line=dict(color=C_ALT, width=2), marker=dict(size=8)))
    st.plotly_chart(base_layout(fig, 360), width="stretch")
    k = st.columns(3)
    k[0].metric(f"Forecast, next {HORIZON} weeks", f"{fc.sum():,.0f} units")
    k[1].metric(f"Same {HORIZON} weeks last year", f"{df['units'].iloc[-52:-52 + HORIZON].sum():,.0f} units"
                if len(df) >= 52 else "–")
    k[2].metric(f"Last {HORIZON} weeks", f"{df['units'].iloc[-HORIZON:].sum():,.0f} units")

# ================================================================== 4 EXPORT
with tab_exp:
    st.markdown(f"##### Weekly forecast for all products and regions ({HORIZON} weeks)")
    st.caption("Uses the model and planned promotions you chose per series in the Forecast tab; "
               "series you have not opened use the most accurate model (★) and no promotions.")
    out, summary = {}, []
    for col in sales.columns:
        d = series_frame(sales[col], prices[col])
        b = series_backtest(d)
        m = st.session_state["model_choice"].get(col, b.loc[b["MAPE %"].idxmin(), "Model"])
        p = st.session_state["promo_plan"].get(col)
        dsc = p["Planned discount %"].values / 100 if p is not None else None
        out[col] = np.round(forecast(m, d, future_weeks, dsc)).astype(int)
        summary.append({"Series": col, "Model": m,
                        "Test MAPE %": float(b.loc[b["Model"] == m, "MAPE %"].iloc[0]),
                        f"Total {HORIZON} weeks": int(out[col].sum())})
    fc_all = pd.DataFrame(out, index=pd.Index([w.date() for w in future_weeks], name="Week"))
    st.dataframe(pd.DataFrame(summary).style.format({"Test MAPE %": "{:.1f}", f"Total {HORIZON} weeks": "{:,}"}),
                 hide_index=True, width="stretch")
    st.dataframe(fc_all, width="stretch")
    st.download_button("Download forecast (CSV)", fc_all.to_csv().encode("utf-8"),
                       file_name="hafen_home_forecast_13w.csv", mime="text/csv", type="primary")
