"""
Parking Demand Analytics — Decision Support Dashboard
=====================================================
Final Year Project (CSDA)

Presents the analytical outputs of the project:
  • Temporal demand patterns          (Objective 1)
  • Daily behavioural profiles        (Objective 2)
  • Short-term demand forecasts       (Objective 3)
  • Interactive decision support      (Objective 4)

Run with:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "dashboard"
MODEL_DIR = PROJECT_ROOT / "models"

SITE_COLOURS = {"Site A": "#2E75B6", "Site B": "#C55A11", "Site C": "#548235"}

CLUSTER_NAMES = {
    0: "Low-Volume / Near-Inactive Day",
    1: "Regular High-Activity Morning-Demand Day",
}
CLUSTER_COLOURS = {0: "#C55A11", 1: "#2E75B6"}

st.set_page_config(
    page_title="Parking Demand Analytics",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1, h2, h3 {color: #1F3864;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(name, parse_dates=None):
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import joblib
        model = joblib.load(MODEL_DIR / "xgb_demand_model.pkl")
        features = joblib.load(MODEL_DIR / "feature_cols.pkl")
        return model, features
    except Exception:
        return None, None


clusters = load_csv("site_day_clusters.csv", parse_dates=["entry_date"])
daily = load_csv("daily_demand_features.csv", parse_dates=["date"])
test_pred = load_csv("test_predictions.csv", parse_dates=["date"])
metrics = load_csv("model_metrics.csv")
pat_hour = load_csv("pattern_hourly.csv")
pat_week = load_csv("pattern_weekday.csv")
pat_month = load_csv("pattern_monthly.csv")

model, feature_cols = load_model()

if clusters is None or daily is None:
    st.error(
        f"Required data files were not found in `{DATA_DIR}`.\n\n"
        "Run the export cells at the end of notebooks 03 and 04 first."
    )
    st.stop()

ALL_SITES = sorted(clusters["site"].unique())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def forecast_ahead(model, feature_cols, history, site_name, n_days, holiday_dates):
    """Recursively forecast n_days beyond the end of `history` for one site."""
    hist = history[["date", "daily_transactions"]].copy().sort_values("date")
    last_date = hist["date"].max()
    out = []

    for step in range(1, n_days + 1):
        target = last_date + pd.Timedelta(days=step)
        series = hist.set_index("date")["daily_transactions"]

        row = {
            "day_of_week": target.dayofweek,
            "month": target.month,
            "day_of_year": target.dayofyear,
            "is_weekend": int(target.dayofweek >= 5),
            "is_holiday": int(target.date() in holiday_dates),
            "lag_1": series.get(target - pd.Timedelta(days=1), np.nan),
            "lag_7": series.get(target - pd.Timedelta(days=7), np.nan),
            "lag_14": series.get(target - pd.Timedelta(days=14), np.nan),
            "roll_mean_7": series.loc[
                target - pd.Timedelta(days=7): target - pd.Timedelta(days=1)
            ].mean(),
        }
        for col in feature_cols:
            if col.startswith("site_"):
                row[col] = int(col == f"site_{site_name}")

        frame = pd.DataFrame([row]).reindex(columns=feature_cols).fillna(0)
        pred = max(0.0, float(model.predict(frame)[0]))

        out.append({"date": target, "forecast": pred})
        hist = pd.concat(
            [hist, pd.DataFrame([{"date": target, "daily_transactions": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Parking Demand Analytics")
st.sidebar.caption("Decision-support dashboard")

page = st.sidebar.radio(
    "View",
    [
        "Overview",
        "Temporal Patterns",
        "Behavioural Profiles",
        "Demand Forecast",
        "Operational Alerts",
    ],
)

st.sidebar.markdown("---")
selected_sites = st.sidebar.multiselect(
    "Parking sites", ALL_SITES, default=ALL_SITES
)
if not selected_sites:
    st.warning("Select at least one site from the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: 87,440 parking transactions across three commercial and office "
    "facilities, January – December 2025. Site identities are anonymised."
)

view = clusters[clusters["site"].isin(selected_sites)]


# ==========================================================================
# PAGE 1 — OVERVIEW
# ==========================================================================
if page == "Overview":
    st.title("Parking Demand Analytics")
    st.caption(
        "Operational overview of daily parking activity across the monitored facilities"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{int(view['total_transactions'].sum()):,}")
    c2.metric("Site-days observed", f"{len(view):,}")
    c3.metric("Mean daily demand", f"{view['total_transactions'].mean():.0f}")
    c4.metric("Total revenue", f"RM {view['total_revenue'].sum():,.0f}")

    st.markdown("---")

    section_header("Daily demand over time")
    fig = px.line(
        view.sort_values("entry_date"),
        x="entry_date",
        y="total_transactions",
        color="site",
        color_discrete_map=SITE_COLOURS,
        labels={"entry_date": "Date", "total_transactions": "Transactions",
                "site": "Site"},
    )
    fig.update_layout(height=380, hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        section_header("Demand distribution by site")
        fig = px.box(
            view, x="site", y="total_transactions", color="site",
            color_discrete_map=SITE_COLOURS,
            labels={"site": "", "total_transactions": "Daily transactions"},
        )
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_header("Site comparison")
        summary = (
            view.groupby("site")
            .agg(
                Days=("entry_date", "count"),
                Mean_demand=("total_transactions", "mean"),
                Median_duration=("median_duration", "median"),
                Paid_ratio=("paid_ratio", "mean"),
            )
            .round(1)
            .reset_index()
        )
        summary.columns = ["Site", "Days", "Mean demand",
                           "Median duration (min)", "Paid ratio"]
        summary["Paid ratio"] = (summary["Paid ratio"] * 100).round(1).astype(str) + "%"
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "The three facilities differ substantially in both volume and "
            "typical length of stay, which motivated comparing them on a "
            "relative rather than absolute basis during analysis."
        )


# ==========================================================================
# PAGE 2 — TEMPORAL PATTERNS  (Objective 1)
# ==========================================================================
elif page == "Temporal Patterns":
    st.title("Temporal Demand Patterns")
    st.caption("When parking demand occurs across the day, week, and year")

    if pat_hour is None:
        st.info("Temporal pattern files not found. Run the export cell in notebook 04.")
        st.stop()

    ph = pat_hour[pat_hour["site"].isin(selected_sites)]
    pw = pat_week[pat_week["site"].isin(selected_sites)]
    pm = pat_month[pat_month["site"].isin(selected_sites)]

    section_header(
        "Entries by hour of day",
        "Aggregated across the full observation period",
    )
    fig = px.bar(
        ph, x="entry_hour", y="transactions", color="site", barmode="group",
        color_discrete_map=SITE_COLOURS,
        labels={"entry_hour": "Hour of entry", "transactions": "Transactions",
                "site": "Site"},
    )
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Demand rises sharply through the early morning and peaks around "
        "08:00, consistent with commuter and business arrivals at the start "
        "of the working day. Morning readiness is therefore the period of "
        "greatest operational pressure."
    )

    left, right = st.columns(2)

    with left:
        section_header("Entries by day of week")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
        fig = px.bar(
            pw, x="entry_day_name", y="transactions", color="site",
            barmode="group", color_discrete_map=SITE_COLOURS,
            category_orders={"entry_day_name": order},
            labels={"entry_day_name": "", "transactions": "Transactions",
                    "site": "Site"},
        )
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Weekday activity is consistently higher, with a marked fall at "
            "the weekend."
        )

    with right:
        section_header("Entries by month")
        fig = px.line(
            pm, x="entry_month", y="transactions", color="site", markers=True,
            color_discrete_map=SITE_COLOURS,
            labels={"entry_month": "Month", "transactions": "Transactions",
                    "site": "Site"},
        )
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Monthly volumes remain broadly stable, indicating demand driven "
            "by routine weekly activity rather than seasonal effects."
        )

    st.markdown("---")
    section_header("Time-of-day composition by site")
    comp = (
        view.groupby("site")[
            ["early_morning_ratio", "morning_ratio",
             "afternoon_ratio", "evening_ratio"]
        ]
        .mean()
        .reset_index()
        .melt(id_vars="site", var_name="Period", value_name="Share")
    )
    comp["Period"] = comp["Period"].map({
        "early_morning_ratio": "Early morning (00–06)",
        "morning_ratio": "Morning (06–12)",
        "afternoon_ratio": "Afternoon (12–18)",
        "evening_ratio": "Evening (18–24)",
    })
    fig = px.bar(
        comp, x="site", y="Share", color="Period", barmode="stack",
        labels={"site": "", "Share": "Share of daily entries"},
    )
    fig.update_layout(height=330, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE 3 — BEHAVIOURAL PROFILES  (Objective 2)
# ==========================================================================
elif page == "Behavioural Profiles":
    st.title("Daily Behavioural Profiles")
    st.caption(
        "Site-days grouped by operating behaviour using K-Means clustering"
    )

    view = view.copy()
    view["profile"] = view["kmeans_cluster"].map(CLUSTER_NAMES)

    counts = view["kmeans_cluster"].value_counts().sort_index()
    cols = st.columns(len(counts))
    for col, (cid, n) in zip(cols, counts.items()):
        col.metric(
            CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
            f"{n:,} days",
            f"{n / len(view) * 100:.1f}% of observed days",
        )

    st.markdown("---")
    section_header("Profile characteristics")

    prof = (
        view.groupby("profile")
        .agg(
            Days=("entry_date", "count"),
            Demand=("total_transactions", "mean"),
            Median_duration=("median_duration", "median"),
            Revenue=("total_revenue", "mean"),
            Paid=("paid_ratio", "mean"),
            Weekend=("is_weekend", "mean"),
            Morning=("morning_ratio", "mean"),
        )
        .round(2)
        .reset_index()
    )
    for c in ["Paid", "Weekend", "Morning"]:
        prof[c] = (prof[c] * 100).round(1).astype(str) + "%"
    prof.columns = ["Profile", "Days", "Mean transactions/day",
                    "Median duration (min)", "Mean revenue (RM)",
                    "Paid ratio", "Weekend share", "Morning entries"]
    st.dataframe(prof, use_container_width=True, hide_index=True)

    st.info(
        "**Low-Volume / Near-Inactive Days** are quiet, predominantly weekend "
        "days with very low transaction counts. Their high median duration "
        "reflects a small number of abandoned or unclosed sessions dominating "
        "days with few transactions, rather than genuine long-stay behaviour — "
        "so these days also warrant a light data-quality review.\n\n"
        "**Regular High-Activity Morning-Demand Days** represent normal "
        "weekday-oriented operation, with pronounced morning arrivals and "
        "higher revenue. These are the days that drive staffing and gate-readiness needs."
    )

    left, right = st.columns([3, 2])

    with left:
        section_header("Cluster separation (PCA projection)")
        if {"PCA1", "PCA2"}.issubset(view.columns):
            fig = px.scatter(
                view, x="PCA1", y="PCA2", color="profile",
                color_discrete_map={CLUSTER_NAMES[k]: v
                                    for k, v in CLUSTER_COLOURS.items()},
                hover_data=["site", "entry_date", "total_transactions"],
                labels={"profile": "Profile"},
                opacity=0.7,
            )
            fig.update_layout(height=420,
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Two principal components representing 58.8% of the "
                "transformed feature variance. Used for visualisation only."
            )
        else:
            st.info("PCA columns not present in the exported data.")

    with right:
        section_header("Profile mix by site")
        mix = (
            view.groupby(["site", "profile"]).size()
            .reset_index(name="days")
        )
        fig = px.bar(
            mix, x="site", y="days", color="profile", barmode="stack",
            color_discrete_map={CLUSTER_NAMES[k]: v
                                for k, v in CLUSTER_COLOURS.items()},
            labels={"site": "", "days": "Site-days"},
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Both profiles occur at all three facilities, confirming they "
            "describe behaviour rather than simply re-labelling sites."
        )


# ==========================================================================
# PAGE 4 — DEMAND FORECAST  (Objective 3)
# ==========================================================================
elif page == "Demand Forecast":
    st.title("Short-Term Demand Forecast")
    st.caption("Predicted daily transaction demand using the XGBoost model")

    if metrics is not None:
        cols = st.columns(len(metrics))
        for col, (_, row) in zip(cols, metrics.iterrows()):
            col.metric(row["model"], f"MAE {row['MAE']:.2f}",
                       f"RMSE {row['RMSE']:.2f}")
        st.caption(
            "Accuracy measured on the unseen November–December hold-out "
            "period, in transactions per day."
        )

    st.markdown("---")
    tab1, tab2 = st.tabs(["Model performance", "Forward forecast"])

    # ---------------- Tab 1: performance on the hold-out period
    with tab1:
        if test_pred is None:
            st.info("Test prediction file not found.")
        else:
            tp = test_pred[test_pred["site_name"].isin(selected_sites)]

            section_header(
                "Actual versus predicted demand",
                "Hold-out period — data the model did not observe during training",
            )
            for site in sorted(tp["site_name"].unique()):
                sub = tp[tp["site_name"] == site].sort_values("date")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["daily_transactions"],
                    name="Actual", line=dict(color="#222", width=2.5)))
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["pred_xgb"], name="XGBoost",
                    line=dict(color="#2E75B6", width=1.8, dash="dash")))
                if "pred_lstm" in sub.columns:
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=sub["pred_lstm"], name="LSTM",
                        line=dict(color="#C55A11", width=1.6, dash="dot")))
                fig.update_layout(
                    title=f"{site}", height=290, hovermode="x unified",
                    margin=dict(t=40, b=20),
                    legend=dict(orientation="h", y=1.15),
                    yaxis_title="Transactions",
                )
                st.plotly_chart(fig, use_container_width=True)

            section_header("Accuracy by site")
            rows = []
            for site in sorted(tp["site_name"].unique()):
                sub = tp[tp["site_name"] == site]
                mean_d = sub["daily_transactions"].mean()
                mae = (sub["daily_transactions"] - sub["pred_xgb"]).abs().mean()
                rows.append({
                    "Site": site,
                    "Mean demand": round(mean_d, 1),
                    "MAE": round(mae, 2),
                    "MAE as % of mean": f"{mae / mean_d * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            st.caption(
                "Relative error is lowest at the facility with the most "
                "regular operating pattern. Accuracy is inherently limited at "
                "sites subject to closures that cannot be anticipated from "
                "the calendar."
            )

    # ---------------- Tab 2: forward forecast
    with tab2:
        if model is None:
            st.info(
                "Trained model not found. Run the export cell in notebook 04 "
                "to save `xgb_demand_model.pkl`."
            )
        else:
            c1, c2 = st.columns([1, 1])
            site_choice = c1.selectbox("Site", ALL_SITES)
            horizon = c2.slider("Days ahead", 7, 30, 14)

            try:
                import holidays as hol
                last_year = int(daily["date"].max().year)
                hset = set(hol.Malaysia(years=[last_year, last_year + 1]).keys())
            except Exception:
                hset = set()
                st.caption(
                    "Holiday calendar unavailable — forecasts will not account "
                    "for public holidays."
                )

            hist = daily[daily["site_name"] == site_choice].sort_values("date")
            fc = forecast_ahead(model, feature_cols, hist, site_choice,
                                horizon, hset)

            recent = hist.tail(45)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent["date"], y=recent["daily_transactions"],
                name="Observed", line=dict(color="#222", width=2.2)))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc["forecast"], name="Forecast",
                line=dict(color="#C55A11", width=2.2, dash="dash")))
            fig.add_vline(x=hist["date"].max(), line_dash="dot",
                          line_color="grey")
            fig.update_layout(
                title=f"{site_choice} — next {horizon} days",
                height=400, hovermode="x unified",
                yaxis_title="Transactions",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("Mean forecast demand", f"{fc['forecast'].mean():.0f}")
            k2.metric("Peak day", f"{fc['forecast'].max():.0f}",
                      fc.loc[fc['forecast'].idxmax(), 'date'].strftime("%d %b"))
            k3.metric("Quietest day", f"{fc['forecast'].min():.0f}",
                      fc.loc[fc['forecast'].idxmin(), 'date'].strftime("%d %b"))

            out = fc.copy()
            out["Day"] = out["date"].dt.day_name()
            out["forecast"] = out["forecast"].round(0).astype(int)
            out.columns = ["Date", "Forecast demand", "Day"]
            st.dataframe(out[["Date", "Day", "Forecast demand"]],
                         use_container_width=True, hide_index=True)

            st.warning(
                "Forecasts are generated recursively: each predicted day "
                "becomes an input to the next. Accuracy therefore declines "
                "with the length of the horizon, and the model cannot "
                "anticipate closures that are not part of the regular "
                "calendar pattern."
            )


# ==========================================================================
# PAGE 5 — OPERATIONAL ALERTS
# ==========================================================================
elif page == "Operational Alerts":
    st.title("Operational Review Alerts")
    st.caption(
        "Site-days flagged for review by the anomaly-detection layer"
    )

    view = view.copy()
    flagged = view[
        (view.get("extended_duration_flag", 0) == 1)
        | (view.get("dbscan_noise_flag", 0) == 1)
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Site-days reviewed", f"{len(view):,}")
    c2.metric("Flagged for review", f"{len(flagged):,}",
              f"{len(flagged) / max(len(view), 1) * 100:.1f}% of days")
    c3.metric("Extended-duration days",
              f"{int(view.get('extended_duration_flag', pd.Series(dtype=int)).sum()):,}")

    st.info(
        "Flags identify days whose operating behaviour falls outside the "
        "normal pattern — for example, very long recorded stays on days with "
        "few transactions, which typically indicate unclosed or abandoned "
        "sessions. These are review prompts for operational and data-quality "
        "checking, not confirmed errors, and records are never removed "
        "automatically."
    )

    st.markdown("---")
    section_header("Flagged site-days")

    if flagged.empty:
        st.success("No site-days flagged for the selected facilities.")
    else:
        tbl = flagged[[
            "site", "entry_date", "total_transactions", "median_duration",
            "total_revenue", "is_weekend",
        ]].copy()
        tbl["entry_date"] = tbl["entry_date"].dt.date
        tbl["is_weekend"] = tbl["is_weekend"].map({1: "Weekend", 0: "Weekday"})
        tbl.columns = ["Site", "Date", "Transactions", "Median duration (min)",
                       "Revenue (RM)", "Day type"]
        tbl = tbl.sort_values("Median duration (min)", ascending=False)
        st.dataframe(tbl.round(1), use_container_width=True, hide_index=True,
                     height=420)

        st.download_button(
            "Download review list (CSV)",
            tbl.to_csv(index=False).encode("utf-8"),
            file_name="review_alerts.csv",
            mime="text/csv",
        )

        section_header("Where flagged days occur")
        by_site = flagged.groupby("site").size().reset_index(name="Flagged days")
        fig = px.bar(
            by_site, x="site", y="Flagged days", color="site",
            color_discrete_map=SITE_COLOURS, labels={"site": ""},
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)