"""
app_task2.py - Taxi Expense Dashboard (Task 2)
Run locally:  streamlit run app_task2.py
"""

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from src.task2.data import (
    AnomalyDetector,
    RideDensityMap,
    RideSharingAnalyzer,
    TaxiAggregator,
    TaxiDataFilter,
    TaxiDataLoader,
)

st.set_page_config(
    page_title="Taxi Expense Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data() -> pd.DataFrame:
    """Load and cache the full taxi expense dataset.

    Returns:
        Preprocessed DataFrame as returned by :meth:`TaxiDataLoader.load`.
    """
    return TaxiDataLoader.load()


@st.cache_data
def get_ridesharing(data_hash: int) -> dict:  # data_hash unused; forces re-cache only when df changes
    """Compute and cache ride-sharing opportunities for the full dataset.

    Args:
        data_hash: Hash used as a cache key; pass ``len(df)`` to invalidate
            whenever the underlying data changes.

    Returns:
        Dictionary as returned by :meth:`RideSharingAnalyzer.compute_opportunities`.
    """
    return RideSharingAnalyzer(get_data()).compute_opportunities()


@st.cache_data
def get_anomaly(data_hash: int) -> dict:
    """Detect and cache anomalous division codes for the full dataset.

    Args:
        data_hash: Hash used as a cache key; pass ``len(df)`` to invalidate
            whenever the underlying data changes.

    Returns:
        Dictionary as returned by :meth:`AnomalyDetector.detect`.
    """
    return AnomalyDetector(get_data()).detect()


df_all = get_data()

# Pre-compute unfiltered results for tabs 3 & 4
ridesharing = get_ridesharing(len(df_all))
anomaly = get_anomaly(len(df_all))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")
st.sidebar.caption("Filters apply to Cost Trends and Ride Patterns tabs only.")

all_month_labels = (
    df_all[["month_ts", "month_label"]]
    .drop_duplicates()
    .sort_values("month_ts")["month_label"]
    .tolist()
)
selected_months = st.sidebar.multiselect(
    "Months", options=all_month_labels, default=all_month_labels
)

top10_divs = (
    df_all.groupby("division_code")["total_cost"]
    .sum()
    .nlargest(10)
    .index.tolist()
)
division_options = top10_divs + ["All Others"]
selected_divisions = st.sidebar.multiselect(
    "Divisions (top 10 shown)", options=division_options, default=division_options
)

include_weekends = st.sidebar.checkbox("Include Weekend Rides", value=True)

# Apply filters
df = TaxiDataFilter(df_all).apply(selected_months, selected_divisions, include_weekends)

# KPI strip
st.sidebar.divider()
st.sidebar.metric("Total Rides", f"{len(df):,}")
st.sidebar.metric("Total Spend", f"${df['total_cost'].sum():,.2f}")
avg = df["total_cost"].mean() if len(df) else 0
st.sidebar.metric("Avg Cost / Trip", f"${avg:.2f}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

agg = TaxiAggregator(df)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cost Trends",
    "Ride Patterns",
    "Ride-Sharing Opportunity",
    "Data Quality Alert",
    "Ride Density Map",
])

# ── Tab 1: Cost Trends ───────────────────────────────────────────────────────
with tab1:
    st.header("Are taxi costs going up over time?")

    monthly = agg.monthly()

    if monthly.empty:
        st.warning("No data for the selected filters.")
    else:
        # Dual-axis: bars = num_rides, line = avg_cost_per_trip
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly["month_label"],
            y=monthly["num_rides"],
            name="Number of Rides",
            marker_color="#4A90D9",
            yaxis="y1",
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly["month_label"],
            y=monthly["avg_cost_per_trip"],
            name="Avg Cost / Trip ($)",
            mode="lines+markers",
            marker=dict(size=8, color="#E8714A"),
            line=dict(width=2.5, color="#E8714A"),
            yaxis="y2",
        ))
        fig_monthly.update_layout(
            title="Monthly Rides and Average Cost per Trip",
            yaxis=dict(title="Number of Rides"),
            yaxis2=dict(title="Avg Cost per Trip ($)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.info(
            "Average cost per trip rose from **\\$15.02 in January** to **\\$16.84 in September** - "
            "a 12% increase over 9 months, even as total monthly spend fluctuated."
        )

        st.divider()

        # Hourly bar chart coloured by avg cost
        hourly = agg.hourly()
        fig_hourly = px.bar(
            hourly,
            x="hour",
            y="num_rides",
            color="avg_cost",
            color_continuous_scale="RdYlGn_r",
            labels={"hour": "Hour of Day", "num_rides": "Number of Rides", "avg_cost": "Avg Cost ($)"},
            title="Trips by Hour of Day - Colour Shows Average Cost per Trip",
        )
        fig_hourly.update_layout(
            xaxis=dict(dtick=1, tickmode="linear"),
            coloraxis_colorbar=dict(title="Avg Cost ($)"),
            height=380,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.warning(
            "Late-night trips (1am–3am) cost **\\$21–22 per trip** - more than twice the off-peak "
            "afternoon rate of ~$10/trip. Restricting or pre-approving late-night trips is the "
            "single largest lever for reducing per-trip costs."
        )

# ── Tab 2: Ride Patterns ─────────────────────────────────────────────────────
with tab2:
    st.header("When and where are trips happening?")

    weekday = agg.weekday()

    if weekday.empty:
        st.warning("No data for the selected filters.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            fig_wd_rides = px.bar(
                weekday,
                x="weekday",
                y="num_rides",
                title="Number of Trips by Day",
                labels={"weekday": "", "num_rides": "Number of Rides"},
                color_discrete_sequence=["#4A90D9"],
                category_orders={"weekday": [w for w in
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    if w in weekday["weekday"].values]},
            )
            fig_wd_rides.update_layout(height=350)
            st.plotly_chart(fig_wd_rides, use_container_width=True)

        with col_right:
            fig_wd_cost = px.bar(
                weekday,
                x="weekday",
                y="total_cost",
                title="Total Spend by Day ($)",
                labels={"weekday": "", "total_cost": "Total Spend ($)"},
                color_discrete_sequence=["#E8714A"],
                category_orders={"weekday": [w for w in
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    if w in weekday["weekday"].values]},
            )
            fig_wd_cost.update_layout(height=350)
            st.plotly_chart(fig_wd_cost, use_container_width=True)

        st.info(
            "Wednesday accounts for the most trips (296) and the highest total spend ($5,033) - "
            "nearly double any other weekday."
        )

        st.divider()

        # Division donut + heatmap
        div_totals = agg.division_totals(top_n=10)

        fig_pie = px.pie(
            div_totals,
            values="total_cost",
            names="division_code",
            title="Total Spend by Division (Top 10 + Other)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_traces(textposition="inside", textinfo="label+percent")
        fig_pie.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.warning(
            "Divisions **Z002, Z000, and Z007** account for **~61% of all taxi spend**. "
            "Any cost-control policy should prioritise these three divisions first."
        )

        st.divider()

        heatmap_pivot = agg.division_monthly_heatmap(top_n=15)

        if not heatmap_pivot.empty:
            fig_heat = px.imshow(
                heatmap_pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Monthly Spend Heatmap - Top 15 Divisions ($)",
                labels=dict(x="Month", y="Division", color="Spend ($)"),
                text_auto=".0f",
                zmin=0,
            )
            fig_heat.update_layout(height=520)
            st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 3: Ride-Sharing Opportunity ─────────────────────────────────────────
with tab3:
    st.header("How much could we save with ride-sharing?")
    st.caption("This analysis uses the full dataset (filters above do not apply).")

    ts = ridesharing["threshold_summary"]
    total_grp = ridesharing["total_concurrent_groups"]

    row2km = ts[ts["threshold_km"] == 2.0].iloc[0] if not ts.empty else None
    row5km = ts[ts["threshold_km"] == 5.0].iloc[0] if not ts.empty else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Concurrent Trip Groups", f"{total_grp}")
    if row2km is not None:
        m2.metric(
            "Shareable at 2 km threshold",
            f"{int(row2km['eligible_groups'])} groups",
            delta=f"{row2km['pct_groups']:.1f}% of concurrent trips",
        )
    if row5km is not None:
        m3.metric(
            "Shareable at 5 km threshold",
            f"{int(row5km['eligible_groups'])} groups",
            delta=f"{row5km['pct_groups']:.1f}% of concurrent trips",
        )

    fig_rs = px.bar(
        ts,
        x="threshold_km",
        y="pct_groups",
        text=ts["pct_groups"].apply(lambda v: f"{v:.1f}%"),
        title="% of Concurrent Trip Groups Eligible for Ride-Sharing by Distance Threshold",
        labels={"threshold_km": "Max Distance Between Destinations (km)", "pct_groups": "% of Groups"},
        color_discrete_sequence=["#2196F3"],
    )
    fig_rs.update_traces(textposition="outside")
    fig_rs.update_layout(
        yaxis=dict(range=[0, 100]),
        xaxis=dict(tickvals=[1, 2, 5], ticktext=["1 km", "2 km", "5 km"]),
        height=380,
    )
    st.plotly_chart(fig_rs, use_container_width=True)

    with st.expander("How is this calculated?"):
        st.markdown(
            """
            A **concurrent group** is two or more trips that depart from the **same pickup location**
            within the **same 15-minute window**.

            For each such group, the straight-line distance between every pair of destinations is
            measured using the **Haversine formula** (accounting for Earth's curvature).

            A group is flagged as **shareable** if any pair of destinations falls within the threshold
            distance. The 15-minute window and 2 km distance are conservative parameters - actual road
            distances are longer than straight-line distances, so the true overlap may be lower.
            """
        )

    st.success(
        "**Recommendation:** Implement a ride-sharing policy for officers departing from the same "
        "location within a 15-minute window. At a **2 km threshold**, 1 in 3 concurrent groups "
        "could be consolidated into shared trips, directly reducing taxi expenditure without "
        "impacting operational needs."
    )

# ── Tab 4: Data Quality Alert ────────────────────────────────────────────────
with tab4:
    st.header("Unusual pattern detected: July–August 2015")
    st.caption("This analysis uses the full dataset (filters above do not apply).")

    n_anomalous = anomaly["count"]
    monthly_uniq = anomaly["monthly_unique_div_counts"]

    st.error(
        f"**{n_anomalous} division codes** appeared exclusively in July and August 2015, "
        "then never again. This may indicate a data entry error, a temporary re-organisation, "
        "or misattributed trips."
    )

    fig_uniq = px.bar(
        monthly_uniq,
        x="month_label",
        y="unique_divisions",
        title="Number of Unique Division Codes per Month",
        labels={"month_label": "Month", "unique_divisions": "Unique Division Codes"},
        color_discrete_sequence=["#4A90D9"],
    )
    # Highlight Jul and Aug bars in red
    bar_colors = [
        "#E8714A" if m in ("Jul 2015", "Aug 2015") else "#4A90D9"
        for m in monthly_uniq["month_label"].astype(str)
    ]
    fig_uniq.update_traces(marker_color=bar_colors)
    fig_uniq.update_layout(height=380)
    st.plotly_chart(fig_uniq, use_container_width=True)

    st.warning(
        "**Recommended action:** Ask the Finance or HR team to verify division code assignments "
        "for July–August 2015 before using division-level figures from those months in any "
        "official report."
    )

    # Sample of anomalous codes
    if anomaly["anomalous_codes"]:
        sample_codes = anomaly["anomalous_codes"][:20]
        st.markdown(f"**Sample of new division codes (showing up to 20 of {n_anomalous}):**")
        anomalous_detail = (
            df_all[df_all["division_code"].isin(sample_codes)]
            .groupby("division_code")
            .agg(
                rides=("total_cost", "count"),
                total_spend=("total_cost", "sum"),
                months_active=("month_label", lambda x: ", ".join(sorted(x.unique()))),
            )
            .reset_index()
            .rename(columns={
                "division_code": "Division Code",
                "rides": "Rides",
                "total_spend": "Total Spend ($)",
                "months_active": "Months Active",
            })
        )
        st.dataframe(anomalous_detail, use_container_width=True, height=320)

# ── Tab 5: Ride Density Map ──────────────────────────────────────────────────
with tab5:
    st.header("Where are trips concentrated across Singapore?")
    st.caption("Uses the full dataset. Sidebar division/weekend filters do not apply here.")

    density_mapper = RideDensityMap(df_all)

    # ── In-tab controls ──────────────────────────────────────────────────────
    ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 1, 2])

    with ctrl_left:
        point_type = st.radio(
            "Show coordinates for",
            options=["pickup", "destination"],
            format_func=lambda x: x.capitalize(),
            horizontal=True,
        )

    with ctrl_mid:
        map_months = st.multiselect(
            "Months",
            options=all_month_labels,
            default=all_month_labels,
            key="map_months",
        )

    with ctrl_right:
        hour_range = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=(0, 23),
            format="%d:00",
        )

    # ── Hourly sparkline ─────────────────────────────────────────────────────
    hourly_counts = density_mapper.hourly_counts(
        point_type=point_type,
        selected_months=map_months or None,
    )
    fig_spark = px.bar(
        hourly_counts,
        x="hour",
        y="num_rides",
        labels={"hour": "Hour of Day", "num_rides": "Trips"},
        title="Trip Volume by Hour (selected months) - drag the slider above to filter the map",
        color_discrete_sequence=["#4A90D9"],
    )
    # Highlight the selected hour window
    fig_spark.add_vrect(
        x0=hour_range[0] - 0.5,
        x1=hour_range[1] + 0.5,
        fillcolor="#E8714A",
        opacity=0.15,
        line_width=0,
    )
    fig_spark.update_layout(
        height=220,
        margin=dict(t=40, b=20),
        xaxis=dict(dtick=1, tickmode="linear"),
    )
    st.plotly_chart(fig_spark, use_container_width=True)

    # ── Build coordinates ────────────────────────────────────────────────────
    coords = density_mapper.get_coordinates(
        point_type=point_type,
        selected_months=map_months or None,
        hour_range=hour_range,
    )

    st.caption(f"Plotting **{len(coords):,}** trips on the map.")

    # ── Folium map ───────────────────────────────────────────────────────────
    sg_map = folium.Map(
        location=[1.3521, 103.8198],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    if coords:
        HeatMap(
            coords,
            radius=10,
            blur=15,
            min_opacity=0.3,
            max_zoom=14,
        ).add_to(sg_map)

    # Layer control toggle for the base tile
    folium.LayerControl().add_to(sg_map)

    st_folium(sg_map, use_container_width=True, height=560, returned_objects=[])

    if not coords:
        st.warning("No trips with valid coordinates for the selected filters.")
