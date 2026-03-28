"""
Classes
-------
TaxiDataLoader      Load and preprocess the raw CSV.
TaxiDataFilter      Filter a DataFrame by month, division, and weekend flag.
TaxiAggregator      Aggregate a (filtered) DataFrame for charting.
AnomalyDetector     Identify anomalous division codes in the full dataset.
RideSharingAnalyzer Identify concurrent trip groups eligible for ride-sharing.
RideDensityMap      Prepare coordinate data for the folium ride-density heatmap.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

_LOCATION_COLS = [
    "pickup_latitude",
    "pickup_longtitude",
    "destination_latitude",
    "destination_longtitude",
]


# ---------------------------------------------------------------------------
# TaxiDataLoader
# ---------------------------------------------------------------------------

class TaxiDataLoader:
    """Load and preprocess the aggregated taxi expense CSV.

    Usage::

        df = TaxiDataLoader.load()
    """

    @staticmethod
    def load(path: str = "data/aggregated_data.csv") -> pd.DataFrame:
        """Load and preprocess the aggregated taxi expense CSV.

        Coerces numeric and datetime columns, then derives ``month_ts``,
        ``month_label``, and ``weekday_sort`` columns used by downstream classes.

        Args:
            path: Relative or absolute path to the CSV file.

        Returns:
            Cleaned DataFrame with additional derived columns.
        """
        df = pd.read_csv(path)

        # Coerce numeric columns
        for col in ["taxi_fare($)", "admin($)", "total_cost", "distance_run(km)", "duration_minutes"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in _LOCATION_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")

        # month_label: "2015-01" → "Jan 2015"
        df["month_ts"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df["month_label"] = df["month_ts"].dt.strftime("%b %Y")

        # weekday sort order (Mon=0 … Sat=5)
        df["weekday_sort"] = pd.Categorical(
            df["weekday"], categories=_WEEKDAY_ORDER, ordered=True
        ).codes

        return df


# ---------------------------------------------------------------------------
# TaxiDataFilter
# ---------------------------------------------------------------------------

class TaxiDataFilter:
    """Filter a taxi expense DataFrame by month, division, and weekend flag.

    Args:
        df: Full taxi expense DataFrame as returned by :meth:`TaxiDataLoader.load`.

    Usage::

        filtered = TaxiDataFilter(df_all).apply(months, divisions, include_weekends)
        top10    = TaxiDataFilter(df_all).top_division_codes(top_n=10)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def apply(
        self,
        selected_months: list[str],
        selected_divisions: list[str],
        include_weekends: bool,
    ) -> pd.DataFrame:
        """Return a filtered copy of the DataFrame.

        Division filtering supports an ``"All Others"`` sentinel that matches
        any division not in the top-10 by total cost.

        Args:
            selected_months: ``month_label`` values to retain (e.g. ``"Jan 2015"``).
                An empty list skips month filtering.
            selected_divisions: Division codes to retain, plus the optional
                ``"All Others"`` bucket. An empty list skips division filtering.
            include_weekends: When ``False``, Saturday rows are dropped.

        Returns:
            A filtered copy of the DataFrame supplied at construction.
        """
        df = self._df
        mask = pd.Series(True, index=df.index)

        if selected_months:
            mask &= df["month_label"].isin(selected_months)

        if selected_divisions:
            # "All Others" bucket = any division not in the explicit top-N list
            explicit = [d for d in selected_divisions if d != "All Others"]
            if "All Others" in selected_divisions and explicit:
                mask &= df["division_code"].isin(explicit) | ~df["division_code"].isin(
                    self.top_division_codes()
                )
            elif explicit:
                mask &= df["division_code"].isin(explicit)
            # if only "All Others" selected, no division filter applied (show all)

        if not include_weekends:
            mask &= df["weekday"] != "Saturday"

        return df[mask].copy()

    def top_division_codes(self, top_n: int = 10) -> list[str]:
        """Return the *top_n* division codes ranked by total cost.

        Args:
            top_n: Number of top divisions to return.

        Returns:
            List of division code strings, highest spend first.
        """
        return (
            self._df.groupby("division_code")["total_cost"]
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )


# ---------------------------------------------------------------------------
# TaxiAggregator
# ---------------------------------------------------------------------------

class TaxiAggregator:
    """Aggregate a (filtered) taxi expense DataFrame for dashboard charts.

    Args:
        df: Filtered taxi expense DataFrame.

    Usage::

        agg = TaxiAggregator(df)
        monthly_df  = agg.monthly()
        hourly_df   = agg.hourly()
        weekday_df  = agg.weekday()
        div_totals  = agg.division_totals()
        heatmap_df  = agg.division_monthly_heatmap()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def monthly(self) -> pd.DataFrame:
        """Aggregate ride counts and costs by calendar month.

        Returns:
            DataFrame with columns ``month_ts``, ``month_label``, ``num_rides``,
            ``total_cost``, and ``avg_cost_per_trip``, sorted chronologically.
        """
        agg = (
            self._df.groupby(["month_ts", "month_label"], sort=True)
            .agg(num_rides=("total_cost", "count"), total_cost=("total_cost", "sum"))
            .reset_index()
            .sort_values("month_ts")
        )
        agg["avg_cost_per_trip"] = agg["total_cost"] / agg["num_rides"]
        return agg

    def hourly(self) -> pd.DataFrame:
        """Aggregate ride counts and costs by hour of day.

        Returns:
            DataFrame with columns ``hour``, ``num_rides``, ``total_cost``, and
            ``avg_cost``, sorted by hour ascending.
        """
        agg = (
            self._df.groupby("hour", sort=True)
            .agg(num_rides=("total_cost", "count"), total_cost=("total_cost", "sum"))
            .reset_index()
        )
        agg["avg_cost"] = agg["total_cost"] / agg["num_rides"]
        return agg.sort_values("hour")

    def weekday(self) -> pd.DataFrame:
        """Aggregate ride counts and costs by day of the week.

        Returns:
            DataFrame with columns ``weekday``, ``weekday_sort``, ``num_rides``,
            ``total_cost``, and ``avg_cost_per_trip``, sorted Monday–Saturday.
        """
        agg = (
            self._df.groupby(["weekday", "weekday_sort"], sort=False)
            .agg(num_rides=("total_cost", "count"), total_cost=("total_cost", "sum"))
            .reset_index()
            .sort_values("weekday_sort")
        )
        agg["avg_cost_per_trip"] = agg["total_cost"] / agg["num_rides"]
        return agg

    def division_totals(self, top_n: int = 10) -> pd.DataFrame:
        """Aggregate total spend per division, grouping the tail into an "Other" bucket.

        Args:
            top_n: Number of individual divisions to surface; the remainder are
                collapsed into a single ``"Other"`` row.

        Returns:
            DataFrame with columns ``division_code``, ``total_cost``, ``num_rides``,
            ``pct`` (share of grand total), and ``cumulative_pct``.
        """
        agg = (
            self._df.groupby("division_code")
            .agg(total_cost=("total_cost", "sum"), num_rides=("total_cost", "count"))
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )
        grand_total = agg["total_cost"].sum()

        top = agg.head(top_n).copy()
        other_cost = agg.iloc[top_n:]["total_cost"].sum()
        other_rides = agg.iloc[top_n:]["num_rides"].sum()

        other_row = pd.DataFrame([{
            "division_code": "Other",
            "total_cost": other_cost,
            "num_rides": other_rides,
        }])
        result = pd.concat([top, other_row], ignore_index=True)
        result["pct"] = result["total_cost"] / grand_total * 100
        result["cumulative_pct"] = result["pct"].cumsum()
        return result

    def division_monthly_heatmap(self, top_n: int = 15) -> pd.DataFrame:
        """Build a pivot table of monthly spend for the top divisions.

        Rows are the *top_n* divisions by total spend (descending); columns are
        calendar months in chronological order.

        Args:
            top_n: Number of highest-spending divisions to include.

        Returns:
            Pivot DataFrame indexed by ``division_code`` with ``month_label``
            columns. Missing month/division combinations are filled with 0.
        """
        df = self._df
        top_divs = (
            df.groupby("division_code")["total_cost"]
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )
        filtered = df[df["division_code"].isin(top_divs)]

        pivot = (
            filtered.groupby(["division_code", "month_label"])["total_cost"]
            .sum()
            .unstack(fill_value=0)
        )

        # Order columns chronologically
        month_order = (
            df[["month_ts", "month_label"]]
            .drop_duplicates()
            .sort_values("month_ts")["month_label"]
            .tolist()
        )
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

        # Order rows by total spend descending
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        return pivot


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Identify division codes that appear only in July and August 2015.

    Args:
        df: Full (unfiltered) taxi expense DataFrame.

    Usage::

        result = AnomalyDetector(df_all).detect()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def detect(self) -> dict[str, object]:
        """Detect division codes exclusive to July and August 2015.

        Such codes are flagged as anomalous because they never appear in any
        other month, suggesting a data entry error or temporary re-organisation.

        Returns:
            Dictionary with keys:
            - ``"anomalous_codes"``: sorted list of division code strings.
            - ``"count"``: number of anomalous codes.
            - ``"monthly_unique_div_counts"``: DataFrame with ``month_label``
              and ``unique_divisions`` columns, sorted chronologically.
        """
        df = self._df
        all_months = set(df["month_label"].unique())
        jul_label = "Jul 2015"
        aug_label = "Aug 2015"
        other_months = all_months - {jul_label, aug_label}

        divs_other = set(df[df["month_label"].isin(other_months)]["division_code"].unique())
        divs_julaug = set(df[df["month_label"].isin({jul_label, aug_label})]["division_code"].unique())
        anomalous = divs_julaug - divs_other

        # Monthly unique division count (for chart)
        monthly_unique = (
            df.groupby("month_label")["division_code"]
            .nunique()
            .reset_index()
            .rename(columns={"division_code": "unique_divisions"})
        )
        month_order = (
            df[["month_ts", "month_label"]]
            .drop_duplicates()
            .sort_values("month_ts")["month_label"]
            .tolist()
        )
        monthly_unique["month_label"] = pd.Categorical(
            monthly_unique["month_label"], categories=month_order, ordered=True
        )
        monthly_unique = monthly_unique.sort_values("month_label")

        return {
            "anomalous_codes": sorted(anomalous),
            "count": len(anomalous),
            "monthly_unique_div_counts": monthly_unique,
        }


# ---------------------------------------------------------------------------
# RideSharingAnalyzer
# ---------------------------------------------------------------------------

class RideSharingAnalyzer:
    """Identify concurrent trip groups eligible for ride-sharing.

    Args:
        df: Full (unfiltered) taxi expense DataFrame.

    Usage::

        result = RideSharingAnalyzer(df_all).compute_opportunities()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute the great-circle distance between two points.

        Uses the Haversine formula to account for Earth's curvature.

        Args:
            lat1: Latitude of the first point in decimal degrees.
            lon1: Longitude of the first point in decimal degrees.
            lat2: Latitude of the second point in decimal degrees.
            lon2: Longitude of the second point in decimal degrees.

        Returns:
            Distance in kilometres.
        """
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _group_destination_similarity(group: pd.DataFrame) -> pd.Series:
        """Compute pairwise destination distances for a concurrent trip group.

        Intended for use with ``groupby().apply()``. Each group represents trips
        that share a pickup location and 15-minute departure window.

        Args:
            group: Subset DataFrame for a single pickup-location/time-bucket group.

        Returns:
            Series with keys ``num_rides``, ``num_pairs``, and
            ``min_pairwise_dest_km`` (NaN when fewer than two rides exist).
        """
        coords = group[["destination_latitude", "destination_longtitude"]]
        n = len(coords)
        if n < 2:
            return pd.Series({"num_rides": n, "num_pairs": 0, "min_pairwise_dest_km": np.nan})

        distances = [
            RideSharingAnalyzer._haversine(
                coords.loc[i, "destination_latitude"],
                coords.loc[i, "destination_longtitude"],
                coords.loc[j, "destination_latitude"],
                coords.loc[j, "destination_longtitude"],
            )
            for i, j in combinations(coords.index, 2)
        ]
        return pd.Series({
            "num_rides": n,
            "num_pairs": len(distances),
            "min_pairwise_dest_km": float(np.min(distances)),
        })

    def compute_opportunities(
        self,
        window_minutes: int = 15,
        dist_thresholds_km: list[float] = [1.0, 2.0, 5.0],
    ) -> dict[str, object]:
        """Identify concurrent trip groups eligible for ride-sharing.

        Groups trips that depart from the same pickup postal code within a
        rolling time window, then measures destination proximity using the
        Haversine formula to assess whether trips could realistically be shared.

        Args:
            window_minutes: Width of the departure time window used to bucket
                concurrent trips (default 15 minutes).
            dist_thresholds_km: Straight-line destination distance thresholds
                (km) at which a group is considered shareable.

        Returns:
            Dictionary with keys:
            - ``"total_concurrent_groups"``: total number of groups with 2+ trips.
            - ``"threshold_summary"``: DataFrame with ``threshold_km``,
              ``eligible_groups``, and ``pct_groups`` columns.
            - ``"group_detail"``: detailed per-group DataFrame including
              ``min_pairwise_dest_km``.
        """
        # Drop rows with any missing location coordinate
        df_loc = self._df.dropna(subset=_LOCATION_COLS).copy()

        df_loc["time_bucket"] = df_loc["start_datetime"].dt.floor(f"{window_minutes}min")

        group_sim = (
            df_loc.groupby(["pickup_postal", "time_bucket"])
            .apply(self._group_destination_similarity)
            .reset_index()
        )

        # Keep only groups with 2+ concurrent rides
        group_sim = group_sim[group_sim["num_rides"] >= 2].copy()
        total_concurrent = len(group_sim)

        results = []
        for t in dist_thresholds_km:
            eligible = group_sim[group_sim["min_pairwise_dest_km"] <= t]
            results.append({
                "threshold_km": t,
                "eligible_groups": len(eligible),
                "pct_groups": len(eligible) / total_concurrent * 100 if total_concurrent else 0,
            })

        return {
            "total_concurrent_groups": total_concurrent,
            "threshold_summary": pd.DataFrame(results),
            "group_detail": group_sim,
        }


# ---------------------------------------------------------------------------
# RideDensityMap
# ---------------------------------------------------------------------------

class RideDensityMap:
    """Prepare coordinate data for the folium ride-density heatmap tab.

    Args:
        df: Full (unfiltered) taxi expense DataFrame.

    Usage::

        coords = RideDensityMap(df_all).get_coordinates(
            point_type="pickup",
            selected_months=["Jan 2015"],
            hour_range=(8, 10),
        )
    """

    # Singapore bounding box — used to drop erroneous coordinates
    _LAT_BOUNDS = (1.15, 1.50)
    _LON_BOUNDS = (103.55, 104.10)

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_coordinates(
        self,
        point_type: str = "pickup",
        selected_months: list[str] | None = None,
        hour_range: tuple[int, int] = (0, 23),
    ) -> list[list[float]]:
        """Return ``[lat, lon]`` pairs suitable for a folium HeatMap layer.

        Rows outside Singapore's bounding box and those with missing coordinates
        are dropped before returning.

        Args:
            point_type: ``"pickup"`` to use pickup coordinates, ``"destination"``
                to use destination coordinates.
            selected_months: ``month_label`` values to retain. ``None`` or an
                empty list includes all months.
            hour_range: Inclusive ``(start_hour, end_hour)`` tuple (0–23) used
                to restrict trips by departure hour.

        Returns:
            List of ``[lat, lon]`` pairs for each qualifying trip.

        Raises:
            ValueError: If *point_type* is not ``"pickup"`` or ``"destination"``.
        """
        if point_type == "pickup":
            lat_col, lon_col = "pickup_latitude", "pickup_longtitude"
        elif point_type == "destination":
            lat_col, lon_col = "destination_latitude", "destination_longtitude"
        else:
            raise ValueError(f"point_type must be 'pickup' or 'destination', got {point_type!r}")

        df = self._df.dropna(subset=[lat_col, lon_col, "hour"]).copy()

        # Month filter
        if selected_months:
            df = df[df["month_label"].isin(selected_months)]

        # Hour range filter
        start_h, end_h = hour_range
        df = df[(df["hour"] >= start_h) & (df["hour"] <= end_h)]

        # Clip to Singapore bounding box
        lat_min, lat_max = self._LAT_BOUNDS
        lon_min, lon_max = self._LON_BOUNDS
        df = df[
            df[lat_col].between(lat_min, lat_max)
            & df[lon_col].between(lon_min, lon_max)
        ]

        return df[[lat_col, lon_col]].values.tolist()

    def hourly_counts(
        self,
        point_type: str = "pickup",
        selected_months: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return per-hour trip counts for the sparkline above the map.

        Args:
            point_type: ``"pickup"`` or ``"destination"`` — selects which
                coordinate columns are used for the non-null filter.
            selected_months: ``month_label`` values to retain. ``None`` or an
                empty list includes all months.

        Returns:
            DataFrame with columns ``hour`` (0–23) and ``num_rides``, sorted
            by hour ascending. Hours with zero trips are included.
        """
        if point_type == "pickup":
            coord_cols = ["pickup_latitude", "pickup_longtitude"]
        else:
            coord_cols = ["destination_latitude", "destination_longtitude"]

        df = self._df.dropna(subset=coord_cols + ["hour"]).copy()

        if selected_months:
            df = df[df["month_label"].isin(selected_months)]

        counts = (
            df.groupby("hour")
            .size()
            .reindex(range(24), fill_value=0)
            .reset_index()
        )
        counts.columns = ["hour", "num_rides"]
        return counts
