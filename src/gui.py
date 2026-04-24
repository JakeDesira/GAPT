import importlib
import re
import sqlite3
import struct
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st

from paths import (
    BASELINE_TRIPS_PATH,
    FOUR_DAY_TRIPS_PATH,
    GENERATED_TRIP_DATA_DIR,
    LOCALITY_BOUNDARIES_GPKG_PATH,
)
from syndata import (
    DEFAULT_SEED,
    generate_trips_with_seed,
    save_generated_trips,
)

four_day_module = importlib.import_module("4_day")
DEFAULT_EMPLOYED_TRIP_RETENTION = four_day_module.DEFAULT_EMPLOYED_TRIP_RETENTION
generate_4day_week_dataset = four_day_module.generate_4day_week_dataset
save_4day_week_dataset = four_day_module.save_4day_week_dataset


st.set_page_config(page_title="Malta 4DW Impact Analysis", layout="wide")

TIME_ORDER = [
    "00:00 - 02:59",
    "03:00 - 05:59",
    "06:00 - 08:59",
    "09:00 - 11:59",
    "12:00 - 14:59",
    "15:00 - 17:59",
    "18:00 - 20:59",
    "21:00 - 23:59",
]

HEATMAP_SCALE = [
    (0.00, "#f8f3e8"),
    (0.20, "#f1d7a2"),
    (0.45, "#e49b42"),
    (0.70, "#c95d2d"),
    (1.00, "#6e1d1b"),
]
MAP_CENTER = {"lat": 35.95, "lon": 14.40}
MAP_ZOOM = 8.85


@st.cache_data(show_spinner="Generating trips and refreshing plots...")
def build_scenarios(
    trip_count: int,
    seed: int,
    employed_trip_retention: float,
):
    baseline_df = generate_trips_with_seed(trip_count, seed=seed)
    baseline_df["scenario"] = "5-Day (Baseline)"

    four_day_df = generate_4day_week_dataset(
        baseline_df,
        employed_trip_retention=employed_trip_retention,
        seed=seed,
    )
    four_day_df["scenario"] = "4-Day Week"

    combined_df = pd.concat([baseline_df, four_day_df], ignore_index=True)
    return baseline_df, four_day_df, combined_df


def normalize_locality_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("'", "'").replace("`", "'").replace("'", " ")
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def parse_gpkg_header(blob: bytes):
    flags = blob[3]
    endian = "<" if (flags & 1) else ">"
    envelope_code = (flags >> 1) & 0b111
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code, 0)

    bbox = None
    if envelope_bytes >= 32:
        minx, maxx, miny, maxy = struct.unpack_from(f"{endian}dddd", blob, 8)
        bbox = {
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy,
        }

    return endian, 8 + envelope_bytes, bbox


def parse_ring(data: bytes, offset: int, endian: str):
    point_count = struct.unpack_from(f"{endian}I", data, offset)[0]
    offset += 4
    points = []
    for _ in range(point_count):
        lon, lat = struct.unpack_from(f"{endian}dd", data, offset)
        points.append([lon, lat])
        offset += 16
    return points, offset


def parse_polygon(data: bytes, offset: int):
    byte_order = data[offset]
    endian = "<" if byte_order == 1 else ">"
    offset += 1
    geometry_type = struct.unpack_from(f"{endian}I", data, offset)[0] % 1000
    offset += 4
    if geometry_type != 3:
        raise ValueError(f"Expected Polygon WKB, got geometry type {geometry_type}")

    ring_count = struct.unpack_from(f"{endian}I", data, offset)[0]
    offset += 4

    rings = []
    for _ in range(ring_count):
        ring, offset = parse_ring(data, offset, endian)
        rings.append(ring)

    return rings, offset


def parse_wkb_geometry(data: bytes, offset: int):
    byte_order = data[offset]
    endian = "<" if byte_order == 1 else ">"
    geometry_type = struct.unpack_from(f"{endian}I", data, offset + 1)[0] % 1000

    if geometry_type == 6:
        offset += 1
        offset += 4
        polygon_count = struct.unpack_from(f"{endian}I", data, offset)[0]
        offset += 4

        polygons = []
        for _ in range(polygon_count):
            polygon, offset = parse_polygon(data, offset)
            polygons.append(polygon)
        return {"type": "MultiPolygon", "coordinates": polygons}

    if geometry_type == 3:
        polygon, _ = parse_polygon(data, offset)
        return {"type": "MultiPolygon", "coordinates": [polygon]}

    raise ValueError(f"Unsupported WKB geometry type {geometry_type}")


def alias_boundary_name(name: str, center_lat: float) -> str:
    if name == "Ir-Rabat" and center_lat >= 36.0:
        return "Ir-Rabat, Ghawdex"
    return name


@st.cache_data(show_spinner=False)
def load_locality_boundaries():
    connection = sqlite3.connect(LOCALITY_BOUNDARIES_GPKG_PATH)
    rows = connection.execute(
        "SELECT fid, name, geom FROM matching_features WHERE name IS NOT NULL"
    ).fetchall()
    connection.close()

    features = []
    records = []
    for fid, name, geom in rows:
        _, wkb_offset, bbox = parse_gpkg_header(geom)
        geometry = parse_wkb_geometry(geom, wkb_offset)

        center_lon = (bbox["minx"] + bbox["maxx"]) / 2
        center_lat = (bbox["miny"] + bbox["maxy"]) / 2
        display_name = alias_boundary_name(name, center_lat)
        locality_key = normalize_locality_name(display_name)

        features.append(
            {
                "type": "Feature",
                "id": locality_key,
                "properties": {
                    "fid": fid,
                    "locality": display_name,
                    "locality_key": locality_key,
                },
                "geometry": geometry,
            }
        )
        records.append(
            {
                "locality": display_name,
                "locality_key": locality_key,
                "center_lon": center_lon,
                "center_lat": center_lat,
            }
        )

    metadata_df = pd.DataFrame(records).drop_duplicates(subset=["locality_key"], keep="first")
    return {"type": "FeatureCollection", "features": features}, metadata_df


def build_heatmap_dataframe(df: pd.DataFrame):
    geojson, metadata_df = load_locality_boundaries()
    trip_counts = (
        df["predicted_origin"]
        .dropna()
        .map(normalize_locality_name)
        .value_counts()
        .rename_axis("locality_key")
        .reset_index(name="trips")
    )

    map_df = metadata_df.merge(trip_counts, on="locality_key", how="left")
    map_df["trips"] = map_df["trips"].fillna(0).astype(int)
    total_trips = max(int(map_df["trips"].sum()), 1)
    map_df["share_pct"] = (map_df["trips"] / total_trips) * 100
    return geojson, map_df


def build_time_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(["time_bin", "scenario"]).size().reset_index(name="Count")
    stats["time_bin"] = pd.Categorical(stats["time_bin"], categories=TIME_ORDER, ordered=True)
    return stats.sort_values("time_bin")


def build_grouped_counts(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    stats = df.groupby([column, "scenario"]).size().reset_index(name="Count")
    order = (
        df.groupby(column)
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    stats[label] = pd.Categorical(stats[column], categories=order, ordered=True)
    return stats.drop(columns=[column]).sort_values(label)


def build_top_locality_comparison(df: pd.DataFrame, column: str, label: str, top_n: int = 10) -> pd.DataFrame:
    top_localities = df[column].value_counts().nlargest(top_n).index.tolist()
    stats = (
        df[df[column].isin(top_localities)]
        .groupby([column, "scenario"])
        .size()
        .reset_index(name="Count")
    )
    order = (
        df[df[column].isin(top_localities)]
        .groupby(column)
        .size()
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    stats[label] = pd.Categorical(stats[column], categories=order, ordered=True)
    return stats.drop(columns=[column]).sort_values(label)


def render_heatmap(df: pd.DataFrame, title: str, key_prefix: str = ""):
    geojson, map_df = build_heatmap_dataframe(df)
    if map_df.empty:
        st.warning("No mapped localities are available for the current dataset.")
        return

    max_trips = max(int(map_df["trips"].max()), 1)
    choropleth = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="locality_key",
        featureidkey="properties.locality_key",
        color="trips",
        hover_name="locality",
        custom_data=["share_pct"],
        color_continuous_scale=HEATMAP_SCALE,
        range_color=(0, max_trips),
        opacity=0.78,
        zoom=MAP_ZOOM,
        height=680,
        center=MAP_CENTER,
        mapbox_style="open-street-map",
        title=title,
    )
    choropleth.update_traces(
        marker_line_width=1.2,
        marker_line_color="rgba(255, 255, 255, 0.88)",
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Trips: %{z:,}<br>"
            "Share of predicted origins: %{customdata[0]:.2f}%<extra></extra>"
        ),
    )

    figure = choropleth
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        coloraxis_colorbar={"title": "Trips"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.55)",
        },
    )
    st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_heatmap")
    st.caption(
        "Locality polygons are shaded by predicted trip origins. White borders show the locality "
        "boundaries from `malta_localities.gpkg`."
    )


def render_single_scenario_localities(df: pd.DataFrame, key_prefix: str = ""):
    top_origins = df["predicted_origin"].value_counts().nlargest(10).reset_index()
    top_origins.columns = ["Locality", "Trips"]

    top_destinations = df["predicted_destination"].value_counts().nlargest(10).reset_index()
    top_destinations.columns = ["Locality", "Trips"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Origin Localities")
        figure = px.bar(
            top_origins,
            x="Trips",
            y="Locality",
            orientation="h",
            color="Trips",
            color_continuous_scale="Viridis",
            text_auto=".2s",
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_top_origins")
    with col2:
        st.markdown("### Top Destination Localities")
        figure = px.bar(
            top_destinations,
            x="Trips",
            y="Locality",
            orientation="h",
            color="Trips",
            color_continuous_scale="Magma",
            text_auto=".2s",
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_top_destinations")


def render_single_scenario_breakdowns(df: pd.DataFrame, key_prefix: str = ""):
    mode_counts = df["mode"].value_counts().reset_index()
    mode_counts.columns = ["Mode", "Trips"]

    purpose_counts = df["purpose"].value_counts().reset_index()
    purpose_counts.columns = ["Purpose", "Trips"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Mode Split")
        st.plotly_chart(
            px.pie(mode_counts, names="Mode", values="Trips", hole=0.5),
            use_container_width=True,
            key=f"{key_prefix}_mode_pie",
        )
    with col2:
        st.markdown("### Purpose Split")
        figure = px.bar(
            purpose_counts,
            x="Purpose",
            y="Trips",
            color="Trips",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_purpose_bar")


def render_comparison_tab(combined_df: pd.DataFrame):
    st.subheader("Scenario Comparison")
    st.write(
        "Every chart in this tab compares the 5-day baseline against the 4-day week scenario "
        "using the same simulation settings."
    )

    st.markdown("### Traffic Clock")
    st.plotly_chart(
        px.line(
            build_time_stats(combined_df),
            x="time_bin",
            y="Count",
            color="scenario",
            markers=True,
            title="Hourly Volume Comparison",
        ),
        use_container_width=True,
        key="comparison_traffic_clock",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Mode Comparison")
        mode_stats = build_grouped_counts(combined_df, "mode", "Mode")
        st.plotly_chart(
            px.bar(mode_stats, x="Mode", y="Count", color="scenario", barmode="group"),
            use_container_width=True,
            key="comparison_mode",
        )
    with col2:
        st.markdown("### Purpose Comparison")
        purpose_stats = build_grouped_counts(combined_df, "purpose", "Purpose")
        st.plotly_chart(
            px.bar(purpose_stats, x="Purpose", y="Count", color="scenario", barmode="group"),
            use_container_width=True,
            key="comparison_purpose",
        )

    st.markdown("### Busiest Origin Localities")
    locality_stats = build_top_locality_comparison(combined_df, "predicted_origin", "Locality")
    figure = px.bar(
        locality_stats,
        x="Count",
        y="Locality",
        color="scenario",
        barmode="group",
        orientation="h",
    )
    figure.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, use_container_width=True, key="comparison_localities")


def render_single_scenario_tab(df: pd.DataFrame, scenario_title: str, key_prefix: str = ""):
    st.subheader(scenario_title)
    render_heatmap(df, f"Trip Density by Predicted Origin ({scenario_title})", key_prefix=key_prefix)
    render_single_scenario_localities(df, key_prefix=key_prefix)
    render_single_scenario_breakdowns(df, key_prefix=key_prefix)
    with st.expander("Show sample rows"):
        st.dataframe(df.head(50), use_container_width=True)


st.title("Malta 4-Day Work Week Impact Study")
st.markdown(
    "This dashboard always generates **both** scenarios together. "
    "Use the comparison tab to compare them, and the individual scenario tabs "
    "to inspect each dataset on its own."
)

with st.sidebar:
    st.header("Simulation Controls")
    trip_count = st.slider(
        "Trips generated",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=5000,
        help="Total synthetic trips generated for the 5-day baseline before the 4-day scenario is derived from it.",
    )
    employed_trip_retention = st.slider(
        "4-day employed trip retention",
        min_value=0.10,
        max_value=1.00,
        value=float(DEFAULT_EMPLOYED_TRIP_RETENTION),
        step=0.05,
        help="Share of employed trips kept in the 4-day scenario. Lower values mean a stronger reduction.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=DEFAULT_SEED,
        step=1,
        help="Use the same seed to reproduce the same synthetic datasets.",
    )
    with st.expander("What these controls do"):
        st.markdown(
            "- `Trips generated`: sets the baseline simulation size.\n"
            "- `4-day employed trip retention`: controls how many employed trips remain in the 4-day scenario.\n"
            "- `Random seed`: keeps the generated datasets repeatable."
        )
    save_requested = st.button("Save current CSV outputs", use_container_width=True)
    st.caption(
        "Optional export. This saves the currently displayed baseline and 4-day CSV files to "
        f"`{GENERATED_TRIP_DATA_DIR}` for reuse outside the app."
    )

baseline_df, four_day_df, combined_df = build_scenarios(
    trip_count=trip_count,
    seed=int(seed),
    employed_trip_retention=employed_trip_retention,
)

if save_requested:
    baseline_path = save_generated_trips(baseline_df, BASELINE_TRIPS_PATH)
    four_day_path = save_4day_week_dataset(four_day_df, FOUR_DAY_TRIPS_PATH)
    st.success(
        "Saved current CSV outputs:\n\n"
        f"- `{baseline_path.name}`\n"
        f"- `{four_day_path.name}`"
    )

baseline_total = len(baseline_df)
four_day_total = len(four_day_df)
trips_removed = baseline_total - four_day_total
reduction_pct = (trips_removed / baseline_total) * 100 if baseline_total else 0.0
retained_employed = int((four_day_df["labour_status"] == "Employed").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Baseline Trips", f"{baseline_total:,}")
m2.metric("4-Day Trips", f"{four_day_total:,}", delta=f"-{trips_removed:,} vs baseline", delta_color="inverse")
m3.metric("Trips Removed", f"{trips_removed:,}")
m4.metric("Employed Trips Retained", f"{retained_employed:,}")

st.caption("Plots update after each control change. Larger trip counts take longer to regenerate.")
st.info(
    f"Current simulation: {trip_count:,} baseline trips, seed {int(seed)}, "
    f"and 4-day employed trip retention set to {employed_trip_retention:.2f}. "
    f"Overall trip reduction: {reduction_pct:.1f}%."
)

comparison_tab, baseline_tab, four_day_tab = st.tabs(
    ["Scenario Comparison", "5-Day Baseline", "4-Day Week"]
)

with comparison_tab:
    render_comparison_tab(combined_df)

with baseline_tab:
    render_single_scenario_tab(baseline_df, "5-Day Baseline", key_prefix="baseline")

with four_day_tab:
    render_single_scenario_tab(four_day_df, "4-Day Week", key_prefix="four_day")