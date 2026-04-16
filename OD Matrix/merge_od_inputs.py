
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from html import escape

POINT_PREF = {"Shopping", "Personal Errands", "Recreation", "Commuting to work"}
POLY_PREF = {"Education", "Medical Purpose", "Visiting Someone"}

def norm_text(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s)
    return s

def norm_name(x: str) -> str:
    s = norm_text(x)
    s = re.sub(r"[^a-z0-9à-ÿ'\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_locality(x: str) -> str:
    s = norm_text(x)
    # keep it light since both files come from the same locality layer
    s = s.replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def choose_locality_col(df: pd.DataFrame) -> str:
    for col in ["name_2", "locality", "city", "name"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a locality-like column (expected one of: name_2, locality, city, name).")

def source_rank(source: str, trip_category: str) -> int:
    if trip_category in POLY_PREF:
        return 0 if source == "polygon" else 1
    if trip_category in POINT_PREF:
        return 0 if source == "point" else 1
    return 0 if source == "point" else 1

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "keep_for_od" in df.columns:
        df = df[df["keep_for_od"].astype(str).str.lower().eq("yes")].copy()
    if "feature_source" not in df.columns:
        # fallback from file name
        df["feature_source"] = "polygon" if "polygon" in path.stem.lower() else "point"
    locality_col = choose_locality_col(df)
    df["locality"] = df[locality_col]
    df["locality_norm"] = df["locality"].map(norm_locality)
    df["name_norm"] = df["name"].map(norm_name) if "name" in df.columns else ""
    df["trip_category"] = df["trip_category"].fillna("Others")
    df["source_rank"] = [
        source_rank(src, cat)
        for src, cat in zip(df["feature_source"], df["trip_category"])
    ]
    df["source_file"] = path.name
    return df

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    # keep unnamed rows; dedup only named rows
    named = df[df["name_norm"] != ""].copy()
    unnamed = df[df["name_norm"] == ""].copy()

    named = named.sort_values(
        by=["locality_norm", "trip_category", "name_norm", "source_rank"]
    )
    named = named.drop_duplicates(
        subset=["locality_norm", "trip_category", "name_norm"], keep="first"
    )
    return pd.concat([named, unnamed], ignore_index=True)

def matrix_cell_colors(value: float, vmin: float, vmax: float) -> tuple[str, str]:
    if not np.isfinite(value):
        return "#f3f4f6", "#111827"
    if vmax == vmin:
        norm = 1.0 if value > 0 else 0.0
    else:
        norm = (value - vmin) / (vmax - vmin)
    norm = min(max(norm, 0.0), 1.0)
    start = (242, 247, 255)
    end = (0, 90, 153)
    r = round(start[0] + (end[0] - start[0]) * norm)
    g = round(start[1] + (end[1] - start[1]) * norm)
    b = round(start[2] + (end[2] - start[2]) * norm)
    text = "#ffffff" if norm >= 0.55 else "#111827"
    return f"rgb({r},{g},{b})", text

def write_visual_matrix_html(matrix: pd.DataFrame, out_path: Path) -> None:
    if matrix.empty:
        out_path.write_text(
            "<!doctype html><html><body><p>No matrix data available.</p></body></html>",
            encoding="utf-8",
        )
        return

    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 0.0
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))

    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Category vs Locality Matrix</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;}",
        "h1{margin:0 0 12px;font-size:22px;}",
        "p{margin:0 0 16px;color:#334155;}",
        "table{border-collapse:collapse;overflow:auto;max-width:100%;display:block;background:white;}",
        "th,td{border:1px solid #d1d5db;padding:6px 9px;text-align:right;white-space:nowrap;}",
        "thead th{position:sticky;top:0;background:#e5e7eb;z-index:2;}",
        "th.row{position:sticky;left:0;text-align:left;background:#f3f4f6;z-index:1;}",
        ".meta{font-size:13px;color:#475569;margin-top:10px;}",
        "</style></head><body>",
        "<h1>Category vs Locality Matrix</h1>",
        "<p>Cell color intensity represents higher destination counts.</p>",
        "<table><thead><tr><th class='row'>Category</th>",
    ]

    for col in matrix.columns:
        html_parts.append(f"<th>{escape(str(col))}</th>")
    html_parts.append("</tr></thead><tbody>")

    for idx, row in matrix.iterrows():
        html_parts.append(f"<tr><th class='row'>{escape(str(idx))}</th>")
        for val in row:
            value = float(val) if pd.notna(val) else np.nan
            bg_color, text_color = matrix_cell_colors(value, vmin, vmax)
            display = "" if pd.isna(val) else f"{int(val):,}"
            html_parts.append(
                f"<td style='background:{bg_color};color:{text_color};'>{display}</td>"
            )
        html_parts.append("</tr>")

    html_parts.extend(
        [
            "</tbody></table>",
            f"<div class='meta'>Minimum value: {int(vmin):,} | Maximum value: {int(vmax):,}</div>",
            "</body></html>",
        ]
    )
    out_path.write_text("".join(html_parts), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Merge final OSM points/polygons files and build OD attractiveness tables.")
    parser.add_argument("--points", default="points_with_locality_v2_final.csv")
    parser.add_argument("--polygons", default="multipolygons_with_locality_v2_final.csv")
    parser.add_argument("--outdir", default="od_outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pts = load_one(Path(args.points))
    polys = load_one(Path(args.polygons))

    combined = pd.concat([pts, polys], ignore_index=True)

    # Save before dedup for transparency
    combined.to_csv(outdir / "combined_before_dedup.csv", index=False)

    deduped = dedup(combined)
    deduped["locality"] = (
        deduped["locality"]
        .fillna("Unknown locality")
        .replace("", "Unknown locality")
    )
    deduped.to_csv(outdir / "combined_after_dedup.csv", index=False)

    counts = (
        deduped.groupby(["locality", "trip_category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["locality", "trip_category"])
    )
    counts.to_csv(outdir / "counts_by_locality_trip_category.csv", index=False)

    pivot = (
        counts.pivot(index="locality", columns="trip_category", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    pivot.to_csv(outdir / "locality_attractiveness_matrix.csv", index=False)

    category_vs_locality = (
        counts.pivot(index="trip_category", columns="locality", values="count")
        .fillna(0)
        .astype(int)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    category_vs_locality.loc["Total"] = category_vs_locality.sum(axis=0).astype(int)
    category_vs_locality.to_csv(outdir / "category_vs_locality_matrix.csv")
    write_visual_matrix_html(
        category_vs_locality, outdir / "category_vs_locality_matrix.html"
    )

    # Overall totals too
    overall = deduped["trip_category"].value_counts(dropna=False).rename_axis("trip_category").reset_index(name="count")
    overall.to_csv(outdir / "overall_trip_category_counts.csv", index=False)

    # Small report
    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("OD input preparation outputs\n")
        f.write("============================\n\n")
        f.write(f"Points rows kept: {len(pts):,}\n")
        f.write(f"Polygon rows kept: {len(polys):,}\n")
        f.write(f"Combined rows before dedup: {len(combined):,}\n")
        f.write(f"Combined rows after dedup: {len(deduped):,}\n\n")
        f.write("Deduplication rule:\n")
        f.write("- Only named features are deduplicated.\n")
        f.write("- Duplicate key = locality + trip_category + normalized name.\n")
        f.write("- Source preference by category:\n")
        f.write("  * polygon preferred for Education, Medical Purpose, Visiting Someone\n")
        f.write("  * point preferred for Shopping, Personal Errands, Recreation, Commuting to work\n")
        f.write("- Unnamed features are kept as-is.\n\n")
        f.write("Main files to use next:\n")
        f.write("- locality_attractiveness_matrix.csv (locality x trip_category)\n")
        f.write("- category_vs_locality_matrix.csv (trip_category x locality)\n")
        f.write("- category_vs_locality_matrix.html (visual heatmap)\n\n")
        f.write("This matrix can now be used as destination attractiveness A_jk in the OD model.\n")

    print(f"Wrote outputs to: {outdir.resolve()}")
    print("Main next file:", outdir / "locality_attractiveness_matrix.csv")
    print("Visual matrix:", outdir / "category_vs_locality_matrix.html")

if __name__ == "__main__":
    main()
