import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path

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

def main():
    base_dir = Path(__file__).resolve().parent
    qgis_final_exports_dir = base_dir.parent / "QGIS" / "csv_exports" / "final_exports"

    parser = argparse.ArgumentParser(
        description="Merge final OSM points/polygons files and build OD attractiveness tables."
    )
    parser.add_argument(
        "--points",
        default=str(qgis_final_exports_dir / "points_with_locality_v2_final.csv")
    )
    parser.add_argument(
        "--polygons",
        default=str(qgis_final_exports_dir / "multipolygons_with_locality_v2_final.csv")
    )
    parser.add_argument(
        "--outdir",
        default=str(base_dir / "outputs")
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pts = load_one(Path(args.points))
    polys = load_one(Path(args.polygons))

    combined = pd.concat([pts, polys], ignore_index=True)

    # Save before dedup for transparency
    combined.to_csv(outdir / "combined_before_dedup.csv", index=False)

    deduped = dedup(combined)
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

    # Overall totals too
    overall = deduped["trip_category"].value_counts(dropna=False).rename_axis("trip_category").reset_index(name="count")
    overall.to_csv(outdir / "overall_trip_category_counts.csv", index=False)

    print(f"Wrote outputs to: {outdir.resolve()}")
    print("Main next file:", outdir / "locality_attractiveness_matrix.csv")

if __name__ == "__main__":
    main()
