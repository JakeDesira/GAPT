# Running the OD Matrix synthetic trip generation and GUI

This guide shows how to:

- install the Python dependencies,
- generate the baseline synthetic trips and the derived 4-day-week dataset,
- run the Streamlit GUI that visualises statistics and maps.

---

## Prerequisites

- Python 3.8+ (3.9/3.10 recommended)
- The repository checked out locally
- The following reference data placed under the repository (required for generation and maps):
  - `OD Matrix/locality_reference_data/localities_population.csv`
  - `OD Matrix/locality_reference_data/localities_counts.csv`
  - `OD Matrix/locality_reference_data/localities_region.csv`
  - `OD Matrix/QGIS/geopackages/malta_localities.gpkg` (required by the map in the GUI)

If these files are missing the generator or GUI will raise errors when trying to build locality-level predictions.

---

## Install dependencies

Create and activate a virtual environment, then install packages from `requirements.txt`:

```bash
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` in this repo includes the packages used by the generator and the Streamlit GUI (numpy, pandas, plotly, streamlit).

---

## Generate baseline synthetic trips

The baseline trip CSV can be generated from the `src/syndata.py` script. From the project root run:

```bash
python3 src/syndata.py
```

This will create the directory `OD Matrix/generated_trip_data/` (if it does not exist) and write the baseline file:

`OD Matrix/generated_trip_data/synthetic_trips.csv`

You can change the default behaviour by editing the top-level constants in `src/syndata.py` or calling the `generate_and_save_trips()` function from Python.

---

## Create the 4-day week dataset (derived scenario)

Once the baseline CSV exists, run the 4-day derivation script which reads the baseline and writes a 4-day scenario CSV:

```bash
python3 src/4_day.py
```

This writes:

`OD Matrix/generated_trip_data/4dw_full_population.csv`

Alternatively, the Streamlit GUI (below) generates both scenarios on demand and offers an optional "Save current CSV outputs" button to export them.

---

## Run the Streamlit GUI (recommended for interactive analysis)

From the project root either:

```bash
# Option A (recommended): change into src/ then run the app
cd src
streamlit run gui.py

# Option B: run from repo root (make sure PYTHONPATH includes src)
PYTHONPATH=src streamlit run src/gui.py
```

The GUI automatically generates both the baseline and the 4-day datasets (using the same generation logic as the CLI scripts). Use the sidebar controls to adjust:

- number of trips to generate ("Trips generated"),
- 4-day employed trip retention (how many employed trips remain in the 4-day scenario),
- random seed (to reproduce runs).

Click the "Save current CSV outputs" button in the sidebar to export the currently-generated `synthetic_trips.csv` and `4dw_full_population.csv` to `OD Matrix/generated_trip_data/`.

---

## Files & paths

Key paths are defined in `src/paths.py`. By default:

- Reference data: `OD Matrix/locality_reference_data/`
- Generated trips: `OD Matrix/generated_trip_data/`
- GPKG used for mapping: `OD Matrix/QGIS/geopackages/malta_localities.gpkg`

Adjust `src/paths.py` if you want to use different folders.

---

## Troubleshooting

- Missing CSV / GPKG errors: ensure the reference CSVs and the `malta_localities.gpkg` file are present under the paths above.
- Streamlit import errors: make sure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- If maps do not render or GeoPackage parsing fails, confirm `malta_localities.gpkg` is a valid GeoPackage file.

---

## Quick summary

1. pip install -r requirements.txt
2. python3 src/syndata.py        # generate baseline
3. python3 src/4_day.py         # derive 4-day dataset (optional — GUI does this automatically)
4. cd src && streamlit run gui.py   # run the interactive dashboard

If anything should be added to this guide (extra examples, environment tips, or pinned package versions), say which parts to expand and it will be updated.
