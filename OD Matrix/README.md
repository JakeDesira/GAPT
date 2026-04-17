# OD Matrix Directory Layout

- `generated_trip_data/`
  Synthetic trip datasets produced by `src/syndata.py` and `src/4_day.py`.

- `locality_reference_data/`
  Reference tables used by the synthetic trip generator:
  locality populations, locality attractiveness counts, and district-to-locality mappings.

- `od_processing/`
  Scripts and outputs used to build OD/locality attractiveness tables from the QGIS exports.

- `od_processing/outputs/`
  Derived OD processing outputs such as the combined deduplicated tables and the locality attractiveness matrix.

- `QGIS/`
  Parent directory for the GIS workflow assets.

- `QGIS/source_data/`
  Raw GIS source inputs such as the OSM `.pbf` file.

- `QGIS/geopackages/`
  Working and versioned `.gpkg` layers used during the GIS workflow.

- `QGIS/csv_exports/`
  Parent directory for CSV and QMD exports produced from QGIS.

- `QGIS/csv_exports/intermediate/`
  Intermediate and versioned CSV/QMD exports kept for traceability.

- `QGIS/csv_exports/final_exports/`
  Final point and polygon CSV exports consumed by `od_processing/merge_od_inputs.py`.
