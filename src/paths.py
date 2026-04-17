from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

OD_MATRIX_DIR = ROOT_DIR / "OD Matrix"
OD_REFERENCE_DIR = OD_MATRIX_DIR / "locality_reference_data"
GENERATED_TRIP_DATA_DIR = OD_MATRIX_DIR / "generated_trip_data"

QGIS_DIR = OD_MATRIX_DIR / "QGIS"
QGIS_SOURCE_DATA_DIR = QGIS_DIR / "source_data"
QGIS_GEOPACKAGES_DIR = QGIS_DIR / "geopackages"
QGIS_CSV_EXPORTS_DIR = QGIS_DIR / "csv_exports"
QGIS_CSV_INTERMEDIATE_DIR = QGIS_CSV_EXPORTS_DIR / "intermediate"
QGIS_FINAL_EXPORTS_DIR = QGIS_CSV_EXPORTS_DIR / "final_exports"
LOCALITY_BOUNDARIES_GPKG_PATH = QGIS_GEOPACKAGES_DIR / "malta_localities.gpkg"

OD_PROCESSING_DIR = OD_MATRIX_DIR / "od_processing"
OD_PROCESSING_OUTPUTS_DIR = OD_PROCESSING_DIR / "outputs"

LOCALITIES_COUNTS_PATH = OD_REFERENCE_DIR / "localities_counts.csv"
LOCALITIES_POPULATION_PATH = OD_REFERENCE_DIR / "localities_population.csv"
LOCALITIES_REGION_PATH = OD_REFERENCE_DIR / "localities_region.csv"

BASELINE_TRIPS_PATH = GENERATED_TRIP_DATA_DIR / "synthetic_trips.csv"
FOUR_DAY_TRIPS_PATH = GENERATED_TRIP_DATA_DIR / "4dw_full_population.csv"


def ensure_project_directories() -> None:
    for directory in [
        GENERATED_TRIP_DATA_DIR,
        OD_PROCESSING_DIR,
        OD_PROCESSING_OUTPUTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def generated_trip_file(filename: str) -> Path:
    return GENERATED_TRIP_DATA_DIR / filename
