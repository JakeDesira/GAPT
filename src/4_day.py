import pandas as pd
from numpy.random import default_rng

from paths import (
    BASELINE_TRIPS_PATH,
    FOUR_DAY_TRIPS_PATH,
    ensure_project_directories,
)
from syndata import (
    Bus_ticket_labels,
    Bus_users_ticket_type_PROB,
    Cost_parking_PROB,
    District_names,
    No_PV_parking_PROB,
    OD_PROB,
    Parking_cost_labels,
    Parking_type_labels,
    sample_mode,
    sample_predicted_destination,
)


DEFAULT_SEED = 42
DEFAULT_EMPLOYED_TRIP_RETENTION = 0.85

PORTUGAL_ACTIVITY_MAP = {
    "cultural_activities": ("Recreation", 0.17),
    "restaurants": ("Recreation", 0.15),
    "sport_activity": ("Recreation", 0.22),
    "visiting_family_friends": ("Visiting Someone", 0.42),
    "training_development": ("Education", 0.15),
    "shopping": ("Shopping", 0.14),
    "household_work": ("Personal Errands", 0.18),
}

ALL_PURPOSES = [
    "Commuting",
    "Education",
    "Escort Education",
    "Shopping",
    "Personal Errands",
    "Medical Purposes",
    "Recreation",
    "Visiting Someone",
    "Other",
]

TIME_BINS = [
    "00:00 - 02:59",
    "03:00 - 05:59",
    "06:00 - 08:59",
    "09:00 - 11:59",
    "12:00 - 14:59",
    "15:00 - 17:59",
    "18:00 - 20:59",
    "21:00 - 23:59",
]


def _build_adjusted_purpose_probabilities(baseline_employed: pd.DataFrame) -> pd.Series:
    purpose_uplift = {}
    for _, (purpose, uplift) in PORTUGAL_ACTIVITY_MAP.items():
        purpose_uplift[purpose] = purpose_uplift.get(purpose, 0.0) + uplift

    purpose_multiplier = {
        purpose: (0.0 if purpose == "Commuting" else 1.0 + purpose_uplift.get(purpose, 0.0))
        for purpose in ALL_PURPOSES
    }

    baseline_purpose_counts = (
        baseline_employed["purpose"].value_counts().reindex(ALL_PURPOSES, fill_value=0)
    )
    baseline_purpose_prob = baseline_purpose_counts / baseline_purpose_counts.sum()
    adjusted_weights = baseline_purpose_prob * pd.Series(purpose_multiplier)
    return adjusted_weights / adjusted_weights.sum()


def _build_adjusted_time_probabilities(baseline_employed: pd.DataFrame):
    time_prob = (
        baseline_employed["time_bin"]
        .value_counts()
        .reindex(TIME_BINS, fill_value=0)
        .pipe(lambda series: series / series.sum())
        .values.copy()
    )

    shift = time_prob[2] * 0.20
    time_prob[2] -= shift
    time_prob[3] += shift / 2
    time_prob[4] += shift / 2
    return time_prob / time_prob.sum()


def generate_4day_week_dataset(
    baseline_df: pd.DataFrame,
    employed_trip_retention: float = DEFAULT_EMPLOYED_TRIP_RETENTION,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if baseline_df.empty:
        return baseline_df.copy()

    rng = default_rng(seed)

    employed_mask = baseline_df["labour_status"] == "Employed"
    baseline_employed = baseline_df[employed_mask].copy()
    non_employed = baseline_df[~employed_mask].copy()

    if baseline_employed.empty:
        result = non_employed.reset_index(drop=True)
        result["scenario"] = "4day_week_day_off"
        return result

    adjusted_purpose_prob = _build_adjusted_purpose_probabilities(baseline_employed)
    adjusted_time_prob = _build_adjusted_time_probabilities(baseline_employed)

    employed_trip_retention = min(max(employed_trip_retention, 0.0), 1.0)
    employed_4dw = (
        baseline_employed
        .sample(frac=employed_trip_retention, random_state=seed)
        .copy()
        .reset_index(drop=True)
    )

    for index, row in employed_4dw.iterrows():
        new_purpose = rng.choice(ALL_PURPOSES, p=adjusted_purpose_prob.values)
        new_mode = sample_mode(new_purpose, rng=rng)
        new_time_bin = rng.choice(TIME_BINS, p=adjusted_time_prob)

        origin_idx = District_names.index(row["origin"])
        new_destination = rng.choice(District_names, p=OD_PROB[origin_idx])
        new_predicted_destination = sample_predicted_destination(
            new_destination,
            new_purpose,
            rng=rng,
        )

        new_parking_type = None
        new_parking_cost = None
        if new_mode == "Personal Vehicle":
            new_parking_type = rng.choice(Parking_type_labels, p=No_PV_parking_PROB)
            new_parking_cost = rng.choice(Parking_cost_labels, p=Cost_parking_PROB)

        new_bus_ticket = None
        if new_mode == "Bus":
            new_bus_ticket = rng.choice(Bus_ticket_labels, p=Bus_users_ticket_type_PROB)

        employed_4dw.at[index, "purpose"] = new_purpose
        employed_4dw.at[index, "mode"] = new_mode
        employed_4dw.at[index, "time_bin"] = new_time_bin
        employed_4dw.at[index, "destination"] = new_destination
        employed_4dw.at[index, "predicted_destination"] = new_predicted_destination
        employed_4dw.at[index, "parking_type"] = new_parking_type
        employed_4dw.at[index, "parking_cost"] = new_parking_cost
        employed_4dw.at[index, "bus_ticket"] = new_bus_ticket

    result = pd.concat([non_employed, employed_4dw], ignore_index=True)
    result["scenario"] = "4day_week_day_off"
    return result


def save_4day_week_dataset(
    dataset: pd.DataFrame,
    output_path=FOUR_DAY_TRIPS_PATH,
):
    ensure_project_directories()
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def generate_and_save_4day_week_dataset(
    baseline_df: pd.DataFrame,
    employed_trip_retention: float = DEFAULT_EMPLOYED_TRIP_RETENTION,
    seed: int = DEFAULT_SEED,
    output_path=FOUR_DAY_TRIPS_PATH,
):
    dataset = generate_4day_week_dataset(
        baseline_df,
        employed_trip_retention=employed_trip_retention,
        seed=seed,
    )
    saved_path = save_4day_week_dataset(dataset, output_path=output_path)
    return dataset, saved_path


def main():
    baseline_df = pd.read_csv(BASELINE_TRIPS_PATH)
    dataset, saved_path = generate_and_save_4day_week_dataset(baseline_df)

    employed_count = int((baseline_df["labour_status"] == "Employed").sum())
    non_employed_count = len(baseline_df) - employed_count
    retained_employed_count = int((dataset["labour_status"] == "Employed").sum())

    print(f"Total trips:        {len(baseline_df)}")
    print(f"Employed trips:     {employed_count}")
    print(f"Non-employed trips: {non_employed_count}")
    print(f"\nOriginal full dataset:  {len(baseline_df)} trips")
    print(f"4DW full dataset:       {len(dataset)} trips")
    print(f"  └─ Non-employed:      {non_employed_count} (completely unchanged)")
    print(
        f"  └─ Employed (4DW):    {retained_employed_count} "
        "(labour_status preserved as Employed)"
    )
    print(f"\nSaved -> {saved_path}")


if __name__ == "__main__":
    main()
