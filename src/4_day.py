import numpy as np
import pandas as pd
from numpy.random import default_rng

from syndata import (
    sample_predicted_origin, sample_predicted_destination,
    sample_mode, District_names, OD_PROB,
    Parking_type_labels, No_PV_parking_PROB,
    Cost_parking_PROB, Parking_cost_labels,
    Bus_ticket_labels, Bus_users_ticket_type_PROB
)

RNG = default_rng(42)

# Load  full baseline 
df = pd.read_csv("synthetic_trips.csv")

# splitting groups
employed_mask    = df['labour_status'] == 'Employed' # only employed people/trips will be affected 
baseline_employed = df[employed_mask].copy()
non_employed      = df[~employed_mask].copy() # not effected (NOT employed)

print(f"Total trips:        {len(df)}")
print(f"Employed trips:     {len(baseline_employed)}")
print(f"Non-employed trips: {len(non_employed)}")

# Portugal uplift, this was taken from a portugal 4-day week survey 
PORTUGAL_ACTIVITY_MAP = { # interpretations so they match our malta dataset
    "cultural_activities":     ("Recreation",       0.17),
    "restaurants":             ("Recreation",       0.15),
    "sport_activity":          ("Recreation",       0.22),
    "visiting_family_friends": ("Visiting Someone", 0.42),
    "training_development":    ("Education",        0.15),
    "shopping":                ("Shopping",         0.14),
    "household_work":          ("Personal Errands", 0.18),
}

ALL_PURPOSES = ['Commuting', 'Education', 'Escort Education', 'Shopping',
                'Personal Errands', 'Medical Purposes', 'Recreation',
                'Visiting Someone', 'Other']

purpose_uplift = {}
# loop through portugal activity map then sum every relating category together (in simple terms: recreation+recreation, etc...)
for _, (purpose, uplift) in PORTUGAL_ACTIVITY_MAP.items():
    purpose_uplift[purpose] = purpose_uplift.get(purpose, 0.0) + uplift

PURPOSE_MULTIPLIER = {
    p: (0.0 if p == 'Commuting' else 1.0 + purpose_uplift.get(p, 0.0)) # forces commuting to zero, as its an off day 
    for p in ALL_PURPOSES
}

# Derive adjusted purpose probabilities from employed baseline
baseline_purpose_counts = (
    baseline_employed['purpose']
    .value_counts()
    .reindex(ALL_PURPOSES, fill_value=0)
)

# apply new distribution to the employed group 
baseline_purpose_prob = baseline_purpose_counts / baseline_purpose_counts.sum() # get prob for every purpose 
adjusted_weights      = baseline_purpose_prob * pd.Series(PURPOSE_MULTIPLIER) # apply new weight to each prupose (the effect of the 4 day week)
adjusted_purpose_prob = adjusted_weights / adjusted_weights.sum() # normalisation (sum to 1)

# Time-of-day shift 
TIME_BINS = ['00:00 - 02:59', '03:00 - 05:59', '06:00 - 08:59',
             '09:00 - 11:59', '12:00 - 14:59', '15:00 - 17:59',
             '18:00 - 20:59', '21:00 - 23:59']

time_prob = (
    baseline_employed['time_bin']
    .value_counts()
    .reindex(TIME_BINS, fill_value=0)
    .pipe(lambda s: s / s.sum())
    .values.copy()
)

# move load of traffic to later time bins 
shift = time_prob[2] * 0.20
time_prob[2] -= shift
time_prob[3] += shift / 2
time_prob[4] += shift / 2
adjusted_time_prob = time_prob / time_prob.sum()

STAY_HOME_FACTOR = 0.85 # 85% of the employed population will stay at home, this was taken from portugal survey ("Household work")


employed_4dw = (
    baseline_employed
    .sample(frac=STAY_HOME_FACTOR, random_state=42)
    .copy()
    .reset_index(drop=True)
)

for i, row in employed_4dw.iterrows():

    # Only these fields change 
    new_purpose  = RNG.choice(ALL_PURPOSES, p=adjusted_purpose_prob.values) # based on new probabilities 
    new_mode     = sample_mode(new_purpose) 
    new_time_bin = RNG.choice(TIME_BINS, p=adjusted_time_prob)

    # Destination can change because the reason for travel changed
    origin_idx       = District_names.index(row['origin'])
    new_destination  = RNG.choice(District_names, p=OD_PROB[origin_idx]) # based on origin and probability of traveling to each district 
    new_pred_dest    = sample_predicted_destination(new_destination, new_purpose)

    # Parking — only relevant if Personal Vehicle
    new_parking_type = new_parking_cost = None
    if new_mode == 'Personal Vehicle':
        new_parking_type = RNG.choice(Parking_type_labels, p=No_PV_parking_PROB)
        new_parking_cost = RNG.choice(Parking_cost_labels, p=Cost_parking_PROB)

    # Bus ticket — only relevant if Bus
    new_bus_ticket = None
    if new_mode == 'Bus':
        new_bus_ticket = RNG.choice(Bus_ticket_labels, p=Bus_users_ticket_type_PROB)

    # these are untouched fields 
    employed_4dw.at[i, 'purpose']               = new_purpose
    employed_4dw.at[i, 'mode']                  = new_mode
    employed_4dw.at[i, 'time_bin']              = new_time_bin
    employed_4dw.at[i, 'destination']           = new_destination
    employed_4dw.at[i, 'predicted_destination'] = new_pred_dest
    employed_4dw.at[i, 'parking_type']          = new_parking_type
    employed_4dw.at[i, 'parking_cost']          = new_parking_cost
    employed_4dw.at[i, 'bus_ticket']            = new_bus_ticket

# Reassemble full population
df_4dw_full = pd.concat(
    [non_employed, employed_4dw],
    ignore_index=True
)

df_4dw_full['scenario'] = '4day_week_day_off'

print(f"\nOriginal full dataset:  {len(df)} trips")
print(f"4DW full dataset:       {len(df_4dw_full)} trips")
print(f"  └─ Non-employed:      {len(non_employed)} (completely unchanged)")
print(f"  └─ Employed (4DW):    {len(employed_4dw)} (labour_status preserved as Employed)")

df_4dw_full.to_csv("4dw_full_population.csv", index=False, encoding = 'utf-8-sig')
print("\nSaved → 4dw_full_population.csv")