import numpy as np 
import pandas as pd
from numpy.random import default_rng
from paths import (
    BASELINE_TRIPS_PATH,
    LOCALITIES_COUNTS_PATH,
    LOCALITIES_POPULATION_PATH,
    LOCALITIES_REGION_PATH,
    ensure_project_directories,
)


DEFAULT_TRIP_COUNT = 100000
DEFAULT_SEED = 42
RNG = default_rng(DEFAULT_SEED) # seeded random generator for reproducibility

# ── Locality-level data ──────────────────────────────────────────────────────

# Load population data (used for weighted origin selection)
_pop_raw = pd.read_csv(LOCALITIES_POPULATION_PATH)
# The CSV is wide: one row called "Population", localities are the columns
Locality_names = [c for c in _pop_raw.columns if c != "Locality"]
# Values are stored as strings like "29,482" — strip commas before converting
_pop_values = (
    _pop_raw[Locality_names]
    .iloc[0]
    .str.replace(",", "")
    .apply(pd.to_numeric, errors="coerce")  # localities with no data become NaN
    .fillna(0)                              # give them zero weight
    .values
    .astype(float)
)

# Load POI counts per locality, by trip purpose (used for weighted destination)
_poi_raw = pd.read_csv(LOCALITIES_COUNTS_PATH).set_index("Category")

# Map each syndata Purpose label → the matching POI category row name
# Purposes that have no dedicated POI row fall back to "Others"
PURPOSE_TO_POI = {
    "Commuting":        "Commuting to work",
    "Education":        "Education",
    "Escort Education": "Education",         # no dedicated POI row — shares Education
    "Shopping":         "Shopping",
    "Personal Errands": "Personal Errands",
    "Medical Purposes": "Medical Purposes",
    "Recreation":       "Recreation",
    "Visiting Someone": "Visiting Someone",
    "Other":            "Other",
}

# ── Region → locality mapping (for constrained predicted origin/destination) ─
#
# The region CSV is wide: each column header is a district name, each cell is
# a locality in that district.  We pre-compute all weight arrays once at
# startup so generate_trip() just does a fast dict lookup — no per-trip work.
#
#   Region_locality_names[district]                  → list of locality strings
#   Origin_weights_by_region[district]               → population-weighted probs
#   Destination_weights_by_region[district][purpose] → POI-weighted probs

_region_raw = pd.read_csv(LOCALITIES_REGION_PATH)

Region_locality_names = {}
for district in _region_raw.columns:
    Region_locality_names[district] = _region_raw[district].dropna().tolist()

# Pre-compute population-weighted origin probabilities, restricted per region
Origin_weights_by_region = {}
for district, locs in Region_locality_names.items():
    indices = [Locality_names.index(l) for l in locs]
    weights = _pop_values[indices]
    if weights.sum() == 0:
        weights = np.ones(len(locs))
    Origin_weights_by_region[district] = weights / weights.sum()

# Pre-compute POI-weighted destination probabilities, restricted per region × purpose
Destination_weights_by_region = {}
for district, locs in Region_locality_names.items():
    Destination_weights_by_region[district] = {}
    for purpose, poi_category in PURPOSE_TO_POI.items():
        weights = _poi_raw.loc[poi_category, locs].astype(float).values
        if weights.sum() == 0:
            weights = np.ones(len(locs))
        Destination_weights_by_region[district][purpose] = weights / weights.sum()

# ────────────────────────────────────────────────────────────────────────────

# labels for catagorical list
District_names = ['Southern Harbour', 'Northern Harbour', 'South Eastern', 'Western', 'Northern', 'Gozo and Comino']
Age_groups = ['15-24', '25-44', '45-64','65+']
Gender = ['Male', 'Female']
Purpose = ['Commuting', 'Education', 'Escort Education', 'Shopping', 'Personal Errands', 'Medical Purposes', 'Recreation', 'Visiting Someone','Other'] # from chart 2
Mode_transport_4 = ['Personal Vehicle', 'Bus', 'Walking', 'Other'] # for charts 3.2, 3.4, 3.5
Mode_transport_5 = ['Personal Vehicle', 'Bus', 'Walking', 'Organised school transport', 'Other'] # for chart 3.1 and 3.3
Time_bins = ['00:00 - 02:59', '03:00 - 05:59', '06:00 - 08:59', '09:00 - 11:59', '12:00 - 14:59', '15:00 - 17:59', '18:00 - 20:59', '21:00 - 23:59'] # from chart 13
Labour_status = ['Employed', 'Unemployed', 'Student', 'Retired', 'Cannot Work', 'Homemaker']
Bus_purpose = ['Going Home', 'Commuting', 'Education', 'Other']
Bus_concerns = ['Takes Too long', 'Too Many Changes', "Not Punctual", "Stop Too Far", "Inconvenient", "Always Full", "Uncomfortable", "Prefer Alternatives"]
Destination_purposes = ['Education / Escort Education', 'Commuting', 'Shopping', 'Personal Errands', 'Going Home', 'Escort Other', 'Medical Purposes', 'Recreation', 'Visiting Someone', 'Other'] # for table7
# Note: this list mirrors the 10 NSO Table 7 categories, which differ from the 9 Purpose categories.
# 'Going Home' and 'Escort Other' are Table 7 specific — they have no equivalent in Purpose.
# 'Education / Escort Education' maps to both 'Education' and 'Escort Education' in Purpose.
Vehicle_ownership_labels = ['0 vehicles', '1 vehicle', '2 vehicles', '3 vehicles', '4+ vehicles']
Parking_type_labels = ['Did not park', 'Parked on the street', 'Free parking', 'Paid parking', 'Other']
Parking_cost_labels = ['Free', 'Less than 2 euros', '2-4 euros', 'More than 4 euros']
Bus_ticket_labels       = ['Concession', 'Free student', 'Free youth', 'Standard ticket', 'Tallinja on demand'] # order matches Table 18 (users by ticket type)
Bus_trips_ticket_labels = ['Tallinja on demand', 'Standard ticket', 'Free youth', 'Free student', 'Concession'] # intentionally different order — matches Chart 10 (trips by ticket type)
Bus_payment_labels = ['Cash', 'Free', 'Tallinja Card']
Nationality_labels = ['Maltese', 'Foreign']

# helper function to verify that probability arrays sum to 1 (or close enough, given rounding)
def verify_prob(arr, name):
    total = arr.sum(axis=-1)
    if not np.allclose(total, 1.0, atol=0.01):
        print(f"Warning: {name} does not sum to 1.0, got {total}")
    else:
        print(f"Ok: {name} sums to 1.0")

# Table 1:  Number of persons with access to a car by age group and district of residence (LAU 1)
Car_access_district_age = np.array([
    [3505,  22470, 15574,  8614],   # Southern Harbour
    [7076,  49334, 29247, 16778],   # Northern Harbour
    [4299,  23566, 16187,  7021],   # South Eastern
    [3636,  18689, 14345,  7579],   # Western
    [4446,  30259, 20813,  8530],   # Northern
    [2284,  10448,  9342,  5561],   # Gozo and Comino
], dtype=float)

# Table 2: Table 2. Households by number of vehicles and district of residence (LAU 1)
# [0 vehicles, 1 vehicle, 2 vehicles, 3 vehicles, 4+ vehicles]
Vechiles_district = np.array([
    [4708,  9420, 10614, 4818, 3278],   # Southern Harbour
    [8204, 19057, 22693, 8806, 5595],   # Northern Harbour
    [2189,  7162, 10985, 5250, 3922],   # South Eastern
    [1655,  4719,  9635, 4421, 3538],   # Western
    [2566, 10406, 13187, 6348, 4780],   # Northern
    [ 973,  3837,  4733, 3259, 2692],   # Gozo and Comino
], dtype=float)

# normalize into probabilities 
Vehicle_ownership_PROB = (
    Vechiles_district /
    Vechiles_district.sum(axis=1, keepdims=True)
)
verify_prob(Vehicle_ownership_PROB, 'Vehicle_ownership_PROB')

# Table 3:  Total persons who travelled or did not travel by age group and district of residence (LAU 1)
Travellers_district_age = np.array([
    [  3837,  16040,  10834,  6466], # Southern Harbour
    [  8924,  34928,  21231,  10394], # Northern Harbour
    [  5435,  16366,  12303,  5738], # South Eastern
    [  4465,  12691,  10305,  5919], # Western
    [  6123,  22321,  15408,  6201], # Northern
    [2236,6892,6511,3664] # Gozo and Comino

], dtype=float)

Did_not_travel_district_age = np.array([
    [ 4538,  9790,  9175, 11196],   # Southern Harbour
    [ 5971, 24511, 14378, 16920],   # Northern Harbour
    [ 2797,  9152,  6651,  7174],   # South Eastern
    [ 2042,  7030,  6201,  6167],   # Western
    [ 2802, 11531,  7827,  6773],   # Northern
    [ 1631,  4270,  4032,  4659],   # Gozo and Comino
], dtype=float)

Total_population_district_age = (
    Travellers_district_age + Did_not_travel_district_age
)

Car_access_PROB = np.clip(
    Car_access_district_age / Total_population_district_age,
    0, 1
)

# Table 4: Total persons who travelled or did not travel by gender and district of residence (LAU 1)
Travellers_district_gender = np.array([
    [21262, 15914],   # Southern Harbour  [Male, Female]
    [41748, 33728],   # Northern Harbour
    [22356, 17486],   # South Eastern
    [18523, 14856],   # Western
    [28538, 21516],   # Northern
    [10657,  8647],   # Gozo and Comino
], dtype=float)

# Table 5: Total and average number of trips of persons who travelled by district of residence (LAU 1) and age group
Avg_trips_district_age = np.array([
    [2, 2, 2, 2],   # Southern Harbour
    [2, 3, 3, 3],   # Northern Harbour
    [2, 2, 3, 2],   # South Eastern
    [2, 3, 3, 3],   # Western
    [2, 3, 2, 2],   # Northern
    [2, 2, 3, 2],   # Gozo and Comino
], dtype=float)

# Table 6: Total and average number of trips of persons who travelled by gender and district of residence (LAU 1)
# Columns: [Male avg trips, Female avg trips]
AVG_TRIPS_DIST_GENDER = np.array([
    [2, 2],   # Southern Harbour
    [3, 3],   # Northern Harbour
    [2, 2],   # South Eastern
    [3, 3],   # Western
    [2, 2],   # Northern
    [2, 2],   # Gozo and Comino
], dtype=float)

# Table 7: Number of trips by district of destination (LAU 1) and trip purpose
# Rows: purposes, Columns: districts (same order as District_names)
TRIPS_DEST_PURPOSE = np.array([
    # SH    NH       SE      W       N      G&C
    [9827, 22182,  3329,  3661,  6603,  2414],  # Education+Escort
    [43922, 66821, 18298, 16011, 19654,  9706],  # Commuting
    [6119, 13770,  3234,  4777,  8128,  2735],  # Shopping
    [7540, 10470,  3610,  4217,  5366,  2341],  # Personal Errands
    [30996, 66091, 36141, 31362, 42933, 14142],  # Going Home
    [4714, 8075, 2109, 2856, 3732, 1337], # Escort Other
    [2317, 9213, 1142, 50,  1209, 871],  # Medical
    [6596, 10447, 4927, 5765, 6310, 2675],  # Recreation
    [4827, 8551, 3501, 2986, 3750, 1280],  # Visiting Someone
    [4138, 6814, 2218, 3941, 2964, 2181], # Other
], dtype=float)

# Table 8:  Trips by district of origin (LAU 1) and district of destination (LAU 1)
Trips_distO_and_distD = np.array([
    [42182, 33463, 22855, 10344, 11801,    50],  # Southern Harbour
    [33186,115270, 18215, 24399, 30491,    50],  # Northern Harbour
    [21460, 17104, 31355,  4800,  3761,    50],  # South Eastern
    [ 9829, 22230,  4808, 25630, 13344,    50],  # Western
    [10524, 28066,  3079, 12377, 45393,  1211],  # Northern
    [375, 656, 95, 496, 851, 37209],  # Gozo and Comino
], dtype=float)

# normilize into probabilities
OD_PROB = (
    Trips_distO_and_distD /
    Trips_distO_and_distD.sum(axis=1, keepdims=True)
)
verify_prob(OD_PROB, 'OD_PROB')

# Table 9: Number of trips by district of residence (LAU 1) and main mode of transport
Trips_district_residence_main_mode = np.array([
    [72515,  6032, 6425, 3445],   # Southern Harbour
    [156491, 11531, 19534, 7775],  # Northern Harbour
    [85179,  5992,  4242, 3124],   # South Eastern
    [80366,  2358,  4563, 1909],   # Western
    [104050,  5507,  7019, 3898],  # Northern
    [39871,  2057,  2969, 1603],   # Gozo and Comino
], dtype=float)

# After Table 9
Mode_district_PROB = (
    Trips_district_residence_main_mode /
    Trips_district_residence_main_mode.sum(axis=1, keepdims=True)
)
verify_prob(Mode_district_PROB, 'Mode_district_PROB')

# Table 10: Private vehicle users by age group and gender
# [15-24, 25-44, 45-64, 65+]
PV_gender_age = np.array([
    [11929, 54455, 40722, 18742],   # Male
    [10502, 45012, 28627, 10543],   # Female
], dtype=float)

# Table 11: PV users and trips by labour status
# Users
PV_users_labour = np.array([
    162015, # Employed  
    4265, # Unemployed
    12487, # Student
    30978, # Retired
    50, # Cannot Work
    10043 # Homemaker
], dtype=float)

# Trips
PV_trips_labour = np.array([
    399928, # Employed
    9469, # Unemployed
    26008, # Student
    75322, # Retired
    1405, # Cannot Work
    26340 # Homemaker
], dtype=float)

PV_users_labour_PROB = (
    PV_users_labour /
    PV_users_labour.sum()
)
verify_prob(PV_users_labour_PROB, 'PV_users_labour_PROB')

PV_trips_labour_PROB = (
    PV_trips_labour /
    PV_trips_labour.sum()
)
verify_prob(PV_trips_labour_PROB, 'PV_trips_labour_PROB')

# Table 12:  Private vehicle trips by district of origin (LAU 1) and district of destination (LAU 1)
PV_trips_distO_and_distD = np.array([
    [30794, 28004, 20395,  9232,  9720, 50],  # Southern Harbour
    [27820, 88429, 16844, 22377, 26931, 50],  # Northern Harbour
    [18963, 15561, 25255,  4728,  3570, 50],  # South Eastern
    [8791, 20165,  4629, 21091, 12924, 50],  # Western
    [8388, 25614,  2925, 11674, 36390, 1037], # Northern
    [50, 50, 50, 50, 50, 32822], # Gozo and Comino
], dtype=float)

# Normalize into probabilities
PV_OD_PROB = (
    PV_trips_distO_and_distD /
    PV_trips_distO_and_distD.sum(axis=1, keepdims=True)
)
verify_prob(PV_OD_PROB, 'PV_OD_PROB')

# Table 13: Number of personal vehicle trips by time group
No_PV_timeGRP = np.array([
    0.7, # 00:00 - 02:59
    1.2, # 03:00 - 05:59
    25.7, # 06:00 - 08:59
    18.8, # 09:00 - 11:59
    13.5, # 12:00 - 14:59
    20.0, # 15:00 - 17:59
    16.3, # 18:00 - 20:59
    3.7 # 21:00 - 23:59
], dtype=float)

No_PV_timeGRP_PROB = No_PV_timeGRP / 100 # Divide by sum to convert percentages (sum=100) to proportions (sum=1)
# so numpy can use them as valid sampling probabilities
verify_prob(No_PV_timeGRP_PROB, 'No_PV_timeGRP_PROB')

# Table 14: Number of personal vehicle trips by parking type
No_PV_parking = np.array([
    7.0, # Did not park
    72.3, # Parked on the street
    18.7, # Free
    1.8, # Paid
    0.3  # Other
], dtype=float)

No_PV_parking_PROB = (No_PV_parking / 100)
No_PV_parking_PROB = No_PV_parking_PROB / No_PV_parking_PROB.sum()
verify_prob(No_PV_parking_PROB, 'No_PV_parking_PROB')

# Table 15: Cost of parking
Cost_parking = np.array([
    97.8, # Free 
    0.9, # Less than 2 euros
    0.9, # 2-4 euros
    0.4 # More than 4 euros
], dtype=float)

Cost_parking_PROB = Cost_parking / 100
Cost_parking_PROB /= Cost_parking_PROB.sum()
verify_prob(Cost_parking_PROB, 'Cost_parking_PROB')

# Table 16: Bus users by age group and gender
# [15-24, 25-44, 45-64, 65+]
Bus_users_gender_age = np.array([
    [4215, 2717, 1220, 1834],   # Male
    [4063, 1878, 2193, 2345],   # Female
], dtype=float)

# Table 17: Bus users by nationality
Bus_users_nationality = np.array([
    17931, # Maltese
    2535 # Foreigners
], dtype=float)
Bus_users_nationality_PROB = (Bus_users_nationality /Bus_users_nationality.sum())
verify_prob(Bus_users_nationality_PROB, 'Bus_users_nationality_PROB')

# Table 18: Bus user by ticket type
Bus_users_ticket_type = np.array([
    16.5, # Concession
    13.3, # Free student
    24.1, # Free youth
    40.8, # Standard ticket
    5.3 # Tallinja on demand
], dtype=float) 

Bus_users_ticket_type_PROB = Bus_users_ticket_type / 100
verify_prob(Bus_users_ticket_type_PROB, 'Bus_users_ticket_type_PROB')

# Table 19: bus trips by nationality
Bus_trips_nationality = np.array([
    31282, # Maltese
    3398 # Foreigners
], dtype=float)

Bus_trips_nationality_PROB = (
    Bus_trips_nationality /Bus_trips_nationality.sum()
)
verify_prob(Bus_trips_nationality_PROB, 'Bus_trips_nationality_PROB')

# Table 20: bus trips by purpose
Bus_trips_purpose = np.array([
    14089, # Going Home
    6742, # Goint to work
    5736, # Education
    8112 # Other
], dtype=float)

Bus_trips_purpose_PROB = (
    Bus_trips_purpose /
    Bus_trips_purpose.sum()
)
verify_prob(Bus_trips_purpose_PROB, 'Bus_trips_purpose_PROB')

# # Table 21 — bus trips by district of residence
Bus_trips_district = np.array([
    6032,   # Southern Harbour
    12080,  # Northern Harbour
    6194,   # South Eastern
    2542,   # Western
    5546,   # Northern
    2285,   # Gozo and Comino
], dtype=float)

Bus_trips_district_PROB = (
    Bus_trips_district /
    Bus_trips_district.sum()
)
verify_prob(Bus_trips_district_PROB, 'Bus_trips_district_PROB')

# Table 22: Bus trips by payment type
Bus_payment = np.array([
    4.0, # Cash
    56.2, # Free
    39.8 # Tallinja Card
    ], dtype=float)

# Normalize into probabilities
Bus_payment_PROB = Bus_payment / 100
verify_prob(Bus_payment_PROB, 'Bus_payment_PROB')

# table 23:
Bus_walk_to_stop_avg = {
    "Southern Harbour": 6.4,
    "Northern Harbour": 5.5,
    "South Eastern"   : 4.6,
    "Western"         : 4.0,
    "Northern"        : 6.4,
    "Gozo and Comino" : 4.5,
}

Bus_wait_avg = {
    "Southern Harbour": 12.3,
    "Northern Harbour": 13.3,
    "South Eastern"   : 12.5,
    "Western"         : 12.5,
    "Northern"        : 13.0,
    "Gozo and Comino" : 10.6,
}

# Table 24: Average walking time (minutes) from the bus stop to the destination by district of destination (LAU 1)
Bus_walk_from_stop_avg = {
    "Southern Harbour": 6.8,
    "Northern Harbour": 5.8,
    "South Eastern"   : 5.4,
    "Western"         : 6.0,
    "Northern"        : 6.3,
    "Gozo and Comino" : 5.7,
}

# table 25: Main concerns on public transport by people residing in MALTA by district of residence (LAU 1)
Concerns_public_transport = np.array([
    [24398, 12023, 33268,  8870, 19534, 13570,  8268,  9379],  # Southern Harbour
    [54683, 24917, 66694, 16935, 40507, 30444, 15271, 20347],  # Northern Harbour
    [28486, 16723, 31926,  8493, 20392, 10040,  7349,  6409],  # South Eastern
    [26467, 13462, 27574,  9486, 18345,  9422,  6296,  5353],  # Western
    [36643, 19251, 41042, 13927, 28799, 22888, 12130,  8123],   # Northern
    [11802, 4072, 9671, 5174, 9227, 3343, 2782, 5654],   # Gozo and Comino
], dtype=float)

# Normalize into probabilities
Concerns_public_transport_PROB = (
    Concerns_public_transport /
    Concerns_public_transport.sum(axis=1, keepdims=True)
)
verify_prob(Concerns_public_transport_PROB, 'Concerns_public_transport_PROB')

# Table 26: Main concerns on public transport by people residing in MALTA by gender and age group
# Store as two separate arrays, one per gender
# Columns follow same Bus_concerns order

Bus_concerns_male_age = np.array([
    [12132,  6027, 14727,  3433,  6064,  6561,  3788, 2998],  # 15-24
    [48977, 25269, 53436, 14364, 33848, 21988, 12889, 13291], # 25-44
    [27001, 13934, 29041,  8400, 19960, 10046,  5567,  8725], # 45-64
    [9727, 4228, 12156, 4842, 9734, 4971, 3537,     6903], # 65+
], dtype=float)

Bus_concerns_female_age = np.array([
    [12061,  5317, 14967,  3707,  5354,  7596,  4378, 2046],  # 15-24
    [41617, 19941, 45144, 13787, 33244, 21559, 10923, 8800],  # 25-44
    [22556, 11176, 28075,  8447, 20639, 11031,  6769, 5594],  # 45-64
    [8410, 4557, 12630, 5905, 7961, 5955, 4243, 6907],  # 65+
], dtype=float)

# tables 27-30 were ignored because statistics include Covid 19 percentages
# Tables 31: . Employed persons who teleworked by district of residence (LAU 1)
Teleworkers_district = np.array([
    6666,   # Southern Harbour
    21452,   # Northern Harbour
    6900,   # South Eastern
    6125,   # Western
    11426,   # Northern
    2247,   # Gozo and Comino
], dtype=float)

# Normalize into probabilities
Teleworkers_district_PROB = (
    Teleworkers_district /
    Teleworkers_district.sum()
)
verify_prob(Teleworkers_district_PROB, 'Teleworkers_district_PROB')

# Now, hardcoding of the charts
# Chart 1: Total trips of all means of transport, grouped in the already initialised time bins
Total_trips_time_bin = np.array([
    4699, # 00:00 - 02:59
    7406, # 03:00 - 05:59
    163997, # 06:00 - 08:59
    120230, # 09:00 - 11:59
    86460, # 12:00 - 14:59
    127839, # 15:00 - 17:59
    103946, # 18:00 - 20:59
    23879, # 21:00 - 23:59
], dtype=float)

Total_trips_time_bin_PROB = (
    Total_trips_time_bin /
    Total_trips_time_bin.sum()
)
verify_prob(Total_trips_time_bin_PROB, 'Total_trips_time_bin_PROB')

# Chart 2: Trips by main purpose
Trips_main_purpose = np.array([
    42.2, # Commuting
    11.6, # Education
    4.9, # Escort Education
    9.3, # Shopping
    7.8, # Personal Errands
    3.8, # Medical purposes
    8.9, # Recreation
    6.0, # Visitng someone
    5.5  # Other
], dtype=float)

# Normalize into probabilities
Trips_main_purpose_PROB = (Trips_main_purpose / 100)
verify_prob(Trips_main_purpose_PROB, 'Trips_main_purpose_PROB')

# Chart 3: Trips by age group with personal vehicle
Trips_age_group_with_PV = np.array([
    65.6, # 15-24
    89.2, # 25-44
    88.1, # 45-64
    76.9, # 65+
], dtype=float)

# Normalize into probabilities
Trips_age_group_with_PV_PROB = (Trips_age_group_with_PV / 100)

# Chart 3.1: Trips going home by main mode of transport
Trips_going_home_main_mode = np.array([
    82.8, # Personal Vehicle
    6.1, # Bus
    7.5, # Walking
    0.8, # Organised school transport
    2.7, # Other
], dtype=float)

# Normalize into probabilities
Trips_going_home_main_mode_PROB = (Trips_going_home_main_mode / 100)
verify_prob(Trips_going_home_main_mode_PROB, 'Trips_going_home_main_mode_PROB')

# Chart 3.2: Trips going to main place of work by main mode of transport
Trips_going_work_main_mode = np.array([
    89.0, # Personal Vehicle
    3.7, # Bus
    4.3, # Walking
    3.0, # Other
], dtype=float)

# Normalize into probabilities
Trips_going_work_main_mode_PROB = (Trips_going_work_main_mode / 100)
verify_prob(Trips_going_work_main_mode_PROB, 'Trips_going_work_main_mode_PROB')

# Chart 3.3:Education and escort education trips by main mode of transport
Trips_education_and_escort_main_mode = np.array([
    77.9, # Personal Vehicle
    11.7, # Bus
    3.7, # Walking
    3.9, # Organised school transport
    2.9, # Other
], dtype=float)

# Normalize into probabilities
Trips_education_and_escort_main_mode_PROB = (Trips_education_and_escort_main_mode / 100)
verify_prob(Trips_education_and_escort_main_mode_PROB, 'Trips_education_and_escort_main_mode_PROB')

# Chart 3.4: Shopping and personal errands trips by main mode of transport
Trips_shopping_and_errands_main_mode = np.array([
    79.5, # Personal Vehicle
    4.2, # Bus
    13.2, # Walking
    3.1, # Other
], dtype=float)

# Normalize into probabilities
Trips_shopping_and_errands_main_mode_PROB = (Trips_shopping_and_errands_main_mode / 100)
verify_prob(Trips_shopping_and_errands_main_mode_PROB, 'Trips_shopping_and_errands_main_mode_PROB')

# Chart 3.5: Other trips by main mode of transport
Trips_other_main_mode = np.array([
    85.9, # Personal Vehicle
    4.0, # Bus
    7.5, # Walking
    2.6, # Other
], dtype=float)

# Normalize into probabilities
Trips_other_main_mode_PROB = (Trips_other_main_mode / 100)
verify_prob(Trips_other_main_mode_PROB, 'Trips_other_main_mode_PROB')

# Chart 4:  Trips by gender and main mode of transport
Trip_gender_main_mode = np.array([
     [86.1, 3.2,  8.2, 2.5],   # Male
    [79.0, 6.0, 11.2, 3.8],   # Female
], dtype=float)

# Normalize into probabilities
Trip_gender_main_mode_PROB = Trip_gender_main_mode / 100
verify_prob(Trip_gender_main_mode_PROB, 'Trip_gender_main_mode_PROB')

# Chart 5: Trip stages by main mode of transport
Trip_stages_main_mode = np.array([
    84.0, # Personal Vehicle
    5.6, # Bus
    7.1, # Walking
    0.7, # Ride Sharing
    0.7, # Cycling
    0.4, # Ferry
    1.4, # Other
], dtype=float)

# Normalize into probabilities
Trip_stages_main_mode_PROB = (Trip_stages_main_mode / 100)
verify_prob(Trip_stages_main_mode_PROB, 'Trip_stages_main_mode_PROB')

# Chart 6: Private vehicle trips by age group
PV_trips_age_group = np.array([
    8.9, # 15-24
    45.6, # 25-44
    31.9, # 45-64
    13.6, # 65+
], dtype=float)

# Normalize into probabilities
PV_trips_age_group_PROB = (PV_trips_age_group / 100)
verify_prob(PV_trips_age_group_PROB, 'PV_trips_age_group_PROB')

# Chart 7: Private vehicle trips by gender
PV_trips_gender = np.array([
    56.8, # Male
    43.2, # Female
], dtype=float)

# Normalize into probabilities
PV_trips_gender_PROB = (PV_trips_gender / 100)
verify_prob(PV_trips_gender_PROB, 'PV_trips_gender_PROB')

# Chart 8: Bus trips by age group
Bus_trips_age_group = np.array([
    41.6, # 15-24
    20.7, # 25-44
    16.2, # 45-64
    21.5, # 65+
], dtype=float)

# Normalize into probabilities
Bus_trips_age_group_PROB = (Bus_trips_age_group / 100)
verify_prob(Bus_trips_age_group_PROB, 'Bus_trips_age_group_PROB')

# Chart 9: Bus trips by gender
Bus_trips_gender = np.array([
    46.4, # Male
    53.6, # Female
], dtype=float)

# Normalize into probabilities
Bus_trips_gender_PROB = (Bus_trips_gender / 100)
verify_prob(Bus_trips_gender_PROB, 'Bus_trips_gender_PROB')

# Chart 10: Bus trips by ticket type
Bus_trips_ticket_type = np.array([
    4.9, # Tallinja on demand
    38.7, # Standard Ticket
    23.9, # Free youth
    14.0, # Free student
    18.4, # Concession
], dtype=float)

# Normalize into probabilities
Bus_trips_ticket_type_PROB = (Bus_trips_ticket_type / 100)
verify_prob(Bus_trips_ticket_type_PROB, 'Bus_trips_ticket_type_PROB')

def normalize(arr):
    arr = np.array(arr, dtype=float)
    return arr / arr.sum()

def to_4mode(probs_5):
    pv, bus, walk, school, other = probs_5
    arr = np.array([pv, bus, walk, school + other], dtype=float)
    return arr / arr.sum()



# Conditional probabilities lookups
# building Purpose_given_labour_PROB
# Purpose distribution (Chart 2)
purpose_prob = Trips_main_purpose_PROB  # length 9

# Labour trip intensity (Table 11)
labour_trip_intensity = PV_trips_labour / PV_users_labour  # trips per person
labour_trip_intensity = labour_trip_intensity / labour_trip_intensity.sum()

Purpose_given_labour_PROB = {}

for i, labour in enumerate(Labour_status):
    weighted = purpose_prob * labour_trip_intensity[i]
    Purpose_given_labour_PROB[labour] = weighted / weighted.sum()

# Realism constraints: 
FORBIDDEN = {
    'Retired': ['Commuting', 'Education', 'Escort Education'],
    'Student': ['Commuting'],
    'Homemaker': ['Commuting'],
    'Cannot Work': ['Commuting'],
    'Unemployed': ['Commuting','Education', 'Escort Education'], 
}

# Apply realism constraints
for labour, forbidden_list in FORBIDDEN.items():
    arr = Purpose_given_labour_PROB[labour].copy()
    for purpose in forbidden_list:
        idx = Purpose.index(purpose)
        arr[idx] = 0.0
    # Renormalize after zeroing
    Purpose_given_labour_PROB[labour] = arr / arr.sum()


Mode_given_purpose_PROB = {
    'Commuting': Trips_going_work_main_mode_PROB,
    'Education': to_4mode(Trips_education_and_escort_main_mode_PROB),
    'Escort Education': to_4mode(Trips_education_and_escort_main_mode_PROB),
    'Shopping': Trips_shopping_and_errands_main_mode_PROB,
    'Personal Errands': Trips_shopping_and_errands_main_mode_PROB,
    'Medical Purposes': to_4mode(Trips_going_home_main_mode_PROB),
    'Recreation': to_4mode(Trips_going_home_main_mode_PROB),
    'Visiting Someone': to_4mode(Trips_going_home_main_mode_PROB),
    'Other': to_4mode(Trips_going_home_main_mode_PROB),
}



def sample_predicted_origin(origin_district, rng=None):
    """Pick a locality as the predicted trip origin.
    Only localities within origin_district are eligible, weighted by population."""
    active_rng = rng or RNG
    locs = Region_locality_names[origin_district]
    weights = Origin_weights_by_region[origin_district]
    return active_rng.choice(locs, p=weights)

def sample_predicted_destination(destination_district, purpose, rng=None):
    """Pick a locality as the predicted trip destination.
    Only localities within destination_district are eligible, weighted by POI counts for that purpose."""
    active_rng = rng or RNG
    locs = Region_locality_names[destination_district]
    weights = Destination_weights_by_region[destination_district][purpose]
    return active_rng.choice(locs, p=weights)

def sample_purpose(labour, rng=None):
    active_rng = rng or RNG
    return active_rng.choice(Purpose, p=Purpose_given_labour_PROB[labour])

def sample_mode(purpose, rng=None):
    active_rng = rng or RNG
    return active_rng.choice(Mode_transport_4, p=Mode_given_purpose_PROB[purpose])

sample_time = RNG.choice(Time_bins, p=Total_trips_time_bin_PROB)

def sample_destination(origin_idx, rng=None):
    active_rng = rng or RNG
    return active_rng.choice(District_names, p=OD_PROB[origin_idx])

def sample_labour_status(rng=None):
    active_rng = rng or RNG
    return active_rng.choice(Labour_status, p=PV_users_labour_PROB)

def sample_origin(rng=None):
    active_rng = rng or RNG
    origin_totals = Trips_distO_and_distD.sum(axis=1)
    origin_prob = origin_totals / origin_totals.sum()
    return active_rng.choice(len(District_names), p=origin_prob)

def sample_parking(mode, rng=None):
    active_rng = rng or RNG
    if mode != 'Personal Vehicle':
        return None, None
    parking_type = active_rng.choice(Parking_type_labels, p=No_PV_parking_PROB)
    parking_cost = active_rng.choice(Parking_cost_labels, p=Cost_parking_PROB)
    return parking_type, parking_cost

def sample_bus_ticket(mode, rng=None):
    active_rng = rng or RNG
    if mode != 'Bus':
        return None
    return active_rng.choice(Bus_ticket_labels, p=Bus_users_ticket_type_PROB)

def generate_trip(rng=None):
    active_rng = rng or RNG
    # 1. Labour status
    labour = sample_labour_status(active_rng)

    # 2. Purpose (conditional on labour)
    purpose = sample_purpose(labour, active_rng)

    # 3. Mode (conditional on purpose)
    mode = sample_mode(purpose, active_rng)

    # 4. Time of day (global distribution)
    time_bin = active_rng.choice(Time_bins, p=Total_trips_time_bin_PROB)

    # 5. Origin & destination (district-level, from NSO trip tables)
    origin_idx = sample_origin(active_rng)
    destination = sample_destination(origin_idx, active_rng)
    origin = District_names[origin_idx]

    # 6. Predicted origin & destination (locality-level, from population / POI data)
    # Constrained to only localities within the already-generated district
    predicted_origin = sample_predicted_origin(origin, active_rng)
    predicted_destination = sample_predicted_destination(destination, purpose, active_rng)

    # 7. Parking (if PV)
    parking_type, parking_cost = sample_parking(mode, active_rng)

    # 8. Bus ticket (if Bus)
    bus_ticket = sample_bus_ticket(mode, active_rng)

    return {
        "labour_status": labour,
        "purpose": purpose,
        "mode": mode,
        "time_bin": time_bin,
        "origin": origin,
        "destination": destination,
        "predicted_origin": predicted_origin,
        "predicted_destination": predicted_destination,
        "parking_type": parking_type,
        "parking_cost": parking_cost,
        "bus_ticket": bus_ticket
    }

def generate_trips(n):
    global RNG
    RNG = default_rng(DEFAULT_SEED)
    return pd.DataFrame([generate_trip(RNG) for _ in range(n)])

def generate_trips_with_seed(n, seed=DEFAULT_SEED):
    global RNG
    RNG = default_rng(seed)
    return pd.DataFrame([generate_trip(RNG) for _ in range(n)])

def save_generated_trips(df, output_path=BASELINE_TRIPS_PATH):
    ensure_project_directories()
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path

def generate_and_save_trips(
    n=DEFAULT_TRIP_COUNT,
    seed=DEFAULT_SEED,
    output_path=BASELINE_TRIPS_PATH,
):
    df = generate_trips_with_seed(n, seed=seed)
    saved_path = save_generated_trips(df, output_path=output_path)
    return df, saved_path

def main():
    df, saved_path = generate_and_save_trips()
    print(f"Saved -> {saved_path}")
    print(f"Trips generated: {len(df)}")

if __name__ == "__main__":
    main()