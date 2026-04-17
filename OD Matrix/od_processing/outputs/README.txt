OD input preparation outputs
============================

Points rows kept: 13,329
Polygon rows kept: 10,089
Combined rows before dedup: 23,418
Combined rows after dedup: 23,187

Deduplication rule:
- Only named features are deduplicated.
- Duplicate key = locality + trip_category + normalized name.
- Source preference by category:
  * polygon preferred for Education, Medical Purpose, Visiting Someone
  * point preferred for Shopping, Personal Errands, Recreation, Commuting to work
- Unnamed features are kept as-is.

Main file to use next:
- locality_attractiveness_matrix.csv

This matrix can now be used as destination attractiveness A_jk in the OD model.
