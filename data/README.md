# Data

This folder is intentionally empty in the public repository.

## What belongs here

| File | Description |
|---|---|
| `Fire_points_dataset_final_csv.csv` | Alberta wildfire occurrence records: 3,370 escaped-fire points, 1984–2024, 58 environmental covariates. Columns `FIRE, LONGITUDE, LATITUDE, YEAR, MONTH, DAY`, then 13 reclassified covariates (`slope, elevation, aspect, plan_curvature, profile_curvature, valley_depth, twi, ndvi, avg_temperature, avg_precipitation, avg_windspeed, river, road`) and 45 continuous ones (prior-month and monthly climatological normals). |
| `WildFire_TrainTest.xlsx` | Quebec validation set: 1,042 points, 521 fire and 521 **observed** non-fire, 13 continuous environmental features. Used for the transfer experiment in §6.8. |

## Availability

Available from the corresponding author on request:
Hossein Bonakdari — `hbonakda@uottawa.ca`

The Alberta records derive from provincial fire records harmonised with the
Canadian national fire database; redistribution terms sit with the original
providers, which is why they are not mirrored here.

## Note on the 13 reclassified covariates

These are **not** measurements in physical units. Each is binned into a small
number of ordinal suitability classes and multiplied by a covariate-specific
weight, so the observed values form an exact arithmetic sequence
`ℓ_{j,c} = c · w_j` for `c = 0 … m_j − 1`. Twelve take five levels; `river`
takes nine. This structure is the subject of the paper — it is what makes
interpolated pseudo-absences detectable, and what AP-WGAN's ordinal heads
respect by construction.

The level sets are recoverable exactly from any presence sample with
`numpy.unique`; nothing needs to be estimated.
