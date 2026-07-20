Place `Fire_points_dataset_final_csv.csv` in this folder before running any script in `src/`.

Expected columns: `FIRE, LONGITUDE, LATITUDE, YEAR, MONTH, DAY` plus 58 environmental feature
columns (slope, elevation, climate normals, monthly bands, etc.). Scripts drop rows with any
missing values and treat every column outside the six listed above as a feature.
