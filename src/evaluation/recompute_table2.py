"""
Recompute Table 2 (feature discriminability) without Simulated Annealing.

Table 2 ranks the top-15 of 58 environmental features by
Total|bias| = sum over methods of |(PA mean - fire mean) / fire std|.
The published table summed this over 5 methods (Heuristic, Random, SA, GAN,
DA-GP-WGAN). SA is excluded from this revision's comparison entirely, so the
sum -- and therefore the ranking and the top-15 selection itself -- must be
recomputed over the remaining 4 methods, not just have its SA column deleted.

Reuses the same fire-feature-loading logic as admissibility_test.py
(the shared, already-verified source of the feature column list).
"""
import numpy as np
import pandas as pd

DATA = r"E:/Working Docs/Papers/NonFire Points/Revision/Pseudo-Absence_Sample_Generator/data/Fire_points_dataset_final_csv.csv"
PA_DIR = r"E:/Working Docs/uOttawa/Semesters/Summer 2026/AI/Project/Presentation/cWGAN codes"
OUT = r"E:/Working Docs/Papers/NonFire Points/Revision/Revision Code/results/table2_recomputed.csv"

METHODS = {
    "Heuristic":   "heur_feats.npy",
    "Random":      "rand_feats.npy",
    "GAN":         "gan_feats.npy",              # clip arm
    "DA-GP-WGAN":  "gan_feats_gp_domain_aware.npy",
}

FEATURE_LABELS = {
    "river": "Distance to nearest river (km)",
    "road": "Distance to nearest road (km)",
    "b12_monthly_avg_temp": "Long-term mean temperature: December (\\textdegree C)",
    "b2_monthly_avg_temp": "Long-term mean temperature: February (\\textdegree C)",
    "b1_monthly_avg_temp": "Long-term mean temperature: January (\\textdegree C)",
    "b3_monthly_avg_temp": "Long-term mean temperature: March (\\textdegree C)",
    "b5_monthly_avg_precipitation": "Long-term mean precipitation: May (mm/month)",
    "b4_monthly_avg_precipitation": "Long-term mean precipitation: April (mm/month)",
    "b11_monthly_avg_temp": "Long-term mean temperature: November (\\textdegree C)",
    "b8_monthly_avg_precipitation": "Long-term mean precipitation: August (mm/month)",
    "b11_monthly_avg_precipitation": "Long-term mean precipitation: November (mm/month)",
    "b4_monthly_avg_temp": "Long-term mean temperature: April (\\textdegree C)",
    "temp_2m_prior": "Mean temperature: 2 months prior to fire (\\textdegree C)",
    "temp_3m_prior": "Mean temperature: 3 months prior to fire (\\textdegree C)",
    "precip_3m_prior": "Cumulative precipitation: 3 months prior (mm)",
    "b9_monthly_avg_precipitation": "Long-term mean precipitation: September (mm/month)",
    "precip_1m_prior": "Cumulative precipitation: 1 month prior (mm)",
    "precip_2m_prior": "Cumulative precipitation: 2 months prior (mm)",
    "wind_1m_prior": "Mean wind speed: 1 month prior to fire (m/s)",
    "wind_2m_prior": "Mean wind speed: 2 months prior to fire (m/s)",
    "wind_3m_prior": "Mean wind speed: 3 months prior to fire (m/s)",
    "slope": "Slope (degrees)",
    "elevation": "Elevation (m)",
    "aspect": "Aspect (degrees)",
    "plan_curvature": "Plan curvature",
    "profile_curvature": "Profile curvature",
    "valley_depth": "Valley depth (m)",
    "twi": "Topographic wetness index",
    "ndvi": "NDVI",
    "avg_temperature": "Day-of-event temperature (\\textdegree C)",
    "avg_precipitation": "Day-of-event precipitation (mm)",
    "avg_windspeed": "Day-of-event wind speed (m/s)",
}
for i in range(1, 13):
    FEATURE_LABELS.setdefault(f"b{i}_monthly_avg_temp", f"Long-term mean temperature: b{i} (\\textdegree C)")
    FEATURE_LABELS.setdefault(f"b{i}_monthly_avg_precipitation", f"Long-term mean precipitation: b{i} (mm/month)")
    FEATURE_LABELS.setdefault(f"b{i}_monthly_avg_wind", f"Long-term mean wind speed: b{i} (m/s)")


def main():
    df = pd.read_csv(DATA).dropna()
    fc = [c for c in df.columns if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    fire = df[fc].values.astype(float)
    fire_mean = fire.mean(axis=0)
    fire_std = fire.std(axis=0)
    fire_std_safe = np.where(fire_std == 0, 1.0, fire_std)

    bias = {}
    for name, fname in METHODS.items():
        pa = np.load(f"{PA_DIR}/{fname}")
        assert pa.shape[1] == len(fc), f"{name}: {pa.shape[1]} cols, expected {len(fc)}"
        pa_mean = pa.mean(axis=0)
        bias[name] = np.abs((pa_mean - fire_mean) / fire_std_safe)

    rows = []
    for j, col in enumerate(fc):
        row = {"feature": col}
        total = 0.0
        for name in METHODS:
            v = bias[name][j]
            row[name] = v
            total += v
        row["total_bias"] = total
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("total_bias", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.to_csv(OUT, index=False)

    top15 = out.head(15)
    print(top15.to_string(index=False))
    print(f"\nwritten -> {OUT}")

    print("\n--- LaTeX rows (top 15, no SA) ---")
    for _, r in top15.iterrows():
        label = FEATURE_LABELS.get(r["feature"], r["feature"])
        print(f"{int(r['rank']):<2} & {label:<55} & {r['total_bias']:.3f} & "
              f"{r['Heuristic']:.3f} & {r['Random']:.3f} & {r['GAN']:.3f} & {r['DA-GP-WGAN']:.3f} \\\\")


if __name__ == "__main__":
    main()
