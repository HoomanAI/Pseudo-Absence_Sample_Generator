"""
Aggregate the 5-seed v1_da_gp study: K-SSE (raw K) and border fraction per
seed, using recompute_spatial.py's own functions (imported, not reimplemented)
so the methodology is identical to Tables 3/6.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:/Working Docs/Papers/NonFire Points/Revision/Revision Code")
import recompute_spatial as rs

DATA = r"E:/Working Docs/Papers/NonFire Points/Revision/Pseudo-Absence_Sample_Generator/data/Fire_points_dataset_final_csv.csv"
SEED_DIR = r"E:/Working Docs/Papers/NonFire Points/Revision/Revision Code/seed_study"
SEEDS = [42, 43, 44, 45, 46]
OUT = r"E:/Working Docs/Papers/NonFire Points/Revision/Revision Code/results/seed_study_summary.csv"

EXPECTED_KSSE = 3.8235
EXPECTED_BORDER = 0.2525


def main():
    df = pd.read_csv(DATA).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    area = ((fire[:, 0].max() - fire[:, 0].min()) *
            (fire[:, 1].max() - fire[:, 1].min()))
    bounds = (fire[:, 0].min() - rs.BBOX_PAD, fire[:, 0].max() + rs.BBOX_PAD,
              fire[:, 1].min() - rs.BBOX_PAD, fire[:, 1].max() + rs.BBOX_PAD)
    kf = rs.k_curve(fire, area)

    rows = []
    for seed in SEEDS:
        pts = np.load(f"{SEED_DIR}/seed_{seed}/pts_v1_da_gp.npy")
        kp = rs.k_curve(pts, area)
        ksse = rs.sse(kp, kf)
        bf = rs.border_frac(pts, bounds)
        rows.append({"seed": seed, "n": len(pts), "K_SSE_rawK": ksse, "Border_Frac": bf})
        print(f"seed={seed}  n={len(pts)}  K_SSE_rawK={ksse:.4f}  Border_Frac={bf:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    for col, expected in [("K_SSE_rawK", EXPECTED_KSSE), ("Border_Frac", EXPECTED_BORDER)]:
        vals = out[col].values
        mean, sd, mn, mx = vals.mean(), vals.std(ddof=1), vals.min(), vals.max()
        inside = mn <= expected <= mx
        print(f"\n{col}: mean={mean:.4f} sd={sd:.4f} min={mn:.4f} max={mx:.4f}")
        print(f"  canary/original value {expected} {'INSIDE' if inside else 'OUTSIDE'} the 5-seed range [{mn:.4f}, {mx:.4f}]")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
