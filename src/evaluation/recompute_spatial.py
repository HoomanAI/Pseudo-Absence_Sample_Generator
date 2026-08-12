"""
Authoritative recomputation of all spatial-quality metrics from the released
point sets, for Tables 3, 3b and 6.

WHY
---
Every column of the submitted Table 3 reproduces exactly from
outputs/spatial_results_5methods.csv EXCEPT K-SSE, which is lower in the paper
for all five strategies. Table 6 (the three-arm ablation) has no corresponding
output file in the repository at all. Border fractions for the three ablation
arms DO reproduce to three decimals (0.2773 / 0.2433 / 0.2525 against the
published 0.277 / 0.243 / 0.253), which confirms the arm-to-file mapping used
here is correct.

This script recomputes every spatial metric from the saved .npy point sets so
that the manuscript can quote a single reproducible source of truth.

TWO K-SSE DEFINITIONS ARE REPORTED
----------------------------------
The manuscript's Eq. (1) specifies the variance-stabilised L-transform,
    L(r) = sqrt(K(r)/pi),   K-SSE = sum_k [ L_PA(r_k) - L_fire(r_k) ]^2
but run_spatial_eval.py sums squared differences of the raw K(r) estimates.
The two differ by roughly an order of magnitude, so the equation and the code
as released are inconsistent. Both are computed here under the column names
K_SSE_rawK and K_SSE_L so the manuscript can adopt one and state it
unambiguously. The published values track K_SSE_rawK in ordering and rough
magnitude, so that is treated as the intended estimator.

Ripley's K estimator (as in run_spatial_eval.py, retained for continuity):
    K(r) = mean_i [ #{j != i : d_ij <= r} ] / lambda,   lambda = n / area
    area = tight bounding box of the fire records
No edge correction is applied; the same uncorrected estimator is applied to the
PA set and to the fire reference, and only their difference is interpreted.

USAGE
    python recompute_spatial.py --data <fire csv> --pa-dir <dir of .npy> --out <dir>

OUTPUTS
    spatial_recomputed.csv        all metrics, five strategies + three arms
    spatial_bootstrap.csv         bootstrap 95% CIs vs each baseline
    spatial_published_vs_recomputed.csv   side-by-side discrepancy audit
    spatial_recomputed.json       machine-readable roll-up
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_VALS = np.linspace(0.05, 0.80, 15)   # 15 radii, as in Section 5.1
BORDER_DEG = 0.15                      # border zone width, as in Section 5.1
BBOX_PAD = 0.20                        # padding used by gen_gan.py for the study bbox
GRID_N = 10                            # 10x10 grid for grid variance
N_BOOT = 500                           # matches the paper's spatial bootstrap
SEED = 42

# label -> (points file, published K-SSE, published border fraction)
MAIN = [
    ("Random",     "rand_pts.npy",                6.915, 0.330),
    ("SA",         "sa_pts.npy",                   6.888, 0.498),
    ("Heuristic",  "heur_pts.npy",                 5.729, 0.487),
    ("GAN (clip)", "gan_pts.npy",                  4.872, 0.282),
    ("DA-GP-WGAN", "gan_pts_gp_domain_aware.npy",  3.194, 0.253),
]
ARMS = [
    ("Clip",        "gan_pts_clip.npy",            4.326, 0.277),
    ("GP-standard", "gan_pts_gp_standard.npy",     3.197, 0.243),
    ("DA-GP",       "gan_pts_gp_domain_aware.npy", 3.194, 0.253),
]


def k_curve(pts, area):
    """Ripley's K at each radius in R_VALS.

    Uses cKDTree.count_neighbors, which returns the cumulative pair count for a
    whole array of radii in a single C-level call. This is numerically identical
    to the per-radius query_ball_point loop in run_spatial_eval.py -- both count
    ordered pairs (i, j), i != j, with d_ij <= r -- but is orders of magnitude
    faster, which is what makes the 500-resample bootstrap tractable.
    """
    t = cKDTree(pts)
    n = len(pts)
    lam = n / area
    pairs = t.count_neighbors(t, R_VALS)      # includes the n self-pairs
    return (pairs - n) / n / lam


def sse(a, b):
    return float(np.sum((a - b) ** 2))


def l_transform(k):
    return np.sqrt(np.maximum(k, 0.0) / np.pi)


def centroid_d(p, fire):
    return float(np.sqrt(((p.mean(0) - fire.mean(0)) ** 2).sum()))


def grid_var(pts, n=GRID_N):
    le = np.linspace(pts[:, 0].min(), pts[:, 0].max(), n + 1)
    ae = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n + 1)
    return float(np.var([((pts[:, 0] >= le[i]) & (pts[:, 0] < le[i + 1]) &
                          (pts[:, 1] >= ae[j]) & (pts[:, 1] < ae[j + 1])).sum()
                         for i in range(n) for j in range(n)]))


def nn_d(pts):
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(d[:, 1].mean())


def border_frac(pts, bounds):
    lo_x, hi_x, lo_y, hi_y = bounds
    b = ((pts[:, 0] - lo_x < BORDER_DEG) | (hi_x - pts[:, 0] < BORDER_DEG) |
         (pts[:, 1] - lo_y < BORDER_DEG) | (hi_y - pts[:, 1] < BORDER_DEG))
    return float(b.mean())


def metrics(pts, fire, area, bounds):
    kp, kf = k_curve(pts, area), k_curve(fire, area)
    return {
        "n": int(len(pts)),
        "K_SSE_rawK": sse(kp, kf),
        "K_SSE_L": sse(l_transform(kp), l_transform(kf)),
        "Centroid": centroid_d(pts, fire),
        "Grid_Var": grid_var(pts),
        "Mean_NN": nn_d(pts),
        "Border_Frac": border_frac(pts, bounds),
    }


def boot_ci(pts_a, pts_b, fire, area, bounds, metric, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for metric(A) - metric(B), resampling points with replacement."""
    rng = np.random.default_rng(seed)
    kf = k_curve(fire, area)

    def one(p):
        if metric == "K_SSE_rawK":
            return sse(k_curve(p, area), kf)
        if metric == "K_SSE_L":
            return sse(l_transform(k_curve(p, area)), l_transform(kf))
        if metric == "Centroid":
            return centroid_d(p, fire)
        if metric == "Grid_Var":
            return grid_var(p)
        if metric == "Mean_NN":
            return nn_d(p)
        if metric == "Border_Frac":
            return border_frac(p, bounds)
        raise ValueError(metric)

    d = np.empty(n_boot)
    for b in range(n_boot):
        ia = rng.integers(0, len(pts_a), len(pts_a))
        ib = rng.integers(0, len(pts_b), len(pts_b))
        d[b] = one(pts_a[ia]) - one(pts_b[ib])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pa-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot-metrics", default="K_SSE_rawK,Border_Frac",
                    help="comma-separated metrics to bootstrap (each is O(n_boot) K-curves)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    area = ((fire[:, 0].max() - fire[:, 0].min()) *
            (fire[:, 1].max() - fire[:, 1].min()))
    bounds = (fire[:, 0].min() - BBOX_PAD, fire[:, 0].max() + BBOX_PAD,
              fire[:, 1].min() - BBOX_PAD, fire[:, 1].max() + BBOX_PAD)
    print(f"fire n={len(fire)}  area={area:.4f} deg^2", flush=True)

    # fire reference values (for the closeness-to-reference metrics)
    ref = {"label": "Fire (ref.)", "table": "reference", "n": len(fire),
           "K_SSE_rawK": 0.0, "K_SSE_L": 0.0, "Centroid": 0.0,
           "Grid_Var": grid_var(fire), "Mean_NN": nn_d(fire),
           "Border_Frac": border_frac(fire, bounds),
           "published_K_SSE": np.nan, "published_Border_Frac": np.nan}

    rows = [ref]
    pts_cache = {}
    for table, spec in (("table3", MAIN), ("table6", ARMS)):
        for label, f, pub_k, pub_b in spec:
            p = np.load(os.path.join(args.pa_dir, f))
            pts_cache[(table, label)] = p
            m = metrics(p, fire, area, bounds)
            m.update({"label": label, "table": table,
                      "published_K_SSE": pub_k, "published_Border_Frac": pub_b})
            rows.append(m)
            print(f"[{table}] {label:12s} K_SSE_rawK={m['K_SSE_rawK']:8.4f} "
                  f"(published {pub_k:6.3f})  K_SSE_L={m['K_SSE_L']:7.4f}  "
                  f"border={m['Border_Frac']:.4f} (published {pub_b:.3f})", flush=True)

    cols = ["table", "label", "n", "K_SSE_rawK", "K_SSE_L", "Centroid",
            "Grid_Var", "Mean_NN", "Border_Frac",
            "published_K_SSE", "published_Border_Frac"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(os.path.join(args.out, "spatial_recomputed.csv"), index=False)

    # discrepancy audit
    aud = out[out.table != "reference"].copy()
    aud["K_SSE_abs_diff"] = aud.K_SSE_rawK - aud.published_K_SSE
    aud["K_SSE_ratio_pub_over_recomputed"] = aud.published_K_SSE / aud.K_SSE_rawK
    aud["Border_abs_diff"] = aud.Border_Frac - aud.published_Border_Frac
    aud[["table", "label", "K_SSE_rawK", "published_K_SSE", "K_SSE_abs_diff",
         "K_SSE_ratio_pub_over_recomputed", "Border_Frac", "published_Border_Frac",
         "Border_abs_diff"]].to_csv(
        os.path.join(args.out, "spatial_published_vs_recomputed.csv"), index=False)

    # bootstrap: DA-GP-WGAN vs each Table-3 baseline, and each ablation contrast
    boot_metrics = [m.strip() for m in args.boot_metrics.split(",") if m.strip()]
    boot = []
    contrasts = [("table3", b, "DA-GP-WGAN") for b, *_ in
                 [x for x in MAIN if x[0] != "DA-GP-WGAN"]]
    contrasts += [("table6", "Clip", "GP-standard"),
                  ("table6", "Clip", "DA-GP"),
                  ("table6", "GP-standard", "DA-GP")]
    for table, A, B in contrasts:
        pa, pb = pts_cache[(table, A)], pts_cache[(table, B)]
        for met in boot_metrics:
            dm, lo, hi = boot_ci(pa, pb, fire, area, bounds, met, n_boot=args.n_boot)
            boot.append({"table": table, "comparison": f"{A} - {B}", "metric": met,
                         "delta": dm, "ci_lo": lo, "ci_hi": hi,
                         "decision": "Significant" if (lo > 0 or hi < 0) else "NS"})
            print(f"  boot [{table}] {A} - {B:12s} {met:12s} "
                  f"D={dm:+8.4f} CI=[{lo:+8.4f},{hi:+8.4f}] "
                  f"{'Sig' if (lo>0 or hi<0) else 'NS'}", flush=True)
    pd.DataFrame(boot).to_csv(os.path.join(args.out, "spatial_bootstrap.csv"), index=False)

    with open(os.path.join(args.out, "spatial_recomputed.json"), "w") as fh:
        json.dump({"metrics": rows, "bootstrap": boot,
                   "config": {"R_VALS": R_VALS.tolist(), "BORDER_DEG": BORDER_DEG,
                              "BBOX_PAD": BBOX_PAD, "GRID_N": GRID_N,
                              "N_BOOT": args.n_boot, "SEED": SEED,
                              "area_deg2": area}}, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
