r"""
Seed-replicated regeneration of the two non-GAN baselines (Random, Heuristic).

WHY
---
Both baselines are stochastic — Random is uniform sampling, Heuristic uses
probabilistic distance-graded acceptance plus random within-cell selection — so
both have genuine run-to-run variance. The released point sets are single
realisations, which means the main comparison currently scores n=1 baselines
against n=3 GAN arms. Random in particular carries the headline downstream
result (worst on all five classifiers), so it cannot rest on one draw.

This regenerates both under multiple seeds and appends them to the same
manifest.csv that retrain_all.py writes, so evaluate_arms.py picks them up
automatically with no --legacy-dir needed.

FIDELITY TO gen_rand_bp.py
--------------------------
Transcribed from the script that produced the released rand_pts.npy /
heur_pts.npy. Differences from that script are limited to:

  * Water polygons are tested analytically rather than via shapely 60-gon
    approximations. Identical up to the sliver between a 60-gon and its
    circumscribed ellipse, and removes the shapely dependency.
  * Seeding is explicit per run instead of a fixed np.random.seed(42).

Everything else is preserved exactly, including the details that differ from
gen_gan.py and would silently change results if "tidied":

  * bbox padding is 0.3 deg here (gen_gan.py uses 0.2)
  * Heuristic allocates on a 0.5 deg grid via ceil(F/Fmax * N_TARGET)
  * candidate pool is N_TARGET * 25 for Heuristic, N_TARGET * 4 per batch for Random
  * filter order is: distance >= D_MIN, then water, then probabilistic acceptance
  * shortage top-up draws from the accepted pool, then the pool is shuffled

USAGE
    python regen_baselines.py ^
        --data ..\Pseudo-Absence_Sample_Generator\data\Fire_points_dataset_final_csv.csv ^
        --out retrain --seeds 42 43 44

Then re-run the evaluation WITHOUT --legacy-dir, e.g.

    python evaluate_arms.py --data <csv> --retrain retrain --out results ^
        --snap-all --tag _snapped
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

N_TARGET = 5500
DEG = 1.0 / 111.0
D_MIN = 10 * DEG
LAM = 20 * DEG
PAD = 0.3                 # gen_rand_bp.py uses 0.3, NOT gen_gan.py's 0.2
GRID_RES = 0.5
K_IDW = 7
NOISE_FRAC = 0.05

# scoring constants, identical to recompute_spatial.py / retrain_all.py
R_VALS = np.linspace(0.05, 0.80, 15)
BORDER_DEG = 0.15
BBOX_PAD_SCORE = 0.20     # scoring bbox matches the GAN arms so metrics compare

WATER_ELLIPSES = [
    (-115.36, 55.43, 0.69, 0.115),
    (-115.49, 55.78, 0.25, 0.090),
    (-113.19, 55.27, 0.11, 0.095),
    (-113.27, 54.73, 0.10, 0.075),
    (-114.70, 55.33, 0.07, 0.055),
    (-113.10, 55.45, 0.07, 0.055),
]

MANIFEST_COLS = ["label", "arm", "seed", "tau_pctl", "omega", "k",
                 "K_SSE_rawK", "K_SSE_L", "border_fraction", "on_lattice_rate",
                 "wall_clock_s", "pts_file", "finished_utc"]


def in_water(pts):
    m = np.zeros(len(pts), dtype=bool)
    for cx, cy, sx, sy in WATER_ELLIPSES:
        m |= (((pts[:, 0] - cx) / sx) ** 2 + ((pts[:, 1] - cy) / sy) ** 2) <= 1.0
    return m


def idw_feats(pts, tree, fire_feats, rng, k=K_IDW):
    d, ix = tree.query(pts, k=k)
    d = np.maximum(d, 1e-9)
    w = 1.0 / d ** 2
    w /= w.sum(1, keepdims=True)
    f = np.einsum("nk,nkf->nf", w, fire_feats[ix])
    return f + rng.normal(0, NOISE_FRAC * fire_feats.std(0), f.shape)


def gen_random(rng, bounds):
    lo_x, hi_x, lo_y, hi_y = bounds
    pool = []
    while len(pool) < N_TARGET:
        b = np.column_stack([rng.uniform(lo_x, hi_x, N_TARGET * 4),
                             rng.uniform(lo_y, hi_y, N_TARGET * 4)])
        pool.extend(b[~in_water(b)].tolist())
    return np.array(pool[:N_TARGET])


def gen_heuristic(rng, bounds, fire_pts, tree):
    lo_x, hi_x, lo_y, hi_y = bounds
    lon_edges = np.arange(lo_x, hi_x + GRID_RES, GRID_RES)
    lat_edges = np.arange(lo_y, hi_y + GRID_RES, GRID_RES)
    n_lon, n_lat = len(lon_edges) - 1, len(lat_edges) - 1

    fi = np.clip(np.digitize(fire_pts[:, 0], lon_edges) - 1, 0, n_lon - 1)
    fj = np.clip(np.digitize(fire_pts[:, 1], lat_edges) - 1, 0, n_lat - 1)
    fc = np.zeros((n_lon, n_lat), int)
    for ii, jj in zip(fi, fj):
        fc[ii, jj] += 1
    alloc = np.ceil(fc / fc.max() * N_TARGET).astype(int)

    cands = np.column_stack([rng.uniform(lo_x, hi_x, N_TARGET * 25),
                             rng.uniform(lo_y, hi_y, N_TARGET * 25)])
    dists, _ = tree.query(cands, k=1)
    ok, okd = cands[dists >= D_MIN], dists[dists >= D_MIN]
    land = ~in_water(ok)
    ok, okd = ok[land], okd[land]
    p_acc = 1 - np.exp(-(okd - D_MIN) / LAM)
    keep = rng.uniform(0, 1, len(ok)) <= p_acc
    ok = ok[keep]

    gi = np.clip(np.digitize(ok[:, 0], lon_edges) - 1, 0, n_lon - 1)
    gj = np.clip(np.digitize(ok[:, 1], lat_edges) - 1, 0, n_lat - 1)
    pool = []
    for i in range(n_lon):
        for j in range(n_lat):
            ni = alloc[i, j]
            if ni == 0:
                continue
            cell = ok[(gi == i) & (gj == j)]
            if len(cell):
                n_take = min(ni, len(cell))
                pool.extend(cell[rng.choice(len(cell), n_take, replace=False)].tolist())
    shortage = N_TARGET - len(pool)
    if shortage > 0 and len(ok) > shortage:
        pool.extend(ok[rng.choice(len(ok), shortage + 100, replace=False)][:shortage].tolist())
    pool = np.array(pool)
    rng.shuffle(pool)
    return pool[:N_TARGET]


def k_curve(pts, area):
    t = cKDTree(pts)
    n = len(pts)
    return (t.count_neighbors(t, R_VALS) - n) / n / (n / area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.csv")

    df = pd.read_csv(args.data).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    fcols = [c for c in df.columns
             if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    fire_feats = df[fcols].values.astype(float)
    tree = cKDTree(fire)

    gen_bounds = (fire[:, 0].min() - PAD, fire[:, 0].max() + PAD,
                  fire[:, 1].min() - PAD, fire[:, 1].max() + PAD)
    area = ((fire[:, 0].max() - fire[:, 0].min()) *
            (fire[:, 1].max() - fire[:, 1].min()))
    sb = (fire[:, 0].min() - BBOX_PAD_SCORE, fire[:, 0].max() + BBOX_PAD_SCORE,
          fire[:, 1].min() - BBOX_PAD_SCORE, fire[:, 1].max() + BBOX_PAD_SCORE)
    kf = k_curve(fire, area)
    L = lambda k: np.sqrt(np.maximum(k, 0.0) / np.pi)

    existing = set()
    if os.path.exists(manifest):
        m = pd.read_csv(manifest)
        existing = set(zip(m.label, m.seed))

    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        for label, fn in (("random", gen_random), ("heuristic", gen_heuristic)):
            if (label, seed) in existing:
                print(f"  skip {label} seed {seed} (already in manifest)", flush=True)
                continue
            t0 = time.time()
            pts = (fn(rng, gen_bounds) if label == "random"
                   else fn(rng, gen_bounds, fire, tree))
            feats = idw_feats(pts, tree, fire_feats, rng)

            d = os.path.join(args.out, f"{label}_seed{seed}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"pts_{label}.npy"), pts)
            np.save(os.path.join(d, f"feats_{label}.npy"), feats)

            kp = k_curve(pts, area)
            ksse = float(np.sum((kp - kf) ** 2))
            kl = float(np.sum((L(kp) - L(kf)) ** 2))
            b = float((((pts[:, 0] - sb[0]) < BORDER_DEG) | ((sb[1] - pts[:, 0]) < BORDER_DEG) |
                       ((pts[:, 1] - sb[2]) < BORDER_DEG) | ((sb[3] - pts[:, 1]) < BORDER_DEG)).mean())

            with open(os.path.join(d, f"meta_{label}.json"), "w") as fh:
                json.dump({"arm": label, "seed": seed, "n_points": int(len(pts)),
                           "on_lattice_rate": 0.0, "border_fraction": b,
                           "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)

            row = {"label": label, "arm": label, "seed": seed,
                   "tau_pctl": "", "omega": "", "k": "",
                   "K_SSE_rawK": ksse, "K_SSE_L": kl, "border_fraction": b,
                   "on_lattice_rate": 0.0, "wall_clock_s": round(time.time() - t0, 1),
                   "pts_file": os.path.join(d, f"pts_{label}.npy"),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            pd.DataFrame([row], columns=MANIFEST_COLS).to_csv(
                manifest, mode="a", index=False, header=not os.path.exists(manifest))
            print(f"  {label:10s} seed {seed}  n={len(pts)}  K-SSE {ksse:7.4f}  "
                  f"border {b:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    m = pd.read_csv(manifest)
    sub = m[m.label.isin(["random", "heuristic"])]
    if len(sub):
        print("\n=== baselines, mean +- sd over seeds ===", flush=True)
        g = sub.groupby("label").agg(n=("seed", "nunique"),
                                     K_m=("K_SSE_rawK", "mean"), K_sd=("K_SSE_rawK", "std"),
                                     B_m=("border_fraction", "mean"), B_sd=("border_fraction", "std"))
        print(g.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
