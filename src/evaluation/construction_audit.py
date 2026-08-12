"""
Construction audit + matched controls for the Alberta pseudo-absence experiment.

WHY THIS SCRIPT EXISTS
----------------------
Section 7.1 of the submitted manuscript attributes the RF/XGBoost AUC ~= 1.000
ceiling to a "local-geometry fingerprint" of the shared inverse-distance-weighted
(IDW) pseudo-absence feature assignment, reporting a geometry-only AUC of
0.979-0.999. Auditing the released dataset shows that diagnosis is incomplete
and that the true mechanism is both simpler and stronger.

FINDING 1 -- the 13 core environmental features are ORDINAL CLASS CODES.
In Fire_points_dataset_final_csv.csv the 13 "core" predictors
(slope, elevation, aspect, plan/profile curvature, valley_depth, twi, ndvi,
avg_temperature, avg_precipitation, avg_windspeed, river, road) take only 5
distinct values each (9 for river) across all 3,370 fire records. They are
reclassified, weight-scaled susceptibility classes, not continuous
measurements. The remaining 45 columns (prior-month and monthly climatology)
are continuous.

FINDING 2 -- IDW moves pseudo-absences OFF that lattice, 100% of the time.
Because PA features are inverse-distance-weighted averages of neighbouring
fire records plus Gaussian noise, they land on an allowed class level in
0.000 of cases, while fire records land on one by construction in 1.000 of
cases. A single scalar -- the summed absolute deviation from the nearest
allowed level, carrying zero environmental information -- separates the two
classes at AUC = 1.00000 for every one of the five PA strategies. This
exactly and completely accounts for XGBoost reaching AUC = 1.00000 with zero
variance in every fold of every strategy.

The local-geometry fingerprint reported in Section 7.1 is real but is a
downstream symptom of the same IDW smoothing, not the primary mechanism.

CONDITIONS EVALUATED
--------------------
  baseline     as published: fire = observed values, PA = IDW + noise
  lattice      PA's 13 ordinal dimensions snapped to the nearest allowed
               class level, so both classes occupy the identical lattice
  lattice_bw   lattice snapping, PLUS bandwidth-matched leave-out IDW applied
               to the fire side on the 45 continuous dimensions

Bandwidth matching matters because the two classes are IDW-smoothed at very
different spatial lags:
    FIRE, leave-one-out :  mean 7-NN lag  0.0248 deg
    PA  -> fire         :  mean 7-NN lag  0.2744 - 0.4133 deg   (11x - 17x)
For each fire point a lag rho_i is drawn from the empirical distribution of
that strategy's PA->fire nearest-neighbour distances, every fire point within
rho_i is excluded, and the 7 nearest survivors are interpolated over. Both
classes are then IDW estimates over 7 fire records at statistically identical
spatial lags.

USAGE
-----
python construction_audit.py \
    --data   .../Fire_points_dataset_final_csv.csv \
    --pa-dir .../cWGAN codes \
    --out    .../audit_out

OUTPUTS
-------
  audit_discretization.csv  unique-level counts and on-lattice rates per feature
  audit_tells.csv           AUC of each zero-environment "tell" per condition
  audit_bandwidth.csv       realised interpolation lags per class and condition
  audit_ml.csv              4-model / 5-fold spatial-block CV per condition
  audit_bootstrap.csv       bootstrap dAUC CIs per condition
  audit_summary.json        machine-readable roll-up of everything above
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

K_IDW = 7            # matches interpolate_features() in pseudo_absence_pipeline.py
NOISE_FRAC = 0.05    # matches interpolate_features()
K_GEOM = 7           # neighbourhood size for local-geometry meta-features
DISC_MAX_LEVELS = 12 # a feature with <= this many distinct values is treated as ordinal
N_FOLDS = 5
SEED = 42
CONDITIONS = ["baseline", "lattice", "lattice_bw"]

STRATEGIES = [
    ("Heuristic",  "heur_pts.npy",                "heur_feats.npy"),
    ("Random",     "rand_pts.npy",                "rand_feats.npy"),
    ("SA",         "sa_pts.npy",                  "sa_feats.npy"),
    ("GAN (clip)", "gan_pts.npy",                 "gan_feats.npy"),
    ("DA-GP-WGAN", "gan_pts_gp_domain_aware.npy", "gan_feats_gp_domain_aware.npy"),
]


# ---------------------------------------------------------------------------
# spatial block CV -- procedure identical to run_ml_v2.py / run_ml_da_gp_wgan.py
# ---------------------------------------------------------------------------
def sfolds(coords, n=N_FOLDS):
    lat = pd.qcut(coords[:, 1], n, labels=False, duplicates="drop")
    lon = pd.qcut(coords[:, 0], n, labels=False, duplicates="drop")
    bl = lat * n + lon
    u = np.unique(bl)
    np.random.shuffle(u)
    fid = np.zeros(len(coords), int)
    for fi, gr in enumerate(np.array_split(u, n)):
        for b in gr:
            fid[bl == b] = fi
    return fid


def models(subset=None):
    m = {
        "RandomForest": RandomForestClassifier(100, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, objective="binary:logistic",
                                 eval_metric="logloss", verbosity=0, random_state=SEED),
        "KNN": KNeighborsClassifier(9, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    }
    return m if subset is None else {k: v for k, v in m.items() if k in subset}


def run_cv(X, y, coords, subset=None):
    fid = sfolds(coords)
    out = {}
    for nm, clf in models(subset).items():
        aucs, tsss = [], []
        for fold in range(N_FOLDS):
            tr, te = fid != fold, fid == fold
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            if nm == "SVM":
                i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
                s = np.concatenate([np.random.choice(i0, min(500, len(i0)), False),
                                    np.random.choice(i1, min(500, len(i1)), False)])
                Xtr, ytr = Xtr[s], ytr[s]
            if nm in ("SVM", "KNN"):
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, p))
            pred = (p >= 0.5).astype(int)
            tp = ((pred == 1) & (yte == 1)).sum(); tn = ((pred == 0) & (yte == 0)).sum()
            fp = ((pred == 1) & (yte == 0)).sum(); fn = ((pred == 0) & (yte == 1)).sum()
            tsss.append(tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9) - 1)
        out[nm] = {"AUC": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
                   "TSS": float(np.mean(tsss)), "TSS_std": float(np.std(tsss)),
                   "folds": [float(a) for a in aucs]}
    return out


# ---------------------------------------------------------------------------
# the two zero-environment "tells"
# ---------------------------------------------------------------------------
def lattice_dev(A, levels):
    """Per-ordinal-dimension absolute deviation from the nearest allowed level."""
    return np.column_stack([np.min(np.abs(A[:, [j]] - levels[j][None, :]), axis=1)
                            for j in sorted(levels)])


def geometry_meta(X, k=K_GEOM):
    """m1 = distance to own k-NN centroid, m2 = mean k-NN distance,
    m3 = std of k-NN distances. Standardized feature space, self excluded."""
    Z = StandardScaler().fit_transform(X)
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    d, idx = d[:, 1:], idx[:, 1:]
    m1 = np.linalg.norm(Z - Z[idx].mean(axis=1), axis=1)
    return np.column_stack([m1, d.mean(axis=1), d.std(axis=1)])


# ---------------------------------------------------------------------------
# matched constructions
# ---------------------------------------------------------------------------
def snap_to_lattice(A, levels):
    """Snap the ordinal dimensions of A onto the nearest allowed class level."""
    B = A.copy()
    for j, lv in levels.items():
        B[:, j] = lv[np.argmin(np.abs(A[:, [j]] - lv[None, :]), axis=1)]
    return B


def _idw(feats_src, idx, dist, noise_scale, rng):
    dist = np.maximum(dist, 1e-9)
    w = 1.0 / (dist ** 2)
    w /= w.sum(axis=1, keepdims=True)
    out = np.einsum("nk,nkf->nf", w, feats_src[idx])
    return out + rng.normal(0.0, np.where(noise_scale > 0, noise_scale, 1e-12), out.shape)


def bw_matched_fire(fire_pts, fire_feats, pa_lags, cont_idx, k=K_IDW, rng=None,
                    chunk_size=256):
    """Bandwidth-matched leave-out IDW on the CONTINUOUS dimensions only.

    Ordinal dimensions are left at their observed class levels, because
    interpolating them would push the fire class off the shared lattice and
    simply re-introduce the tell with the sign reversed.
    """
    rng = rng or np.random.default_rng(SEED)
    n = len(fire_pts)
    rho = rng.choice(np.asarray(pa_lags), size=n, replace=True)

    idx = np.empty((n, k), dtype=int)
    dist = np.empty((n, k), dtype=float)
    # Chunking is numerically identical to the original full n x n matrix,
    # while bounding peak distance/sort storage at chunk_size x n.
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        D = np.sqrt(((fire_pts[start:stop, None, :] -
                      fire_pts[None, :, :]) ** 2).sum(-1))
        order = np.argsort(D, axis=1)
        Ds = np.take_along_axis(D, order, axis=1)
        for q, i in enumerate(range(start, stop)):
            ok = np.nonzero(Ds[q] >= rho[i])[0]
            if len(ok) < k:
                ok = np.arange(len(Ds[q]) - k, len(Ds[q]))
            sel = ok[:k]
            idx[i] = order[q, sel]
            dist[i] = Ds[q, sel]

    out = fire_feats.copy()
    sub = fire_feats[:, cont_idx]
    out[:, cont_idx] = _idw(sub, idx, dist, NOISE_FRAC * sub.std(axis=0), rng)
    return out, dist


def assert_bw_chunk_equivalence(fire_pts, fire_feats, pa_lags, cont_idx,
                                n_check=500, seed=SEED):
    """Assert chunked selection/output equals the former full-matrix algorithm."""
    n = min(n_check, len(fire_pts)); P = fire_pts[:n]; F = fire_feats[:n]
    lags = np.asarray(pa_lags); rng0 = np.random.default_rng(seed)
    rho = rng0.choice(lags, size=n, replace=True)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    order = np.argsort(D, axis=1); Ds = np.take_along_axis(D, order, axis=1)
    idx=np.empty((n,K_IDW),int); dist=np.empty((n,K_IDW),float)
    for i in range(n):
        ok=np.nonzero(Ds[i]>=rho[i])[0]
        if len(ok)<K_IDW: ok=np.arange(len(Ds[i])-K_IDW,len(Ds[i]))
        sel=ok[:K_IDW]; idx[i]=order[i,sel]; dist[i]=Ds[i,sel]
    old=F.copy(); sub=F[:,cont_idx]
    old[:,cont_idx]=_idw(sub,idx,dist,NOISE_FRAC*sub.std(0),rng0)
    new,newdist=bw_matched_fire(P,F,lags,cont_idx,rng=np.random.default_rng(seed),chunk_size=73)
    assert np.array_equal(dist,newdist), np.max(np.abs(dist-newdist))
    assert np.array_equal(old,new), np.max(np.abs(old-new))
    return {"n":n,"max_abs_feature_diff":float(np.max(np.abs(old-new))),
            "max_abs_lag_diff":float(np.max(np.abs(dist-newdist)))}


def bdelta(a, b, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(n)])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pa-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    fire_pts = df[["LONGITUDE", "LATITUDE"]].values
    fcols = [c for c in df.columns
             if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    fire_feats = df[fcols].values.astype(float)
    fire_tree = cKDTree(fire_pts)
    n_f = len(fire_pts)

    # ---- FINDING 1: which dimensions are ordinal class codes? ----
    nun = {c: int(df[c].nunique()) for c in fcols}
    disc_idx = [j for j, c in enumerate(fcols) if nun[c] <= DISC_MAX_LEVELS]
    cont_idx = [j for j in range(len(fcols)) if j not in disc_idx]
    levels = {j: np.unique(fire_feats[:, j]) for j in disc_idx}
    print(f"fire points {n_f}  features {len(fcols)}  "
          f"ordinal {len(disc_idx)}  continuous {len(cont_idx)}", flush=True)

    disc_rows = [{"feature": c, "n_unique": nun[c],
                  "type": "ordinal" if j in disc_idx else "continuous"}
                 for j, c in enumerate(fcols)]

    geom_rows, tell_rows, ml_rows, bw_rows = [], [], [], []
    folds = {c: {} for c in CONDITIONS}

    for label, pf, ff in STRATEGIES:
        pa_pts = np.load(os.path.join(args.pa_dir, pf))
        pa_feats_raw = np.load(os.path.join(args.pa_dir, ff))
        coords = np.vstack([fire_pts, pa_pts])
        y = np.array([1] * n_f + [0] * len(pa_pts))

        pa_lag = fire_tree.query(pa_pts, k=K_IDW)[0]
        pa_snapped = snap_to_lattice(pa_feats_raw, levels)
        fire_bw, bw_lag = bw_matched_fire(fire_pts, fire_feats, pa_lag[:, 0], cont_idx)

        variants = {
            "baseline":   (fire_feats, pa_feats_raw, None),
            "lattice":    (fire_feats, pa_snapped,   None),
            "lattice_bw": (fire_bw,    pa_snapped,   bw_lag),
        }

        # on-lattice rate, reported once per strategy
        for cond in CONDITIONS:
            _, pa_v, _ = variants[cond]
            rate = float(np.mean([
                np.all([np.min(np.abs(pa_v[i, j] - levels[j])) < 1e-9 for j in disc_idx])
                for i in range(0, len(pa_v), 10)]))
            disc_rows.append({"feature": f"__on_lattice_rate__{label}__{cond}",
                              "n_unique": -1, "type": f"{rate:.4f}"})

        for cond in CONDITIONS:
            f_v, pa_v, flag = variants[cond]
            X = np.vstack([f_v, pa_v])

            bw_rows.append({
                "strategy": label, "condition": cond,
                "pa_lag_mean_deg": float(pa_lag.mean()),
                "fire_lag_mean_deg": float(flag.mean()) if flag is not None else np.nan,
                "lag_ratio_pa_over_fire": (float(pa_lag.mean() / flag.mean())
                                           if flag is not None else np.nan)})

            # --- tell 1: lattice deviation (zero environmental content) ---
            L = lattice_dev(X, levels)
            np.random.seed(SEED)
            tl = run_cv(L, y, coords, subset=["RandomForest"])
            auc_scalar = roc_auc_score(1 - y, L.sum(axis=1))

            # --- tell 2: local-geometry meta-features ---
            M = geometry_meta(X)
            m1f, m1p = M[:n_f, 0], M[n_f:, 0]
            np.random.seed(SEED)
            gm = run_cv(M, y, coords, subset=["RandomForest"])

            tell_rows.append({
                "strategy": label, "condition": cond,
                "lattice_tell_AUC": tl["RandomForest"]["AUC"],
                "lattice_tell_scalar_AUC": float(auc_scalar),
                "geometry_tell_AUC": gm["RandomForest"]["AUC"],
                "m1_fire_median": float(np.median(m1f)),
                "m1_pa_median": float(np.median(m1p)),
                "m1_fire_over_pa": float(np.median(m1f) / np.median(m1p))})
            print(f"[{cond:10s}] {label:12s} lattice-tell AUC={tl['RandomForest']['AUC']:.5f} "
                  f"(scalar {auc_scalar:.5f})  geom-tell AUC={gm['RandomForest']['AUC']:.5f}  "
                  f"m1 ratio={np.median(m1f)/np.median(m1p):5.2f}x", flush=True)

            # --- full 4-model CV ---
            np.random.seed(SEED)
            r = run_cv(X, y, coords)
            folds[cond][label] = {m: v["folds"] for m, v in r.items()}
            for m, v in r.items():
                ml_rows.append({"strategy": label, "condition": cond, "model": m,
                                "AUC": v["AUC"], "AUC_std": v["AUC_std"],
                                "TSS": v["TSS"], "TSS_std": v["TSS_std"]})
                print(f"    {cond:10s} {label:12s} {m:13s} AUC={v['AUC']:.5f} "
                      f"+-{v['AUC_std']:.5f} TSS={v['TSS']:.3f}", flush=True)

    pd.DataFrame(disc_rows).to_csv(os.path.join(args.out, "audit_discretization.csv"), index=False)
    pd.DataFrame(tell_rows).to_csv(os.path.join(args.out, "audit_tells.csv"), index=False)
    pd.DataFrame(bw_rows).to_csv(os.path.join(args.out, "audit_bandwidth.csv"), index=False)
    pd.DataFrame(ml_rows).to_csv(os.path.join(args.out, "audit_ml.csv"), index=False)

    pairs = [("DA-GP-WGAN", "Random"), ("DA-GP-WGAN", "Heuristic"),
             ("DA-GP-WGAN", "GAN (clip)"), ("DA-GP-WGAN", "SA"),
             ("Heuristic", "Random"), ("GAN (clip)", "Random")]
    boot = []
    for cond in CONDITIONS:
        for A, B in pairs:
            for m in ["RandomForest", "XGBoost", "KNN", "SVM"]:
                dm, lo, hi = bdelta(folds[cond][A][m], folds[cond][B][m])
                boot.append({"condition": cond, "comparison": f"{A} - {B}", "model": m,
                             "dAUC": dm, "ci_lo": lo, "ci_hi": hi,
                             "decision": "Significant" if (lo > 0 or hi < 0) else "NS"})
    pd.DataFrame(boot).to_csv(os.path.join(args.out, "audit_bootstrap.csv"), index=False)

    with open(os.path.join(args.out, "audit_summary.json"), "w") as fh:
        json.dump({"discretization": disc_rows, "tells": tell_rows, "bandwidth": bw_rows,
                   "ml": ml_rows, "bootstrap": boot, "folds": folds,
                   "ordinal_features": [fcols[j] for j in disc_idx],
                   "config": {"K_IDW": K_IDW, "NOISE_FRAC": NOISE_FRAC, "K_GEOM": K_GEOM,
                              "DISC_MAX_LEVELS": DISC_MAX_LEVELS, "N_FOLDS": N_FOLDS,
                              "SEED": SEED}}, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
