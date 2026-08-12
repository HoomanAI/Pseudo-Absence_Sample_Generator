"""
Quebec: real-absence validation of pseudo-absence quality.

WHY QUEBEC IS USEFUL
--------------------
The Quebec file (WildFire_TrainTest.xlsx, 1,042 points, 521 fire / 521 non-fire,
13 environmental features) differs from the Alberta file in two ways that make it
a natural control rather than merely a second study area:

  1. Its features are CONTINUOUS (275-1,041 distinct values per feature across
     1,042 rows), not reclassified into 5 ordinal levels. The admissibility
     artifact that drives Alberta's AUC = 1.00000 therefore cannot arise.
  2. Its Fire=0 points carry NO interpolation fingerprint (geometry-only
     RF AUC 0.542, m1 fire:non-fire ratio 0.968, against Alberta's 0.9995 and
     7.91). They were not constructed by interpolating from fire records.

So Quebec supplies something the Alberta experiment structurally cannot: a set of
non-fire points with directly-measured environmental values, usable as a
ground-truth target for judging synthetic negatives.

THE THREE EXPERIMENTS
---------------------
E1  Natural experiment. Report the admissibility tell and tree-ensemble AUC for
    Alberta (reclassified + interpolated negatives) beside Quebec (continuous +
    observed negatives). Contaminated construction gives AUC 1.00000; clean
    construction gives ~0.74.

E2  Distributional fidelity to observed absences. For each PA strategy, measure
    how closely its synthetic negatives match the observed Fire=0 points, in
    geographic space (Ripley K-SSE, centroid offset, border fraction) and in
    feature space (energy distance, per-feature standardized bias). The observed
    absences are the reference, not the fire points -- this is a target no
    experiment in the submission uses.

E3  Transfer test: train on synthetic, evaluate against observed.
    Train a classifier on (fire vs synthetic PA), then score it two ways:
        held-out synthetic PA   -> the number a conventional CV protocol reports
        observed Fire=0 points  -> what the model is actually worth
    The gap between the two is the inflation attributable to PA construction.
    An honest PA method should show a small gap.

    Feature assignment for synthetic points uses the same IDW-over-k=7-nearest-
    fire-records operator as pseudo_absence_pipeline.py, so the gap measured
    here is the cost of that operator in a clean, continuous covariate space --
    complementary to the ordinal-lattice artifact documented for Alberta.

USAGE
    python quebec_validation.py --quebec <xlsx or csv> --out <dir>
    python quebec_validation.py --quebec ... --alberta <fire csv> --out <dir>
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATS = ["Elevation", "Slope", "Aspect", "Profile Curvature", "Plan Curvature",
         "Valley Depth", "TWI", "Distance From Rivers", "Distance From Roads",
         "NDVI", "Mean Annual Precipitation", "Mean Annual Temperature",
         "Mean Annual Wind Speed"]
R_VALS = np.linspace(0.05, 0.80, 15)
BORDER_DEG = 0.15
K_IDW = 7
NOISE_FRAC = 0.05
BUFFER_KM = 10.0
SEED = 42


# --------------------------------------------------------------------------
def load_quebec(path):
    if path.lower().endswith((".xlsx", ".xls")):
        parts = [pd.read_excel(path, sheet_name=s).assign(split=s)
                 for s in ("Train", "Test")]
        q = pd.concat(parts, ignore_index=True)
    else:
        q = pd.read_csv(path)
    return q.dropna(subset=FEATS + ["LATITUDE", "LONGITUDE", "Fire"])


def k_curve(pts, area):
    t = cKDTree(pts)
    n = len(pts)
    return (t.count_neighbors(t, R_VALS) - n) / n / (n / area)


def k_sse(a, b, area):
    return float(np.sum((k_curve(a, area) - k_curve(b, area)) ** 2))


def border_frac(pts, bounds):
    lo_x, hi_x, lo_y, hi_y = bounds
    return float((((pts[:, 0] - lo_x) < BORDER_DEG) | ((hi_x - pts[:, 0]) < BORDER_DEG) |
                  ((pts[:, 1] - lo_y) < BORDER_DEG) | ((hi_y - pts[:, 1]) < BORDER_DEG)).mean())


def energy_distance(A, B):
    """Multivariate energy distance between two standardized samples.
    E = 2*E|X-Y| - E|X-X'| - E|Y-Y'|; zero iff the distributions coincide."""
    def m(P, Q):
        return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * m(A, B) - m(A, A) - m(B, B))


def idw_assign(pts, fire_pts, fire_feats, k=K_IDW, seed=SEED):
    d, ix = cKDTree(fire_pts).query(pts, k=k)
    d = np.maximum(d, 1e-9)
    w = 1.0 / d ** 2
    w /= w.sum(1, keepdims=True)
    f = np.einsum("nk,nkf->nf", w, fire_feats[ix])
    return f + np.random.RandomState(seed).normal(
        0, NOISE_FRAC * fire_feats.std(0), f.shape)


# --------------------------------------------------------------------------
def gen_random(fire_pts, n, bounds, rng):
    lo_x, hi_x, lo_y, hi_y = bounds
    return np.column_stack([rng.uniform(lo_x, hi_x, n), rng.uniform(lo_y, hi_y, n)])


def gen_heuristic(fire_pts, n, bounds, rng, n_cells=10):
    """10 km hard buffer, exponential acceptance, fire-density weighting --
    the same construction as generate_bp_v6() in pseudo_absence_pipeline.py."""
    lo_x, hi_x, lo_y, hi_y = bounds
    d_min = BUFFER_KM / 111.0
    lam = 2.0 * BUFFER_KM / 111.0
    tree = cKDTree(fire_pts)

    xe = np.linspace(lo_x, hi_x, n_cells + 1)
    ye = np.linspace(lo_y, hi_y, n_cells + 1)
    dens = np.zeros((n_cells, n_cells))
    xi = np.clip(np.digitize(fire_pts[:, 0], xe) - 1, 0, n_cells - 1)
    yi = np.clip(np.digitize(fire_pts[:, 1], ye) - 1, 0, n_cells - 1)
    for i, j in zip(xi, yi):
        dens[i, j] += 1
    dens /= max(dens.max(), 1)

    cand = np.column_stack([rng.uniform(lo_x, hi_x, n * 200),
                            rng.uniform(lo_y, hi_y, n * 200)])
    dd, _ = tree.query(cand, k=1)
    keep = dd >= d_min
    cand, dd = cand[keep], dd[keep]
    p = 1.0 - np.exp(-(dd - d_min) / lam)
    cand = cand[rng.uniform(0, 1, len(cand)) < p]

    ci = np.clip(np.digitize(cand[:, 0], xe) - 1, 0, n_cells - 1)
    cj = np.clip(np.digitize(cand[:, 1], ye) - 1, 0, n_cells - 1)
    w = 0.3 + 0.7 * dens[ci, cj]
    w = w / w.sum()
    take = min(n, len(cand))
    return cand[rng.choice(len(cand), take, replace=False, p=w)]


# --------------------------------------------------------------------------
def transfer_test(fire_feats, pa_feats, real_abs_feats, pa_is_observed=False, seed=SEED,
                  single_thread=False):
    """E3: train on fire vs synthetic PA; score on held-out PA and on observed
    absences, using folds so neither score is computed on training data.

    pa_is_observed : set True for the oracle arm, where the PA set IS the
        observed absences. In that case the observed-absence score must be
        restricted to the absences in the held-out fold; scoring against all
        of them would leak the ~80% seen during training and inflate the result.
        For synthetic arms no observed absence is ever used in training, so the
        full observed set is legitimately held out.
    """
    Xf, yf = fire_feats, np.ones(len(fire_feats))
    res = {}
    for nm, mk in (("RF", lambda: RandomForestClassifier(300, random_state=seed, n_jobs=1 if single_thread else -1)),
                   ("XGB", lambda: XGBClassifier(n_estimators=300, eval_metric="logloss",
                                                 verbosity=0, random_state=seed,
                                                 n_jobs=1 if single_thread else None))):
        syn, real = [], []
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        Xall = np.vstack([Xf, pa_feats])
        yall = np.r_[yf, np.zeros(len(pa_feats))]
        n_f = len(Xf)
        for tr, te in skf.split(Xall, yall):
            clf = mk().fit(Xall[tr], yall[tr])
            syn.append(roc_auc_score(yall[te], clf.predict_proba(Xall[te])[:, 1]))
            fire_te = Xall[te][yall[te] == 1]
            if pa_is_observed:
                # only the observed absences held out in this fold
                neg = real_abs_feats[te[te >= n_f] - n_f]
            else:
                neg = real_abs_feats
            if len(neg) == 0 or len(fire_te) == 0:
                continue
            Xr = np.vstack([fire_te, neg])
            yr = np.r_[np.ones(len(fire_te)), np.zeros(len(neg))]
            real.append(roc_auc_score(yr, clf.predict_proba(Xr)[:, 1]))
        res[nm] = {"auc_synthetic": float(np.mean(syn)), "auc_synthetic_std": float(np.std(syn)),
                   "auc_observed": float(np.mean(real)), "auc_observed_std": float(np.std(real)),
                   "inflation": float(np.mean(syn) - np.mean(real))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quebec", required=True)
    ap.add_argument("--alberta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-points", default=None,
                    help="optional directory for the already-generated Quebec point arrays")
    ap.add_argument("--single-thread", action="store_true",
                    help="diagnostic mode: force RF and XGBoost to one worker")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(SEED)

    q = load_quebec(args.quebec)
    fire = q[q.Fire == 1]
    absn = q[q.Fire == 0]
    fire_pts = fire[["LONGITUDE", "LATITUDE"]].values
    abs_pts = absn[["LONGITUDE", "LATITUDE"]].values
    fire_feats = fire[FEATS].values.astype(float)
    abs_feats = absn[FEATS].values.astype(float)
    n_pa = len(abs_pts)

    all_pts = q[["LONGITUDE", "LATITUDE"]].values
    bounds = (all_pts[:, 0].min(), all_pts[:, 0].max(),
              all_pts[:, 1].min(), all_pts[:, 1].max())
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    print(f"Quebec: {len(fire_pts)} fire, {len(abs_pts)} observed absence, "
          f"{len(FEATS)} features, area {area:.4f} deg^2", flush=True)

    # ---------------- E1: natural experiment ----------------
    e1 = []
    nun = {c: int(q[c].nunique()) for c in FEATS}
    e1.append({"region": "Quebec", "features": len(FEATS),
               "min_distinct_values": min(nun.values()),
               "max_distinct_values": max(nun.values()),
               "reclassified": False,
               "negatives": "observed",
               "note": "distinct-value counts far exceed 5; no admissible lattice"})
    if args.alberta:
        a = pd.read_csv(args.alberta).dropna()
        ac = [c for c in a.columns
              if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
        an = {c: int(a[c].nunique()) for c in ac}
        ordn = [c for c in ac if an[c] <= 12]
        e1.append({"region": "Alberta", "features": len(ac),
                   "min_distinct_values": min(an.values()),
                   "max_distinct_values": max(an.values()),
                   "reclassified": True,
                   "negatives": "IDW-interpolated",
                   "note": f"{len(ordn)} ordinal features with 5-9 admissible levels"})
    pd.DataFrame(e1).to_csv(os.path.join(args.out, "qc_e1_natural_experiment.csv"), index=False)

    # honest Quebec baseline: fire vs observed absences
    Xo = np.vstack([fire_feats, abs_feats])
    yo = np.r_[np.ones(len(fire_feats)), np.zeros(len(abs_feats))]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    honest = {}
    jobs = 1 if args.single_thread else -1
    for nm, mk in (("RF", lambda: RandomForestClassifier(300, random_state=SEED, n_jobs=jobs)),
                   ("XGB", lambda: XGBClassifier(n_estimators=300, eval_metric="logloss",
                                                 verbosity=0, random_state=SEED,
                                                 n_jobs=(1 if args.single_thread else None)))):
        s = [roc_auc_score(yo[te], mk().fit(Xo[tr], yo[tr]).predict_proba(Xo[te])[:, 1])
             for tr, te in skf.split(Xo, yo)]
        honest[nm] = {"auc": float(np.mean(s)), "std": float(np.std(s))}
        print(f"E1 Quebec honest (fire vs observed absence) {nm}: "
              f"{np.mean(s):.4f} +- {np.std(s):.4f}", flush=True)

    # ---------------- E2 + E3 per strategy ----------------
    strategies = {
        "Random": gen_random(fire_pts, n_pa, bounds, rng),
        "Heuristic": gen_heuristic(fire_pts, n_pa, bounds, rng),
        "Observed (oracle)": abs_pts,
    }

    # Save only after all stochastic point generation is complete so enabling
    # the dump cannot consume RNG state or change any evaluation result.
    if args.dump_points:
        os.makedirs(args.dump_points, exist_ok=True)
        np.save(os.path.join(args.dump_points, "qc_pts_fire.npy"), fire_pts)
        np.save(os.path.join(args.dump_points, "qc_pts_random.npy"), strategies["Random"])
        np.save(os.path.join(args.dump_points, "qc_pts_observed_absence.npy"), abs_pts)
        np.save(os.path.join(args.dump_points, "qc_pts_heuristic.npy"), strategies["Heuristic"])
        lo_x, hi_x, lo_y, hi_y = (float(v) for v in bounds)
        dump_meta = {"lo_x": lo_x, "hi_x": hi_x, "lo_y": lo_y, "hi_y": hi_y,
                     "n": {"fire": int(len(fire_pts)),
                           "random": int(len(strategies["Random"])),
                           "observed_absence": int(len(abs_pts)),
                           "heuristic": int(len(strategies["Heuristic"]))}}
        with open(os.path.join(args.dump_points, "qc_bounds.json"), "w") as fh:
            json.dump(dump_meta, fh, indent=2)

    sc = StandardScaler().fit(np.vstack([fire_feats, abs_feats]))
    abs_z = sc.transform(abs_feats)

    e2, e3 = [], []
    for name, pts in strategies.items():
        if name == "Observed (oracle)":
            pf = abs_feats
        else:
            pf = idw_assign(pts, fire_pts, fire_feats)

        e2.append({
            "strategy": name, "n": int(len(pts)),
            "K_SSE_vs_observed_absence": k_sse(pts, abs_pts, area),
            "K_SSE_vs_fire": k_sse(pts, fire_pts, area),
            "centroid_offset_vs_observed": float(
                np.sqrt(((pts.mean(0) - abs_pts.mean(0)) ** 2).sum())),
            "border_frac": border_frac(pts, bounds),
            "energy_distance_vs_observed": energy_distance(sc.transform(pf), abs_z),
            "mean_abs_standardized_bias": float(
                np.abs(sc.transform(pf).mean(0) - abs_z.mean(0)).mean()),
        })
        print(f"E2 {name:18s} K-SSE(vs observed)={e2[-1]['K_SSE_vs_observed_absence']:7.4f}  "
              f"energy={e2[-1]['energy_distance_vs_observed']:7.4f}  "
              f"bias={e2[-1]['mean_abs_standardized_bias']:.4f}", flush=True)

        t = transfer_test(fire_feats, pf, abs_feats,
                          pa_is_observed=(name == "Observed (oracle)"),
                          single_thread=args.single_thread)
        for m, v in t.items():
            e3.append({"strategy": name, "model": m, **v})
            print(f"E3 {name:18s} {m:4s} synthetic={v['auc_synthetic']:.4f}  "
                  f"observed={v['auc_observed']:.4f}  "
                  f"inflation={v['inflation']:+.4f}", flush=True)

    pd.DataFrame(e2).to_csv(os.path.join(args.out, "qc_e2_fidelity.csv"), index=False)
    pd.DataFrame(e3).to_csv(os.path.join(args.out, "qc_e3_transfer.csv"), index=False)
    with open(os.path.join(args.out, "qc_summary.json"), "w") as fh:
        json.dump({"e1": e1, "quebec_honest_baseline": honest,
                   "e2": e2, "e3": e3,
                   "config": {"R_VALS": R_VALS.tolist(), "K_IDW": K_IDW,
                              "NOISE_FRAC": NOISE_FRAC, "BUFFER_KM": BUFFER_KM,
                              "SEED": SEED, "area_deg2": area}}, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
