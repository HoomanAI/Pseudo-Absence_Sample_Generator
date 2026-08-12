"""
Five-classifier downstream evaluation for the KBS revision.

Replaces the tree-ensemble-only comparison (RF/XGBoost — see FINDINGS.md
section 1: their AUC=1.00000 with zero variance is the admissibility
artifact, not a genuine ranking signal) with five classifiers that do not
share that failure mode:

  Logistic Regression   L2, max_iter=2000, standardized inputs
  Gaussian Naive Bayes  defaults
  Decision Tree (CART)  max_depth=6, min_samples_leaf=20, seed 42
  KNN                   k=9, Minkowski, standardized inputs
  SVM                   RBF, probability=True, seed 42, standardized inputs,
                         500+500 train subsample

Standardization is per-fold, fitting the scaler on the training fold only,
for ALL FIVE models (the original run_ml_v2.py only scaled for KNN/SVM;
Logistic Regression and Naive Bayes need it too and did not get it before).

Spatial block CV (sfolds) is copied verbatim from run_ml_v2.py to stay
procedurally identical to the published protocol (5x5 quantile grid -> 25
blocks -> 5 folds).

Usage:
    python classifier_suite.py --data <fire_csv> --outputs <dir with *_pts.npy/*_feats.npy>
                                --out results/ --arms random,heuristic,gan_clip,gp_standard

Arms are read as <outputs>/<arm>_pts.npy and <outputs>/<arm>_feats.npy, except
the fixed aliases below (matching the existing filenames in cWGAN codes/).
"""
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# arm name -> (points file, features file), relative to --outputs
ARM_FILES = {
    "random":     ("rand_pts.npy",  "rand_feats.npy"),
    "heuristic":  ("heur_pts.npy",  "heur_feats.npy"),
    "gan_clip":   ("gan_pts.npy",   "gan_feats.npy"),
    "gp_standard":("gan_pts_gp_standard.npy",     "gan_feats_gp_standard.npy"),
    "v1_da_gp":   ("gan_pts_gp_domain_aware.npy", "gan_feats_gp_domain_aware.npy"),  # ESWA-submitted arch, reference only
    "dagp_strict":  ("dagp_strict_pts.npy",  "dagp_strict_feats.npy"),
    "dagp_base":    ("dagp_base_pts.npy",    "dagp_base_feats.npy"),
    "dagp_relaxed": ("dagp_relaxed_pts.npy", "dagp_relaxed_feats.npy"),
}

MODELS = {
    "LogReg": lambda: LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED),
    "NaiveBayes": lambda: GaussianNB(),
    "CART": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=9, metric="minkowski"),
    "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
}


def sfolds(coords, n=5):
    """Verbatim from run_ml_v2.py: 5x5 quantile grid -> 25 blocks -> 5 folds."""
    lat = pd.qcut(coords[:, 1], n, labels=False, duplicates="drop")
    lon = pd.qcut(coords[:, 0], n, labels=False, duplicates="drop")
    bl = lat * n + lon
    u = np.unique(bl)
    np.random.shuffle(u)
    g = np.array_split(u, n)
    fid = np.zeros(len(coords), int)
    for fi, gr in enumerate(g):
        for b in gr:
            fid[bl == b] = fi
    return fid


def run_cv(fire_feats, fire_coords, ab_pts, ab_feats):
    X = np.vstack([fire_feats, ab_feats])
    y = np.array([1] * len(fire_feats) + [0] * len(ab_pts))
    coords = np.vstack([fire_coords, ab_pts])
    fid = sfolds(coords)

    out = {}
    for name, make_clf in MODELS.items():
        aucs, tsss = [], []
        for fold in range(5):
            tr, te = fid != fold, fid == fold
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            if name == "SVM":
                i0 = np.where(ytr == 0)[0]
                i1 = np.where(ytr == 1)[0]
                s = np.concatenate([
                    np.random.choice(i0, min(500, len(i0)), False),
                    np.random.choice(i1, min(500, len(i1)), False),
                ])
                Xtr, ytr = Xtr[s], ytr[s]
            # standardize on TRAIN ONLY, for every model (fixes the original
            # code's KNN/SVM-only scaling gap)
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            clf = make_clf()
            clf.fit(Xtr_s, ytr)
            p = clf.predict_proba(Xte_s)[:, 1]
            aucs.append(roc_auc_score(yte, p))
            pred = (p >= 0.5).astype(int)
            tp = ((pred == 1) & (yte == 1)).sum()
            tn = ((pred == 0) & (yte == 0)).sum()
            fp = ((pred == 1) & (yte == 0)).sum()
            fn = ((pred == 0) & (yte == 1)).sum()
            tsss.append(tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9) - 1)
        out[name] = {
            "AUC": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
            "TSS": float(np.mean(tsss)), "TSS_std": float(np.std(tsss)),
            "folds_AUC": [float(a) for a in aucs],
        }
    return out


def bootstrap_dauc(folds_a, folds_b, n=2000, seed=77):
    a = np.array(folds_a); b = np.array(folds_b)
    rb = np.random.RandomState(seed)
    d = np.array([
        rb.choice(a, len(a), True).mean() - rb.choice(b, len(b), True).mean()
        for _ in range(n)
    ])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outputs", required=True, help="dir with *_pts.npy/*_feats.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--baselines", default="random,heuristic,gan_clip",
                     help="comma-separated arms treated as baselines for bootstrap dAUC vs each DA-GP arm")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data).dropna()
    fire_coords = df[["LONGITUDE", "LATITUDE"]].values
    fcols = [c for c in df.columns if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    fire_feats = df[fcols].values.astype(float)

    arms = [a.strip() for a in args.arms.split(",")]
    baselines = [a.strip() for a in args.baselines.split(",")]

    results = {}
    rows = []
    t0 = time.time()
    for arm in arms:
        if arm not in ARM_FILES:
            print(f"SKIP {arm}: no known file mapping"); continue
        pf, ff = ARM_FILES[arm]
        ppath = os.path.join(args.outputs, pf)
        fpath = os.path.join(args.outputs, ff)
        if not (os.path.exists(ppath) and os.path.exists(fpath)):
            print(f"SKIP {arm}: {ppath} or {fpath} not found (not trained yet)")
            continue
        pts = np.load(ppath); feats = np.load(fpath)
        print(f"=== {arm}: {len(pts)} points ===", flush=True)
        res = run_cv(fire_feats, fire_coords, pts, feats)
        results[arm] = res
        for model, v in res.items():
            rows.append({"arm": arm, "model": model, "AUC": v["AUC"], "AUC_std": v["AUC_std"],
                         "TSS": v["TSS"], "TSS_std": v["TSS_std"]})
            print(f"  {model:12s} AUC={v['AUC']:.4f}+/-{v['AUC_std']:.4f}  TSS={v['TSS']:.4f}+/-{v['TSS_std']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "classifier_suite_results.csv"), index=False)
    with open(os.path.join(args.out, "classifier_suite_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # bootstrap dAUC: each DA-GP-family arm vs each baseline, per model
    dagp_arms = [a for a in arms if a in results and a not in baselines]
    boot_rows = []
    for da in dagp_arms:
        for base in baselines:
            if base not in results: continue
            for model in MODELS:
                if model not in results[da] or model not in results[base]: continue
                obs, lo, hi = bootstrap_dauc(results[da][model]["folds_AUC"], results[base][model]["folds_AUC"])
                boot_rows.append({"arm": da, "baseline": base, "model": model,
                                   "dAUC": obs, "CI_lo": lo, "CI_hi": hi,
                                   "significant": bool(lo > 0 or hi < 0)})
    pd.DataFrame(boot_rows).to_csv(os.path.join(args.out, "classifier_suite_bootstrap.csv"), index=False)
    print(f"\nDone in {time.time()-t0:.1f}s. Wrote classifier_suite_results.csv/.json "
          f"and classifier_suite_bootstrap.csv to {args.out}")


if __name__ == "__main__":
    main()
