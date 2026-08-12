r"""
Downstream evaluation of every retrained arm: 5-classifier spatial-block CV
plus the admissibility tell, aggregated as mean +- sd over seeds.

WHY A NEW SCRIPT
----------------
classifier_suite.py reads a flat directory via a hardcoded ARM_FILES map
(gan_pts_gp_standard.npy etc.). retrain_all.py writes per-cell directories
instead — retrain/<label>_seed<N>/pts_<arm>.npy — with three seeds per arm, so
the old loader cannot see them. This script reads retrain/manifest.csv directly
and therefore always evaluates exactly the cells that were trained.

CLASSIFIER SUITE (per the revision brief — XGBoost and Random Forest removed
from the main comparison; they survive only in the artifact-diagnosis section
because their zero-variance AUC = 1.00000 IS the evidence for the artifact):

    Logistic Regression   L2, max_iter=2000
    Gaussian Naive Bayes  defaults
    Decision Tree (CART)  max_depth=6, min_samples_leaf=20
    KNN                   k=9, Minkowski
    SVM                   RBF, probability=True, 500+500 train subsample

Inputs are standardized per fold, fitted on training data only, FOR EVERY
MODEL. The original run_ml_v2.py scaled only for KNN and SVM; Logistic
Regression and Naive Bayes need it too.

THE ADMISSIBILITY TELL is reported alongside every arm. It is the gate: a tell
near 1.0 means the classes are separable by construction alone and no AUC in
that row is interpretable. Arms carrying the admissibility head should read
~0.5; arms without it should read ~1.0.

USAGE
    python evaluate_arms.py ^
        --data ..\Pseudo-Absence_Sample_Generator\data\Fire_points_dataset_final_csv.csv ^
        --retrain retrain ^
        --legacy-dir "..\cWGAN codes" ^
        --out results

Add --only dagp_base to evaluate a single arm, or --skip-svm for a fast pass
(SVM dominates runtime).

OUTPUTS
    arm_eval_percell.csv   one row per (arm, seed, model)
    arm_eval_summary.csv   mean +- sd over seeds, per (arm, model)
    arm_eval_tell.csv      admissibility tell per (arm, seed)
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

N_FOLDS = 5
SEED = 42
DISC_MAX_LEVELS = 12

# non-GAN baselines: single point set each, no seed dimension
LEGACY = {
    "random":    ("rand_pts.npy", "rand_feats.npy"),
    "heuristic": ("heur_pts.npy", "heur_feats.npy"),
}


def sfolds(co, n=N_FOLDS):
    """Identical to run_ml_v2.py::sfolds so folds stay comparable."""
    la = pd.qcut(co[:, 1], n, labels=False, duplicates="drop")
    lo = pd.qcut(co[:, 0], n, labels=False, duplicates="drop")
    bl = la * n + lo
    u = np.unique(bl)
    np.random.shuffle(u)
    fid = np.zeros(len(co), int)
    for fi, gr in enumerate(np.array_split(u, n)):
        for b in gr:
            fid[bl == b] = fi
    return fid


def make_models(skip_svm=False, prob_only=False):
    m = {
        "LogReg": lambda: LogisticRegression(max_iter=2000),
        "NaiveBayes": lambda: GaussianNB(),
        "CART": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                               random_state=SEED),
        "KNN": lambda: KNeighborsClassifier(9, n_jobs=-1),
    }
    if not skip_svm:
        m["SVM"] = lambda: SVC(kernel="rbf", probability=True, random_state=SEED)
    return {k:v for k,v in m.items() if (not prob_only or k in ("LogReg","SVM"))}


def run_cv(X, y, coords, skip_svm=False, dump_probs=False, prob_only=False):
    np.random.seed(SEED)
    fid = sfolds(coords)
    out = {}
    prob_rows = []
    for nm, mk in make_models(skip_svm, prob_only).items():
        aucs, tsss = [], []
        for f in range(N_FOLDS):
            tr, te = fid != f, fid == f
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            if nm == "SVM":
                i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
                s = np.concatenate([np.random.choice(i0, min(500, len(i0)), False),
                                    np.random.choice(i1, min(500, len(i1)), False)])
                Xtr, ytr = Xtr[s], ytr[s]
            # scale for EVERY model, fitted on train only
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf = mk().fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            if dump_probs:
                for obs, yt, yp in zip(np.where(te)[0], yte, p):
                    prob_rows.append({"model":nm,"fold":f,"obs_index":int(obs),
                                      "y_true":int(yt),"y_prob":float(yp)})
            aucs.append(roc_auc_score(yte, p))
            pred = (p >= 0.5).astype(int)
            tp = ((pred == 1) & (yte == 1)).sum(); tn = ((pred == 0) & (yte == 0)).sum()
            fp = ((pred == 1) & (yte == 0)).sum(); fn = ((pred == 0) & (yte == 1)).sum()
            tsss.append(tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9) - 1)
        out[nm] = (float(np.mean(aucs)), float(np.std(aucs)),
                   float(np.mean(tsss)), float(np.std(tsss)))
    return out, prob_rows


def tell_auc(X, y, levels):
    """Zero-environment probe: summed distance to the nearest admissible level."""
    s = np.column_stack([np.min(np.abs(X[:, [j]] - levels[j][None, :]), axis=1)
                         for j in sorted(levels)]).sum(axis=1)
    return float(roc_auc_score(1 - y, s))


def snap_to_lattice(A, levels):
    """Project the ordinal dimensions onto the nearest admissible class level.

    Needed for a FAIR comparison. Arms carrying the admissibility head emit
    on-lattice features natively (tell ~0.5); arms without it do not (tell 1.0),
    and their downstream AUC is inflated by that construction artifact. Scoring
    a de-confounded arm against a confounded one is apples-to-oranges. Snapping
    every arm puts all of them on the same footing, after which differences are
    attributable to pseudo-absence placement rather than to feature construction.
    """
    B = A.copy()
    for j, lv in levels.items():
        B[:, j] = lv[np.argmin(np.abs(A[:, [j]] - lv[None, :]), axis=1)]
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--retrain", required=True)
    ap.add_argument("--legacy-dir", default=None,
                    help="dir holding rand_/heur_ npy files for the non-GAN baselines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--dump-probs", action="store_true",
                    help="write per-fold y_true/y_prob rows for calibration")
    ap.add_argument("--prob-only", action="store_true",
                    help="evaluate only LogReg and SVM (for calibration export)")
    ap.add_argument("--snap-all", action="store_true",
                    help="project EVERY arm's ordinal features onto the admissible "
                         "lattice before evaluation, so all arms share tell ~0.5 "
                         "and the comparison is like-for-like")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. --tag _snapped")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    fc = [c for c in df.columns
          if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    ff = df[fc].values.astype(float)
    disc = [j for j, c in enumerate(fc) if df[c].nunique() <= DISC_MAX_LEVELS]
    levels = {j: np.unique(ff[:, j]) for j in disc}
    print(f"fire {len(fire)}  features {len(fc)}  ordinal {len(disc)}", flush=True)

    jobs = []  # (label, seed, pts_path, feats_path)
    man_path = os.path.join(args.retrain, "manifest.csv")
    if os.path.exists(man_path):
        man = pd.read_csv(man_path)
        for _, r in man.iterrows():
            d = os.path.join(args.retrain, f"{r.label}_seed{int(r.seed)}")
            jobs.append((r.label, int(r.seed),
                         os.path.join(d, f"pts_{r.arm}.npy"),
                         os.path.join(d, f"feats_{r.arm}.npy")))
    if args.legacy_dir:
        for lab, (pf, fpf) in LEGACY.items():
            p = os.path.join(args.legacy_dir, pf)
            q = os.path.join(args.legacy_dir, fpf)
            if os.path.exists(p) and os.path.exists(q):
                jobs.append((lab, -1, p, q))
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    print(f"{len(jobs)} arm-seed cells to evaluate", flush=True)

    rows, tells, probabilities = [], [], []
    for lab, seed, ppath, fpath in jobs:
        if not (os.path.exists(ppath) and os.path.exists(fpath)):
            print(f"  MISSING {lab} seed {seed}: {ppath}", flush=True)
            continue
        pts, pa = np.load(ppath), np.load(fpath)
        if args.snap_all:
            pa = snap_to_lattice(pa, levels)
        X = np.vstack([ff, pa])
        y = np.r_[np.ones(len(ff)), np.zeros(len(pa))]
        coords = np.vstack([fire, pts])

        t = tell_auc(X, y, levels)
        tells.append({"arm": lab, "seed": seed, "tell_AUC": t,
                      "n_pa": int(len(pa))})
        res, probs = run_cv(X, y, coords, args.skip_svm, args.dump_probs, args.prob_only)
        for z in probs:
            z.update({"arm":lab,"seed":seed})
        probabilities.extend(probs)
        for m, (a, asd, ts, tsd) in res.items():
            rows.append({"arm": lab, "seed": seed, "model": m,
                         "AUC": a, "AUC_std": asd, "TSS": ts, "TSS_std": tsd,
                         "tell_AUC": t})
        best = " ".join(f"{m}={v[0]:.4f}" for m, v in res.items())
        print(f"  {lab:14s} seed {seed:3d}  tell={t:.4f}  {best}", flush=True)

    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(args.out, f"arm_eval_percell{args.tag}.csv"), index=False)
    pd.DataFrame(tells).to_csv(os.path.join(args.out, f"arm_eval_tell{args.tag}.csv"), index=False)
    if args.dump_probs:
        pd.DataFrame(probabilities).to_csv(
            os.path.join(args.out, f"arm_eval_probs{args.tag}.csv"), index=False)

    if len(pc):
        s = (pc.groupby(["arm", "model"])
               .agg(n_seeds=("seed", "nunique"),
                    AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
                    TSS_mean=("TSS", "mean"), TSS_sd=("TSS", "std"),
                    tell=("tell_AUC", "mean"))
               .reset_index())
        s.to_csv(os.path.join(args.out, f"arm_eval_summary{args.tag}.csv"), index=False)
        print("\n=== mean +- sd over seeds ===", flush=True)
        print(s.to_string(index=False), flush=True)
        print("\nGATE: any arm whose tell is near 1.0 has classes separable by "
              "construction alone; its AUC column is not interpretable.", flush=True)


if __name__ == "__main__":
    main()
