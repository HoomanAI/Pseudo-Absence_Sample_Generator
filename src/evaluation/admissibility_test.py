"""
Admissibility diagnostic + restoration test (one strategy per invocation).

Run one strategy at a time and append to a shared CSV, so each invocation
finishes quickly:

    python admissibility_test.py --data <csv> --pa-dir <dir> --strategy Heuristic \
        --out results/admissibility.csv

Strategies: Heuristic | Random | SA | GAN (clip) | DA-GP-WGAN

WHAT IT MEASURES
----------------
The 13 core predictors in the Alberta dataset are reclassified ordinal class
codes taking only 5 distinct values each (9 for `river`). Fire records lie on
that admissible lattice by construction; IDW-interpolated pseudo-absences do
not. Two quantities are reported per strategy:

  tell_AUC     AUC of a single scalar -- summed distance to the nearest
               admissible level -- carrying zero environmental information.
               1.0 means the classes are perfectly separable by construction
               artifact alone. 0.5 means the artifact is absent.

  RF / XGB AUC downstream 5-fold spatial-block CV, under
                 baseline    = as published
                 admissible  = PA features snapped to the nearest admissible
                               level, restoring the shared lattice
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
DISC_MAX_LEVELS = 12
N_FOLDS = 5

FILES = {
    "Heuristic":  ("heur_pts.npy", "heur_feats.npy"),
    "Random":     ("rand_pts.npy", "rand_feats.npy"),
    "SA":         ("sa_pts.npy", "sa_feats.npy"),
    "GAN (clip)": ("gan_pts.npy", "gan_feats.npy"),
    "DA-GP-WGAN": ("gan_pts_gp_domain_aware.npy", "gan_feats_gp_domain_aware.npy"),
    "Clip":        ("gan_pts_clip.npy", "gan_feats_clip.npy"),
    "GP-standard": ("gan_pts_gp_standard.npy", "gan_feats_gp_standard.npy"),
}


def sfolds(co, n=N_FOLDS):
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


def snap(A, levels):
    B = A.copy()
    for j, lv in levels.items():
        B[:, j] = lv[np.argmin(np.abs(A[:, [j]] - lv[None, :]), axis=1)]
    return B


def tell_scalar(X, levels):
    return np.column_stack([np.min(np.abs(X[:, [j]] - levels[j][None, :]), axis=1)
                            for j in sorted(levels)]).sum(axis=1)


def cv_trees(X, y, co):
    np.random.seed(SEED)
    fid = sfolds(co)
    out = {}
    for nm, clf in [("RF", RandomForestClassifier(100, random_state=SEED, n_jobs=-1)),
                    ("XGB", XGBClassifier(n_estimators=100, objective="binary:logistic",
                                          eval_metric="logloss", verbosity=0,
                                          random_state=SEED))]:
        a = []
        for f in range(N_FOLDS):
            tr, te = fid != f, fid == f
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            clf.fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        out[nm] = (float(np.mean(a)), float(np.std(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pa-dir", required=True)
    ap.add_argument("--strategy", required=True, choices=sorted(FILES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    fc = [c for c in df.columns
          if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    ff = df[fc].values.astype(float)
    disc = [j for j, c in enumerate(fc) if df[c].nunique() <= DISC_MAX_LEVELS]
    levels = {j: np.unique(ff[:, j]) for j in disc}

    pf, fpf = FILES[args.strategy]
    pp = np.load(os.path.join(args.pa_dir, pf))
    pa = np.load(os.path.join(args.pa_dir, fpf))
    co = np.vstack([fire, pp])
    y = np.r_[np.ones(len(fire)), np.zeros(len(pa))]

    rows = []
    for cond, pv in [("baseline", pa), ("admissible", snap(pa, levels))]:
        X = np.vstack([ff, pv])
        t = roc_auc_score(1 - y, tell_scalar(X, levels))
        r = cv_trees(X, y, co)
        rows.append({"strategy": args.strategy, "condition": cond,
                     "tell_AUC": t,
                     "RF_AUC": r["RF"][0], "RF_std": r["RF"][1],
                     "XGB_AUC": r["XGB"][0], "XGB_std": r["XGB"][1],
                     "n_ordinal": len(disc), "n_features": len(fc)})
        print(f'{args.strategy:12s} {cond:11s} tell={t:.5f}  '
              f'RF={r["RF"][0]:.5f}+-{r["RF"][1]:.5f}  '
              f'XGB={r["XGB"][0]:.5f}+-{r["XGB"][1]:.5f}', flush=True)

    new = pd.DataFrame(rows)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        old = old[~((old.strategy == args.strategy))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(args.out, index=False)
    print(f"appended -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
