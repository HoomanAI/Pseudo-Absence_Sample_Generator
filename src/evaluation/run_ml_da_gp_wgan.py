"""ML evaluation for DA-GP-WGAN (domain-aware gradient-penalty) pseudo-absences,
using the exact same 4-model/5-fold spatial-CV procedure as run_ml_v2.py so the
result is directly comparable to cv_h/cv_r/cv_s/cv_g.npy."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire = df[['LONGITUDE', 'LATITUDE']].values
fcols = [c for c in df.columns if c not in ['FIRE', 'LONGITUDE', 'LATITUDE', 'YEAR', 'MONTH', 'DAY']]
fire_feats = df[fcols].values.astype(float)

def sfolds(coords, n=5):
    lat = pd.qcut(coords[:, 1], n, labels=False, duplicates='drop')
    lon = pd.qcut(coords[:, 0], n, labels=False, duplicates='drop')
    bl = lat * n + lon; u = np.unique(bl); np.random.shuffle(u)
    g = np.array_split(u, n); fid = np.zeros(len(coords), int)
    for fi, gr in enumerate(g):
        for b in gr: fid[bl == b] = fi
    return fid

def run_cv(ab_pts, ab_feats, return_curves=False):
    X = np.vstack([fire_feats, ab_feats]); y = np.array([1] * len(fire) + [0] * len(ab_pts))
    coords = np.vstack([fire, ab_pts]); fid = sfolds(coords)
    models = {
        'RandomForest': RandomForestClassifier(100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, objective='binary:logistic',
                                  eval_metric='logloss', verbosity=0, random_state=42),
        'KNN': KNeighborsClassifier(9, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    out = {}; roc_d = {}; proba_d = {}
    for nm, clf in models.items():
        aucs, tsss, fps_all, tps_all = [], [], [], []; all_yte = []; all_prob = []
        for fold in range(5):
            tr, te = fid != fold, fid == fold
            if te.sum() < 10 or len(np.unique(y[te])) < 2: continue
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            if nm == 'SVM':
                i0 = np.where(ytr == 0)[0]; i1 = np.where(ytr == 1)[0]
                s = np.concatenate([np.random.choice(i0, min(500, len(i0)), False),
                                     np.random.choice(i1, min(500, len(i1)), False)])
                Xtr, ytr = Xtr[s], ytr[s]
            if nm in ('SVM', 'KNN'):
                sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
            clf.fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, p))
            pred = (p >= 0.5).astype(int)
            tp = ((pred == 1) & (yte == 1)).sum(); tn = ((pred == 0) & (yte == 0)).sum()
            fp = ((pred == 1) & (yte == 0)).sum(); fn = ((pred == 0) & (yte == 1)).sum()
            tsss.append(tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9) - 1)
            all_yte.extend(yte.tolist()); all_prob.extend(p.tolist())
            if return_curves:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(yte, p); fps_all.append(fpr); tps_all.append(tpr)
        out[nm] = {'AUC': np.mean(aucs), 'AUC_std': np.std(aucs),
                   'TSS': np.mean(tsss), 'TSS_std': np.std(tsss), 'folds': aucs}
        proba_d[nm] = {'yte': np.array(all_yte), 'prob': np.array(all_prob)}
        if return_curves: roc_d[nm] = (fps_all, tps_all)
    return out, roc_d, proba_d

da_pts = np.load('../../outputs/gan_pts_gp_domain_aware.npy')
da_feats = np.load('../../outputs/gan_feats_gp_domain_aware.npy')
print(f'DA-GP-WGAN: {len(da_pts)} points, {da_feats.shape[1]} features')
print('Running CV for DA-GP-WGAN ...')
res, roc, pr = run_cv(da_pts, da_feats, return_curves=True)
for m, v in res.items():
    print(f"  {m}: AUC={v['AUC']:.4f}  TSS={v['TSS']:.4f}")

np.save('../../outputs/cv_da.npy', {'res': res, 'roc': roc, 'pr': pr}, allow_pickle=True)
print('Saved cv_da.npy')
