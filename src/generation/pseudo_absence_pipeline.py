"""
Pseudo-Absence Generation & Evaluation Pipeline
Fire Points Dataset - Alberta, Canada
Heuristic (BP_V6 + CS_V5) vs Random generation
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv')
df = df.dropna()

fire_pts = df[['LONGITUDE', 'LATITUDE']].values
feature_cols = [c for c in df.columns if c not in
                ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats = df[feature_cols].values

print(f"Fire points: {len(fire_pts)}, Features: {len(feature_cols)}")
print(f"Lat: {fire_pts[:,1].min():.3f} – {fire_pts[:,1].max():.3f}")
print(f"Lon: {fire_pts[:,0].min():.3f} – {fire_pts[:,0].max():.3f}")

N_TARGET = len(fire_pts)
DEG_PER_KM = 1.0 / 111.0    # approx degrees per km at this latitude
D_MIN_DEG  = 10.0 * DEG_PER_KM   # 10 km hard buffer
LAM_DEG    = 20.0 * DEG_PER_KM   # lambda = 20 km

# Build KDTree on fire points (use degree space, approx OK at this lat range)
fire_tree = cKDTree(fire_pts)

# ─────────────────────────────────────────────
# 2. HELPER: IDW feature interpolation
# ─────────────────────────────────────────────
def interpolate_features(new_pts, k=7):
    dists, idxs = fire_tree.query(new_pts, k=k)
    dists = np.maximum(dists, 1e-9)
    weights = 1.0 / (dists ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    feats = np.einsum('nk,nkf->nf', weights,
                      fire_feats[idxs].reshape(len(new_pts), k, fire_feats.shape[1]))
    noise = 0.05 * fire_feats.std(axis=0)
    feats += np.random.normal(0, noise, feats.shape)
    return feats

# ─────────────────────────────────────────────
# 3. RANDOM PSEUDO-ABSENCE
# ─────────────────────────────────────────────
def generate_random_pa(n):
    lons = np.random.uniform(fire_pts[:,0].min(), fire_pts[:,0].max(), n)
    lats = np.random.uniform(fire_pts[:,1].min(), fire_pts[:,1].max(), n)
    return np.column_stack([lons, lats])

# ─────────────────────────────────────────────
# 4. HEURISTIC PSEUDO-ABSENCE (BP_V6 + CS_V5)
# ─────────────────────────────────────────────
def generate_bp_v6(n_pool=15000):
    print("  Generating BP_V6 background pool...")
    lon_min, lon_max = fire_pts[:,0].min()-0.3, fire_pts[:,0].max()+0.3
    lat_min, lat_max = fire_pts[:,1].min()-0.3, fire_pts[:,1].max()+0.3

    # Grid density weighting
    n_cells = 15
    lon_edges = np.linspace(lon_min, lon_max, n_cells+1)
    lat_edges = np.linspace(lat_min, lat_max, n_cells+1)
    lon_idx = np.clip(np.digitize(fire_pts[:,0], lon_edges)-1, 0, n_cells-1)
    lat_idx = np.clip(np.digitize(fire_pts[:,1], lat_edges)-1, 0, n_cells-1)
    density = np.zeros((n_cells, n_cells))
    for i,j in zip(lon_idx, lat_idx):
        density[i,j] += 1
    F_max = max(density.max(), 1)

    # Generate large candidate pool
    total_cands = n_pool * 8
    lons = np.random.uniform(lon_min, lon_max, total_cands)
    lats = np.random.uniform(lat_min, lat_max, total_cands)
    pts  = np.column_stack([lons, lats])

    # Min distance to any fire point
    min_dists, _ = fire_tree.query(pts, k=1)

    # Hard reject < D_MIN
    mask = min_dists >= D_MIN_DEG
    pts      = pts[mask]
    min_dists = min_dists[mask]

    # Hybrid probabilistic acceptance
    p_accept = 1.0 - np.exp(-(min_dists - D_MIN_DEG) / LAM_DEG)
    accept   = np.random.uniform(0, 1, len(pts)) < p_accept
    pts      = pts[accept]
    min_dists = min_dists[accept]

    # Density weighting: cells with more fires get proportionally more candidates
    # (use cell density as acceptance weight)
    cell_i = np.clip(np.digitize(pts[:,0], lon_edges)-1, 0, n_cells-1)
    cell_j = np.clip(np.digitize(pts[:,1], lat_edges)-1, 0, n_cells-1)
    cell_density = density[cell_i, cell_j] / F_max
    # Bias toward fire-dense cells
    w = 0.3 + 0.7 * cell_density
    w /= w.sum()
    if len(pts) > n_pool:
        chosen = np.random.choice(len(pts), n_pool, replace=False, p=w)
        pts = pts[chosen]

    print(f"  BP_V6 pool size: {len(pts)}")
    return pts

def select_cs_v5(pool, n_select, alpha=0.5):
    print("  Selecting CS_V5 control set...")
    if len(pool) < n_select:
        return pool

    # Distance to nearest fire
    d_fire, _ = fire_tree.query(pool, k=1)

    # Distance to centroid of fire
    centroid = fire_pts.mean(axis=0)
    d_centroid = np.sqrt(((pool - centroid)**2).sum(axis=1))

    # Composite score
    scores = d_fire / (1.0 + alpha * d_centroid / (d_centroid.mean() + 1e-9))

    # Regional tertile balancing
    lat_33 = np.percentile(fire_pts[:,1], 33)
    lat_66 = np.percentile(fire_pts[:,1], 66)
    regions = np.where(pool[:,1] > lat_66, 2,
              np.where(pool[:,1] > lat_33, 1, 0))

    n_per = n_select // 3
    selected = []
    for r in [0, 1, 2]:
        idx_r = np.where(regions == r)[0]
        if len(idx_r) == 0:
            continue
        top = idx_r[np.argsort(scores[idx_r])[-n_per:]]
        selected.extend(top.tolist())

    # Top-up from global
    sel_arr = np.array(selected)
    remaining = n_select - len(sel_arr)
    if remaining > 0:
        all_sorted = np.argsort(scores)[::-1]
        extras = [i for i in all_sorted if i not in set(sel_arr.tolist())][:remaining]
        sel_arr = np.concatenate([sel_arr, extras])

    result = pool[sel_arr[:n_select]]
    print(f"  CS_V5 selected: {len(result)}")
    return result

# ─────────────────────────────────────────────
# 5. SPATIAL EVALUATION
# ─────────────────────────────────────────────
def ripleys_k_sse(pts_test, r_values=None):
    """SSE of K-curve vs fire points."""
    if r_values is None:
        r_values = np.linspace(0.05, 0.8, 15)
    lon_r = fire_pts[:,0].max() - fire_pts[:,0].min()
    lat_r = fire_pts[:,1].max() - fire_pts[:,1].min()
    area  = lon_r * lat_r

    def k_curve(pts):
        tree = cKDTree(pts)
        lam  = len(pts) / area
        k = []
        for r in r_values:
            counts = np.array(tree.query_ball_point(pts, r, return_length=True))
            k.append((counts - 1).mean() / lam)
        return np.array(k)

    k_fire = k_curve(fire_pts)
    k_test = k_curve(pts_test)
    sse    = float(np.sum((k_test - k_fire)**2))
    return sse

def centroid_dist_deg(pts_test):
    return float(np.sqrt(((pts_test.mean(0) - fire_pts.mean(0))**2).sum()))

def grid_variance(pts, n_cells=10):
    lon_e = np.linspace(pts[:,0].min(), pts[:,0].max(), n_cells+1)
    lat_e = np.linspace(pts[:,1].min(), pts[:,1].max(), n_cells+1)
    counts = []
    for i in range(n_cells):
        for j in range(n_cells):
            c = ((pts[:,0]>=lon_e[i])&(pts[:,0]<lon_e[i+1])&
                 (pts[:,1]>=lat_e[j])&(pts[:,1]<lat_e[j+1])).sum()
            counts.append(c)
    return float(np.var(counts))

def nn_mean_dist(pts):
    t = cKDTree(pts)
    d, _ = t.query(pts, k=2)
    return float(d[:,1].mean())

# ─────────────────────────────────────────────
# 6. ML EVALUATION
# ─────────────────────────────────────────────
def spatial_block_folds(coords, n_folds=5):
    lat_b = pd.qcut(coords[:,1], n_folds, labels=False, duplicates='drop')
    lon_b = pd.qcut(coords[:,0], n_folds, labels=False, duplicates='drop')
    blocks = lat_b * n_folds + lon_b
    unique = np.unique(blocks)
    np.random.shuffle(unique)
    groups = np.array_split(unique, n_folds)
    fold_id = np.zeros(len(coords), dtype=int)
    for fi, grp in enumerate(groups):
        for b in grp:
            fold_id[blocks == b] = fi
    return fold_id

def run_ml(absence_pts, absence_feats, label):
    X = np.vstack([fire_feats, absence_feats])
    y = np.array([1]*len(fire_pts) + [0]*len(absence_pts))
    coords = np.vstack([fire_pts, absence_pts])
    fold_id = spatial_block_folds(coords)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM':          SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'KNN':          KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(n_estimators=100, use_label_encoder=False,
                                           eval_metric='logloss', verbosity=0,
                                           random_state=RANDOM_STATE)

    out = {}
    for name, clf in models.items():
        aucs, tsss = [], []
        for fold in range(5):
            tr, te = fold_id != fold, fold_id == fold
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]
            if name in ('SVM', 'KNN'):
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte, prob))
            pred = (prob >= 0.5).astype(int)
            tp = ((pred==1)&(yte==1)).sum(); tn = ((pred==0)&(yte==0)).sum()
            fp = ((pred==1)&(yte==0)).sum(); fn = ((pred==0)&(yte==1)).sum()
            sens = tp/(tp+fn+1e-9); spec = tn/(tn+fp+1e-9)
            tsss.append(sens+spec-1)
        out[name] = {'AUC': np.mean(aucs), 'AUC_std': np.std(aucs),
                     'TSS': np.mean(tsss), 'TSS_std': np.std(tsss),
                     'folds': aucs}
        print(f"    {name:14s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  TSS={np.mean(tsss):.4f}")
    return out

def bootstrap_delta(a_h, a_r, n=1000):
    diffs = [np.mean(np.random.choice(a_h,len(a_h),True)) -
             np.mean(np.random.choice(a_r,len(a_r),True)) for _ in range(n)]
    d = np.array(diffs)
    ci = (np.percentile(d,2.5), np.percentile(d,97.5))
    pval = 2*min((d<=0).mean(),(d>=0).mean())
    return float(np.mean(d)), ci, float(pval)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: GENERATING PSEUDO-ABSENCES")
print("="*60)

print("\n[Random PA]")
rand_pts = generate_random_pa(N_TARGET)
print(f"  Generated {len(rand_pts)} random points")

print("\n[Heuristic PA - BP_V6 + CS_V5]")
pool = generate_bp_v6(n_pool=12000)
heur_pts = select_cs_v5(pool, N_TARGET, alpha=0.5)

print("\n" + "="*60)
print("PHASE 2: SPATIAL EVALUATION")
print("="*60)

sse_h = ripleys_k_sse(heur_pts)
sse_r = ripleys_k_sse(rand_pts)
cd_h  = centroid_dist_deg(heur_pts)
cd_r  = centroid_dist_deg(rand_pts)
gv_h  = grid_variance(heur_pts)
gv_r  = grid_variance(rand_pts)
nn_h  = nn_mean_dist(heur_pts)
nn_r  = nn_mean_dist(rand_pts)
nn_f  = nn_mean_dist(fire_pts)

sp_df = pd.DataFrame({
    'Metric':         ["Ripley's K SSE (lower=better)", "Centroid Distance (deg)", "Grid Variance (lower=better)", "Mean NN Distance (deg)"],
    'Heuristic':      [round(sse_h,2), round(cd_h,4), round(gv_h,2), round(nn_h,4)],
    'Random':         [round(sse_r,2), round(cd_r,4), round(gv_r,2), round(nn_r,4)],
    'Fire_Reference': ['0.00', '0.0000', '—', round(nn_f,4)]
})
print(sp_df.to_string(index=False))

print("\n" + "="*60)
print("PHASE 3: ML EVALUATION")
print("="*60)

print("\nInterpolating features...")
heur_feats = interpolate_features(heur_pts)
rand_feats  = interpolate_features(rand_pts)

print("\n[Heuristic models]")
res_h = run_ml(heur_pts, heur_feats, 'Heuristic')
print("\n[Random models]")
res_r = run_ml(rand_pts, rand_feats, 'Random')

print("\n" + "="*60)
print("COMPARISON TABLE")
print("="*60)
rows = []
for m in res_h:
    auc_h, auc_r = res_h[m]['AUC'], res_r[m]['AUC']
    tss_h, tss_r = res_h[m]['TSS'], res_r[m]['TSS']
    delta, ci, pval = bootstrap_delta(res_h[m]['folds'], res_r[m]['folds'])
    winner = "Heuristic✓" if delta>0 and pval<0.05 else ("Random✓" if delta<0 and pval<0.05 else "Tie")
    rows.append({'Model':m,
                 'AUC_Heuristic':round(auc_h,4), 'AUC_Heuristic_std':round(res_h[m]['AUC_std'],4),
                 'AUC_Random':round(auc_r,4),     'AUC_Random_std':round(res_r[m]['AUC_std'],4),
                 'ΔAUC':round(delta,4), 'CI_lo':round(ci[0],4), 'CI_hi':round(ci[1],4),
                 'p_value':round(pval,4),
                 'TSS_Heuristic':round(tss_h,4), 'TSS_Random':round(tss_r,4),
                 'Winner':winner})

ml_df = pd.DataFrame(rows)
print(ml_df[['Model','AUC_Heuristic','AUC_Random','ΔAUC','p_value','TSS_Heuristic','TSS_Random','Winner']].to_string(index=False))

# Save
sp_df.to_csv('../../outputs/spatial_results.csv', index=False)
ml_df.to_csv('../../outputs/ml_results.csv', index=False)
pd.DataFrame(heur_pts, columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/heuristic_pseudo_absences.csv', index=False)
pd.DataFrame(rand_pts, columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/random_pseudo_absences.csv', index=False)

print("\n✓ All outputs saved.")
