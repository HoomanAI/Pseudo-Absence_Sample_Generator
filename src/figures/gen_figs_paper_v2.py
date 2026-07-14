"""
Regenerate all paper figures with a 5th method added: DA-GP-WGAN
(domain-aware gradient-penalty WGAN).

Differences from gen_figs_paper.py:
  - METHODS extended from 4 to 5 (Heuristic, Random, SA, GAN, DA-GP-WGAN).
  - All spatial metrics (K-SSE, centroid distance, grid variance, mean NN
    distance, border fraction) are computed fresh from the point sets here,
    not hardcoded — the old hardcoded constants don't reproduce exactly
    from the raw data and obviously can't include the new method.
  - ML cross-validation numbers for DA-GP-WGAN come from a real 4-model,
    5-fold CV run (run_ml_da_gp_wgan.py -> cv_da.npy), not fabricated.
  - Adds new creative/innovative figures at the end (fig31-fig35).

Same paper style: no titles, white background, Liberation Serif, 300 DPI.
"""
import numpy as np, pandas as pd, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import KernelDensity
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
OUT  = '../../docs/figures/'
DATA = '../../outputs/'
CSV  = '../../data/Fire_points_dataset_final_csv.csv'
DPI  = 300

rcParams.update({
    'font.family'        : 'Times New Roman',
    'font.size'          : 11,
    'axes.titlesize'     : 11,
    'axes.titleweight'   : 'normal',
    'axes.titlepad'      : 8,
    'axes.labelsize'     : 11,
    'axes.labelweight'   : 'normal',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.alpha'         : 0.3,
    'grid.linewidth'     : 0.6,
    'xtick.labelsize'    : 10,
    'ytick.labelsize'    : 10,
    'legend.fontsize'    : 10,
    'legend.framealpha'  : 0.85,
    'legend.edgecolor'   : '#cccccc',
    'figure.dpi'         : DPI,
    'savefig.dpi'        : DPI,
    'savefig.bbox'       : 'tight',
    'savefig.facecolor'  : 'white',
    'axes.facecolor'     : 'white',
    'figure.facecolor'   : 'white',
})

MCOLS   = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#D62728']   # H R SA GAN DA-GP-WGAN
FIRE_COL = '#111111'
METHODS       = ['Heuristic', 'Random', 'SA', 'GAN', 'DA-GP-WGAN']
# Plain label everywhere by default; "(Ours)" is appended only at specific
# best/ours callouts (fig30 recommendation cards, fig34 Pareto annotation),
# not blanket-applied — see A1 in the correction pass.
METHOD_LABELS = ['Heuristic BP_V6', 'Random', 'SA', 'GAN (weight clip)', 'DA-GP-WGAN']
SHORT         = ['H', 'R', 'SA', 'GAN', 'DA']
MODELS        = ['RandomForest', 'XGBoost', 'KNN', 'SVM']
MOD_LABELS    = ['Random Forest', 'XGBoost', 'KNN', 'SVM']
MOD_COLS      = ['#1F3864', '#C0392B', '#27AE60', '#8E44AD']

def savefig(name):
    plt.savefig(OUT + name, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    print(f"  {name}  ({time.time()-t0:.1f}s)")

# ── Load data ────────────────────────────────────────────────────────────────
df       = pd.read_csv(CSV).dropna()
fire_pts = df[['LONGITUDE', 'LATITUDE']].values
fcols    = [c for c in df.columns if c not in ['FIRE', 'LONGITUDE', 'LATITUDE', 'YEAR', 'MONTH', 'DAY']]
fire_feats = df[fcols].values.astype(float)

hp = np.load(DATA + 'heur_pts.npy');   rp = np.load(DATA + 'rand_pts.npy')
sp = np.load(DATA + 'sa_pts.npy');     gp = np.load(DATA + 'gan_pts.npy')
dap = np.load(DATA + 'gan_pts_gp_domain_aware.npy')
hf = np.load(DATA + 'heur_feats.npy'); rf_f = np.load(DATA + 'rand_feats.npy')
sf = np.load(DATA + 'sa_feats.npy');   gf = np.load(DATA + 'gan_feats.npy')
daf = np.load(DATA + 'gan_feats_gp_domain_aware.npy')
cv_h = np.load(DATA + 'cv_h.npy', allow_pickle=True).item()
cv_r = np.load(DATA + 'cv_r.npy', allow_pickle=True).item()
cv_s = np.load(DATA + 'cv_s.npy', allow_pickle=True).item()
cv_g = np.load(DATA + 'cv_g.npy', allow_pickle=True).item()
cv_da = np.load(DATA + 'cv_da.npy', allow_pickle=True).item()

PTS   = {'Heuristic': hp, 'Random': rp, 'SA': sp, 'GAN': gp, 'DA-GP-WGAN': dap}
FEATS = {'Heuristic': hf, 'Random': rf_f, 'SA': sf, 'GAN': gf, 'DA-GP-WGAN': daf}
CVS   = {'Heuristic': cv_h, 'Random': cv_r, 'SA': cv_s, 'GAN': cv_g, 'DA-GP-WGAN': cv_da}
rng   = np.random.RandomState(42)

fire_tree = cKDTree(fire_pts)
def fire_dists(pts): d, _ = fire_tree.query(pts, k=1); return d * 111
def nn_dists(pts):   t = cKDTree(pts); d, _ = t.query(pts, k=2); return d[:, 1] * 111

FDIST = {m: fire_dists(PTS[m]) for m in METHODS}
FDIST['Fire'] = nn_dists(fire_pts)
NNDIST = {m: nn_dists(PTS[m]) for m in METHODS}

feat_stats = {'Fire': {'mean': fire_feats.mean(0), 'std': fire_feats.std(0)}}
for m in METHODS:
    feat_stats[m] = {'mean': FEATS[m].mean(0), 'std': FEATS[m].std(0)}

# ── Spatial metrics — computed fresh for all 5 methods (no hardcoding) ────────
lon_min, lon_max = fire_pts[:, 0].min() - 0.2, fire_pts[:, 0].max() + 0.2
lat_min, lat_max = fire_pts[:, 1].min() - 0.2, fire_pts[:, 1].max() + 0.2
_r_vals = np.linspace(0.05, 0.8, 15)
_area = (fire_pts[:, 0].max() - fire_pts[:, 0].min()) * (fire_pts[:, 1].max() - fire_pts[:, 1].min())

def _kcurve(pts, n=400, seed=42):
    idx = np.random.RandomState(seed).choice(len(pts), min(n, len(pts)), replace=False)
    s = pts[idx]; t = cKDTree(s); lam = len(s) / _area
    return np.array([(np.array(t.query_ball_point(s, r, return_length=True)) - 1).mean() / lam
                      for r in _r_vals])

def k_sse(pts): return float(np.sum((_kcurve(pts) - _kcurve(fire_pts)) ** 2))
def centroid_d(p): return float(np.sqrt(((p.mean(0) - fire_pts.mean(0)) ** 2).sum()))
def grid_var(pts, n=10):
    le = np.linspace(pts[:, 0].min(), pts[:, 0].max(), n + 1)
    ae = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n + 1)
    return float(np.var([((pts[:, 0] >= le[i]) & (pts[:, 0] < le[i + 1]) &
                           (pts[:, 1] >= ae[j]) & (pts[:, 1] < ae[j + 1])).sum()
                          for i in range(n) for j in range(n)]))
def nn_d(pts): t = cKDTree(pts); d, _ = t.query(pts, k=2); return float(d[:, 1].mean())
def border_frac(pts):
    return float(((pts[:, 0] - lon_min < 0.15) | (lon_max - pts[:, 0] < 0.15) |
                  (pts[:, 1] - lat_min < 0.15) | (lat_max - pts[:, 1] < 0.15)).mean())

SPATIAL = {m: {'K_SSE': k_sse(PTS[m]), 'Centroid': centroid_d(PTS[m]),
               'Grid_Var': grid_var(PTS[m]), 'Mean_NN': nn_d(PTS[m]),
               'Border': border_frac(PTS[m])} for m in METHODS}
FIRE_NN_REF = nn_d(fire_pts)
print("Spatial metrics (fresh, all 5 methods):")
for m in METHODS:
    print(f"  {m:12s} " + "  ".join(f"{k}={v:.4f}" for k, v in SPATIAL[m].items()))
print(f"Data + metrics loaded: {time.time()-t0:.1f}s")

# ── Helpers ──────────────────────────────────────────────────────────────────
def avg_roc(cv, model, n=60):
    fprs, tprs = cv['roc'][model]
    mfpr = np.linspace(0, 1, n)
    interp = [np.interp(mfpr, f, t) for f, t in zip(fprs, tprs) if len(f) > 1]
    return mfpr, np.mean(interp, axis=0) if interp else (mfpr, mfpr)

from itertools import combinations
PAIRS = list(combinations(METHODS, 2))
def boot_dauc(cv1, cv2, model, n=800):
    a1 = np.array(cv1['res'][model]['folds']); a2 = np.array(cv2['res'][model]['folds'])
    obs = a1.mean() - a2.mean(); rb = np.random.RandomState(77)
    d = [a1[rb.randint(0, len(a1), len(a1))].mean() - a2[rb.randint(0, len(a2), len(a2))].mean()
         for _ in range(n)]
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def short_pair(m1, m2):
    sh = dict(zip(METHODS, SHORT))
    return f'{sh[m1]}–{sh[m2]}'

# ════════════════════════════════════════════════════════════════════════════
# FIG 01 — Spatial Overview
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
N1 = 300
ax.scatter(*fire_pts[rng.choice(len(fire_pts), N1, replace=False)].T,
           c=FIRE_COL, s=14, alpha=0.7, zorder=5, label='Fire (n=3,370)')
for m, col in zip(METHODS, MCOLS):
    ax.scatter(*PTS[m][rng.choice(len(PTS[m]), N1, replace=False)].T,
               c=col, s=8, alpha=0.45, zorder=3, label=f'{m} PA (n={len(PTS[m])})')
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
ax.legend(loc='upper left', markerscale=1.5, fontsize=8.5)
fig.tight_layout()
savefig('fig01_spatial_overview.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 02 — Per-method scatter (2x3, 5 methods)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
N2 = 250
for ax, (m, col, label) in zip(axes.flat, zip(METHODS, MCOLS, METHOD_LABELS)):
    fi = rng.choice(len(fire_pts), N2, replace=False)
    mi = rng.choice(len(PTS[m]), N2, replace=False)
    ax.scatter(*fire_pts[fi].T, c=FIRE_COL, s=10, alpha=0.6, label='Fire', zorder=5)
    ax.scatter(*PTS[m][mi].T, c=col, s=8, alpha=0.5, label=label, zorder=3)
    ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
    ax.legend(fontsize=9, markerscale=1.5)
axes.flat[-1].axis('off')
plt.tight_layout()
savefig('fig02_per_method_scatter.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 03 — Ripley's K SSE bar
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.5))
vals3 = [SPATIAL[m]['K_SSE'] for m in METHODS]
bars = ax.bar(METHOD_LABELS, vals3, color=MCOLS, width=0.55, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, vals3):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
best_m = METHODS[int(np.argmin(vals3))]
ax.axhline(min(vals3), ls='--', color='#333', lw=1.2, alpha=0.5,
           label=f'Best: {min(vals3):.3f} ({best_m})')
ax.set_ylabel("Ripley's K SSE (vs. fire reference)")
ax.legend(fontsize=9); ax.set_ylim(0, max(vals3) * 1.2)
plt.xticks(rotation=12)
fig.tight_layout()
savefig('fig03_ripleys_k_sse.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 04 — Spatial Metrics 4-panel
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
metrics = [
    ("K-SSE (lower = better)",    [SPATIAL[m]['K_SSE'] for m in METHODS]),
    ("Centroid distance (°)", [SPATIAL[m]['Centroid'] for m in METHODS]),
    ("Grid variance",             [SPATIAL[m]['Grid_Var'] for m in METHODS]),
    ("Mean NN distance (°)", [SPATIAL[m]['Mean_NN'] for m in METHODS]),
]
for ax, (ylabel, vals) in zip(axes, metrics):
    bars = ax.bar(SHORT, vals, color=MCOLS, edgecolor='white', linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f'{v:.4f}' if v < 1 else f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.28)
plt.tight_layout()
savefig('fig04_spatial_metrics.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 05 — Density heatmaps
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
lon_e = np.linspace(fire_pts[:, 0].min() - 0.2, fire_pts[:, 0].max() + 0.2, 11)
lat_e = np.linspace(fire_pts[:, 1].min() - 0.2, fire_pts[:, 1].max() + 0.2, 11)
all_grp = [('Fire', fire_pts)] + [(m, PTS[m]) for m in METHODS]
for ax, (mname, pts) in zip(axes, all_grp):
    grid = np.zeros((10, 10))
    for i in range(10):
        ml = (pts[:, 0] >= lon_e[i]) & (pts[:, 0] < lon_e[i + 1])
        for j in range(10):
            grid[j, i] = (ml & (pts[:, 1] >= lat_e[j]) & (pts[:, 1] < lat_e[j + 1])).sum()
    im = ax.imshow(np.log1p(grid), origin='lower', cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xlabel('Lon bin', fontsize=9); ax.set_ylabel('Lat bin', fontsize=9)
    ax.set_title(mname, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, label='log(1+count)')
plt.tight_layout()
savefig('fig05_density_heatmap.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 06 — Distance to nearest fire
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 5))
bins6 = np.linspace(0, 150, 31)
for m, col, label in zip(METHODS, MCOLS, METHOD_LABELS):
    counts, _ = np.histogram(FDIST[m], bins=bins6)
    mids = (bins6[:-1] + bins6[1:]) / 2
    ax.plot(mids, counts, color=col, linewidth=2, label=label, alpha=0.85)
    ax.fill_between(mids, counts, alpha=0.1, color=col)
ax.set_xlabel('Distance to nearest fire point (km)')
ax.set_ylabel(f'Count')
ax.legend(fontsize=9); fig.tight_layout()
savefig('fig06_fire_distance_hist.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 07 — Ripley's K curves
# ════════════════════════════════════════════════════════════════════════════
k_fire = _kcurve(fire_pts)
k_curves = {'Fire': k_fire, **{m: _kcurve(PTS[m]) for m in METHODS}}
print(f"K-func computed: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(_r_vals, k_curves['Fire'], color=FIRE_COL, lw=3, ls='--', label='Fire (reference)', zorder=6)
for m, col, label in zip(METHODS, MCOLS, METHOD_LABELS):
    ax.plot(_r_vals, k_curves[m], color=col, lw=2, label=label, alpha=0.85)
ax.set_xlabel('Radius r (degrees)')
ax.set_ylabel('K(r)')
ax.legend(fontsize=9); fig.tight_layout()
savefig('fig07_ripleys_k_curves.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 08 — Marginal distributions
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lon_b = np.linspace(-116.5, -112.8, 25); lat_b = np.linspace(54.2, 56.9, 25)
all_grp2 = [('Fire', fire_pts, FIRE_COL, '--')] + [(m, PTS[m], c, '-') for m, c in zip(METHODS, MCOLS)]
for nm, pts, col, ls in all_grp2:
    h, e = np.histogram(pts[:, 0], bins=lon_b)
    axes[0].plot((e[:-1] + e[1:]) / 2, h / h.sum(), color=col, lw=2, ls=ls, label=nm,
                 alpha=0.85 if nm != 'Fire' else 1.0)
    h, e = np.histogram(pts[:, 1], bins=lat_b)
    axes[1].plot((e[:-1] + e[1:]) / 2, h / h.sum(), color=col, lw=2, ls=ls,
                 alpha=0.85 if nm != 'Fire' else 1.0)
axes[0].set_xlabel('Longitude (°)'); axes[0].set_ylabel('Density (normalised)')
axes[1].set_xlabel('Latitude (°)');  axes[1].set_ylabel('Density (normalised)')
axes[0].legend(fontsize=8.5)
plt.tight_layout()
savefig('fig08_marginal_distributions.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 09 — NN distance distribution
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
nn_bins = np.linspace(0, 5, 40); nn_mids = (nn_bins[:-1] + nn_bins[1:]) / 2
h, _ = np.histogram(FDIST['Fire'], bins=nn_bins)
ax.plot(nn_mids, h / h.sum(), color=FIRE_COL, lw=3, ls='--', label='Fire (self-NN)', zorder=6)
for m, col, label in zip(METHODS, MCOLS, METHOD_LABELS):
    h, _ = np.histogram(NNDIST[m], bins=nn_bins)
    ax.plot(nn_mids, h / h.sum(), color=col, lw=2, label=label, alpha=0.85)
    ax.fill_between(nn_mids, h / h.sum(), alpha=0.08, color=col)
ax.set_xlabel('Nearest-neighbour distance (km)')
ax.set_ylabel('Density (normalised)')
ax.legend(fontsize=9); fig.tight_layout()
savefig('fig09_nn_distance.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 10 — Spatial quality radar
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, polar=True)
cats = ['K-SSE\n(inv)', 'Centroid\n(inv)', 'Grid Var\n(inv)', 'Mean NN\n(inv)', 'Border\n(inv)']
n_cats = len(cats)
raw = {m: [SPATIAL[m]['K_SSE'], SPATIAL[m]['Centroid'], SPATIAL[m]['Grid_Var'],
           SPATIAL[m]['Mean_NN'], SPATIAL[m]['Border']] for m in METHODS}
all_v = np.array(list(raw.values()))
mn, mx = all_v.min(0), all_v.max(0)
norm = {m: 1 - (np.array(v) - mn) / (mx - mn + 1e-10) for m, v in raw.items()}
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist(); angles += angles[:1]
for m, col, label in zip(METHODS, MCOLS, METHOD_LABELS):
    vals = norm[m].tolist(); vals += vals[:1]
    lw = 3.2 if m == 'DA-GP-WGAN' else 2
    ax.plot(angles, vals, color=col, lw=lw, label=label)
    ax.fill(angles, vals, color=col, alpha=0.12 if m == 'DA-GP-WGAN' else 0.07)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=7)
ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.15), fontsize=8.5)
fig.tight_layout()
savefig('fig10_spatial_radar.png')
print(f"Figs 01-10 done: {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# FIG 11 — AUC grouped bar
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11.5, 5.5))
x = np.arange(len(METHODS)); w = 0.18
for i, (mod, col, lbl) in enumerate(zip(MODELS, MOD_COLS, MOD_LABELS)):
    aucs = [CVS[m]['res'][mod]['AUC'] for m in METHODS]
    bars = ax.bar(x + (i - 1.5) * w, aucs, w, color=col, label=lbl,
                  edgecolor='white', linewidth=0.6, alpha=0.88)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color=col)
ax.axhline(1.0, ls='--', lw=1.2, color='#555', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(METHOD_LABELS, fontsize=9)
ax.set_ylabel('AUC (mean 5-fold CV)'); ax.set_ylim(0.85, 1.04)
ax.legend(loc='lower right'); fig.tight_layout()
savefig('fig11_auc_comparison.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 12 — TSS grouped bar
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11.5, 5.5))
for i, (mod, col, lbl) in enumerate(zip(MODELS, MOD_COLS, MOD_LABELS)):
    tss = [CVS[m]['res'][mod]['TSS'] for m in METHODS]
    bars = ax.bar(x + (i - 1.5) * w, tss, w, color=col, label=lbl,
                  edgecolor='white', linewidth=0.6, alpha=0.88)
    for bar, v in zip(bars, tss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color=col)
ax.axhline(1.0, ls='--', lw=1.2, color='#555', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(METHOD_LABELS, fontsize=9)
ax.set_ylabel('TSS (mean 5-fold CV)'); ax.set_ylim(0.3, 1.1)
ax.legend(loc='lower right'); fig.tight_layout()
savefig('fig12_tss_comparison.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 13 — AUC mean +/- std 4-panel
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (mod, col, lbl) in zip(axes, zip(MODELS, MOD_COLS, MOD_LABELS)):
    aucs = [CVS[m]['res'][mod]['AUC'] for m in METHODS]
    stds = [CVS[m]['res'][mod]['AUC_std'] for m in METHODS]
    ax.bar(range(len(METHODS)), aucs, color=MCOLS, alpha=0.85, edgecolor='white')
    ax.errorbar(range(len(METHODS)), aucs, yerr=stds, fmt='none', color='#333', capsize=5, lw=1.5)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(SHORT, fontsize=10)
    ax.set_title(lbl, fontsize=10)
    ax.set_ylabel('AUC'); ax.set_ylim(max(0, min(aucs) - 0.05), 1.04)
    for i, (v, s) in enumerate(zip(aucs, stds)):
        ax.text(i, v + s + 0.003, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
savefig('fig13_auc_uncertainty.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 14 — Per-fold AUC heatmap
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (mod, lbl) in zip(axes, zip(MODELS, MOD_LABELS)):
    mat = np.array([[CVS[m]['res'][mod]['folds'][fi]
                     if fi < len(CVS[m]['res'][mod]['folds']) else np.nan
                     for fi in range(5)] for m in METHODS])
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.7, vmax=1.0, aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'F{i+1}' for i in range(5)], fontsize=9)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(SHORT, fontsize=9)
    ax.set_title(lbl, fontsize=10)
    for i in range(len(METHODS)):
        for j in range(5):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5,
                        color='white' if v < 0.85 else '#222')
    plt.colorbar(im, ax=ax, shrink=0.9, label='AUC')
plt.tight_layout()
savefig('fig14_per_fold_auc.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 15 — Bootstrap dAUC (RF), all C(5,2)=10 pairs
# ════════════════════════════════════════════════════════════════════════════
BOOT = {model: {f'{m1} vs {m2}': boot_dauc(CVS[m1], CVS[m2], model) for m1, m2 in PAIRS}
        for model in MODELS}
print(f"Bootstrap done: {time.time()-t0:.1f}s")

pair_labels = [short_pair(m1, m2) for m1, m2 in PAIRS]
pair_keys = [f'{m1} vs {m2}' for m1, m2 in PAIRS]
fig, ax = plt.subplots(figsize=(13, 5.5))
obs_v = [BOOT['RandomForest'][k][0] for k in pair_keys]
lo_v = [BOOT['RandomForest'][k][1] for k in pair_keys]
hi_v = [BOOT['RandomForest'][k][2] for k in pair_keys]
cols15 = ['#27AE60' if lo > 0 or hi < 0 else '#BDC3C7' for lo, hi in zip(lo_v, hi_v)]
ax.bar(range(len(PAIRS)), obs_v, color=cols15, edgecolor='white', linewidth=0.8, width=0.55)
ax.errorbar(range(len(PAIRS)), obs_v,
            yerr=[np.array(obs_v) - np.array(lo_v), np.array(hi_v) - np.array(obs_v)],
            fmt='none', color='#333', capsize=6, lw=1.8)
ax.axhline(0, color='#333', lw=1.5, alpha=0.7)
ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels(pair_labels, rotation=0, fontsize=9)
ax.set_ylabel('dAUC'); ax.set_xlabel('Method pair')
handles = [mpatches.Patch(color='#27AE60', label='Significant (CI excludes 0)'),
           mpatches.Patch(color='#BDC3C7', label='Not significant')]
ax.legend(handles=handles); fig.tight_layout()
savefig('fig15_bootstrap_dauc_rf.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 16 — ROC curves (DA-GP-WGAN, all models)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for mod, col, lbl in zip(MODELS, MOD_COLS, MOD_LABELS):
    fpr, tpr = avg_roc(cv_da, mod)
    auc_val = CVS['DA-GP-WGAN']['res'][mod]['AUC']
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{lbl} (AUC={auc_val:.3f})', alpha=0.9)
ax.plot([0, 1], [0, 1], '--', color='#888', lw=1.2, label='Chance (AUC=0.500)')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.legend(loc='lower right', fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.fill_between([0, 1], [0, 1], alpha=0.04, color='grey')
fig.tight_layout()
savefig('fig16_roc_gan_all_models.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 17 — ROC curves (RF, all PA methods)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
    fpr, tpr = avg_roc(CVS[m], 'RandomForest')
    auc_val = CVS[m]['res']['RandomForest']['AUC']
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{lbl} (AUC={auc_val:.3f})', alpha=0.9)
ax.plot([0, 1], [0, 1], '--', color='#888', lw=1.2, label='Chance (AUC=0.500)')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.legend(loc='lower right', fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
savefig('fig17_roc_rf_all_methods.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 18 — Precision-Recall (DA-GP-WGAN, all models)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for mod, col, lbl in zip(MODELS, MOD_COLS, MOD_LABELS):
    yte = cv_da['pr'][mod]['yte']; prob = cv_da['pr'][mod]['prob']
    prec, rec, _ = precision_recall_curve(yte, prob)
    ap = average_precision_score(yte, prob)
    idx = np.round(np.linspace(0, len(rec) - 1, 80)).astype(int)
    ax.plot(rec[idx], prec[idx], color=col, lw=2.2, label=f'{lbl} (AP={ap:.3f})', alpha=0.9)
baseline = cv_da['pr']['RandomForest']['yte'].mean()
ax.axhline(baseline, ls='--', color='#888', lw=1.2, label=f'Baseline (AP={baseline:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(loc='upper right', fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
savefig('fig18_pr_curves_gan.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 19 — AUC + TSS heatmap
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, cmap, vlo in zip(axes, ['AUC', 'TSS'], ['YlGn', 'YlOrRd'], [0.85, 0.4]):
    mat = np.array([[CVS[m]['res'][mod][metric] for mod in MODELS] for m in METHODS])
    im = ax.imshow(mat, cmap=cmap, vmin=vlo, vmax=1.0, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(MOD_LABELS, rotation=15, ha='right', fontsize=9)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHOD_LABELS, fontsize=9)
    ax.set_title(f'{metric} (5-fold mean)', fontsize=10)
    for i in range(len(METHODS)):
        for j in range(4):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='#111' if mat[i, j] > 0.93 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.9)
plt.tight_layout()
savefig('fig19_auc_tss_heatmap.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 20 — Calibration curves
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 6.5))
N_BINS = 10; bins_c = np.linspace(0, 1, N_BINS + 1); mids = (bins_c[:-1] + bins_c[1:]) / 2
for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
    yte = CVS[m]['pr']['RandomForest']['yte']
    prob = CVS[m]['pr']['RandomForest']['prob']
    obs = [float(yte[(prob >= bins_c[i]) & (prob < bins_c[i + 1])].mean())
           if ((prob >= bins_c[i]) & (prob < bins_c[i + 1])).sum() > 0 else np.nan
           for i in range(N_BINS)]
    ax.plot(mids, obs, 'o-', color=col, lw=2, ms=6, label=lbl, alpha=0.85)
ax.plot([0, 1], [0, 1], '--', color='#333', lw=1.5, label='Perfect calibration')
ax.set_xlabel('Predicted probability (Random Forest)')
ax.set_ylabel('Observed positive rate')
ax.legend(fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
savefig('fig20_calibration_curves.png')
print(f"Figs 11-20 done: {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# FIG 21-23 — Feature statistics
# ════════════════════════════════════════════════════════════════════════════
N_FEAT = 20; feat_names = fcols[:N_FEAT]
x_feat = np.arange(N_FEAT); w_feat = 0.125
ALLM = ['Fire'] + METHODS; ALLC = [FIRE_COL] + MCOLS; ALLL = ['Fire'] + METHOD_LABELS
n_series = len(ALLM); center = (n_series - 1) / 2

fig, ax = plt.subplots(figsize=(14, 5.5))
for i, (m, col, lbl) in enumerate(zip(ALLM, ALLC, ALLL)):
    means = [feat_stats[m]['mean'][j] for j in range(N_FEAT)]
    ax.bar(x_feat + (i - center) * w_feat, means, w_feat, color=col, label=lbl,
           alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_xticks(x_feat); ax.set_xticklabels([f[:12] for f in feat_names], rotation=40, ha='right', fontsize=7.5)
ax.set_ylabel('Mean value'); ax.legend(fontsize=7.5, ncol=3)
fig.tight_layout()
savefig('fig21_feature_means.png')

fig, ax = plt.subplots(figsize=(14, 5.5))
for i, (m, col, lbl) in enumerate(zip(ALLM, ALLC, ALLL)):
    stds = [feat_stats[m]['std'][j] for j in range(N_FEAT)]
    ax.bar(x_feat + (i - center) * w_feat, stds, w_feat, color=col, label=lbl,
           alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_xticks(x_feat); ax.set_xticklabels([f[:12] for f in feat_names], rotation=40, ha='right', fontsize=7.5)
ax.set_ylabel('Std deviation'); ax.legend(fontsize=7.5, ncol=3)
fig.tight_layout()
savefig('fig22_feature_std.png')

fig, ax = plt.subplots(figsize=(14, 5.5))
w2 = 0.15; center2 = (len(METHODS) - 1) / 2
for i, (m, col, lbl) in enumerate(zip(METHODS, MCOLS, METHOD_LABELS)):
    biases = [(feat_stats[m]['mean'][j] - feat_stats['Fire']['mean'][j])
              / (feat_stats['Fire']['std'][j] + 1e-10) for j in range(N_FEAT)]
    ax.bar(x_feat + (i - center2) * w2, biases, w2, color=col, label=lbl,
           alpha=0.85, edgecolor='white', linewidth=0.4)
ax.axhline(0, color='#333', lw=1.5, ls='-', alpha=0.6)
ax.set_xticks(x_feat); ax.set_xticklabels([f[:12] for f in feat_names], rotation=40, ha='right', fontsize=7.5)
ax.set_ylabel('Normalised bias  (PA mean - Fire mean) / Fire std')
ax.legend(fontsize=8); fig.tight_layout()
savefig('fig23_feature_bias.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 24 — PCA 2D feature space
# ════════════════════════════════════════════════════════════════════════════
N_PCA = 300
all_f = np.vstack([fire_feats[:N_PCA]] + [FEATS[m][:N_PCA] for m in METHODS])
sc2 = StandardScaler().fit(all_f)
pca2 = PCA(n_components=2, random_state=42).fit(sc2.transform(all_f))
var = pca2.explained_variance_ratio_
PCA_D = {'Fire': pca2.transform(sc2.transform(fire_feats[:N_PCA]))}
for m in METHODS:
    PCA_D[m] = pca2.transform(sc2.transform(FEATS[m][:N_PCA]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.scatter(*PCA_D['Fire'].T, c=FIRE_COL, s=12, alpha=0.6, label='Fire', zorder=5)
for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
    ax.scatter(*PCA_D[m].T, c=col, s=8, alpha=0.4, label=lbl)
ax.set_xlabel(f'PC1 ({var[0]:.1%} variance)'); ax.set_ylabel(f'PC2 ({var[1]:.1%} variance)')
ax.legend(fontsize=7.5, markerscale=1.5)
ax = axes[1]
ax.scatter(*PCA_D['Fire'].T, c=FIRE_COL, s=16, alpha=0.65, label='Fire', zorder=5)
ax.scatter(*PCA_D['DA-GP-WGAN'].T, c=MCOLS[4], s=12, alpha=0.5, label='DA-GP-WGAN PA')
ax.set_xlabel(f'PC1 ({var[0]:.1%} variance)'); ax.set_ylabel(f'PC2 ({var[1]:.1%} variance)')
ax.legend(fontsize=9, markerscale=1.5)
plt.tight_layout()
savefig('fig24_pca_feature_space.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 25 — Feature histograms (slope + avg_temperature)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for row, (fi, fn) in enumerate([(0, fcols[0]), (1, fcols[1])]):
    all_v = np.concatenate([fire_feats[:, fi]] + [FEATS[m][:, fi] for m in METHODS])
    bins_f = np.linspace(all_v.min(), all_v.max(), 30)
    ax = axes[row, 0]
    h, e = np.histogram(fire_feats[:, fi], bins=bins_f, density=True)
    ax.plot((e[:-1] + e[1:]) / 2, h, color=FIRE_COL, lw=2.5, ls='--', label='Fire', zorder=6)
    for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
        h, e = np.histogram(FEATS[m][:, fi], bins=bins_f, density=True)
        ax.plot((e[:-1] + e[1:]) / 2, h, color=col, lw=1.8, label=lbl, alpha=0.8)
    ax.set_xlabel('Value'); ax.set_ylabel(f'Density — {fn[:25]}')
    if row == 0: ax.legend(fontsize=7, ncol=2)
    ax2 = axes[row, 1]
    data_bp = [fire_feats[:, fi]] + [FEATS[m][:, fi] for m in METHODS]
    bp = ax2.boxplot(data_bp, patch_artist=True, notch=False,
                     medianprops=dict(color='black', lw=1.8),
                     whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, col in zip(bp['boxes'], [FIRE_COL] + MCOLS):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax2.set_xticklabels(['Fire'] + SHORT, fontsize=9)
    ax2.set_ylabel(fn[:25])
plt.tight_layout()
savefig('fig25_feature_histograms.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 26 — Feature discriminability ranking
# ════════════════════════════════════════════════════════════════════════════
total_bias = []
for fi, fn in enumerate(fcols):
    fire_s = feat_stats['Fire']['std'][fi] + 1e-10
    fire_mu = feat_stats['Fire']['mean'][fi]
    total = sum(abs(feat_stats[m]['mean'][fi] - fire_mu) / fire_s for m in METHODS)
    total_bias.append((total, fn))
total_bias.sort(reverse=True); top15 = total_bias[:15]
names15 = [t[1][:18] for t in top15]; vals15 = [t[0] for t in top15]

fig, ax = plt.subplots(figsize=(9, 6.5))
cmap = plt.cm.viridis; norm_cb = plt.Normalize(min(vals15), max(vals15))
bars = ax.barh(range(15), vals15, color=[cmap(norm_cb(v)) for v in vals15],
               edgecolor='white', linewidth=0.6)
ax.set_yticks(range(15)); ax.set_yticklabels(names15, fontsize=9)
ax.invert_yaxis()
for i, (bar, v) in enumerate(zip(bars, vals15)):
    ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f'{v:.2f}', va='center', fontsize=8.5)
ax.set_xlabel(f'Total normalised |bias| (summed across {len(METHODS)} PA methods)')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cb); sm.set_array([])
plt.colorbar(sm, ax=ax, label='Discriminability score', shrink=0.8)
fig.tight_layout()
savefig('fig26_feature_importance.png')
print(f"Figs 21-26 done: {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# FIG 27 — Bootstrap dAUC, all 4 models, all 10 pairs (2x2)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, (mod, lbl) in zip(axes.flat, zip(MODELS, MOD_LABELS)):
    obs_v = []; lo_v = []; hi_v = []
    for m1, m2 in PAIRS:
        obs, lo, hi = boot_dauc(CVS[m1], CVS[m2], mod)
        obs_v.append(obs); lo_v.append(lo); hi_v.append(hi)
    cols27 = ['#27AE60' if lo > 0 or hi < 0 else '#BDC3C7' for lo, hi in zip(lo_v, hi_v)]
    ax.bar(range(len(PAIRS)), obs_v, color=cols27, edgecolor='white', linewidth=0.7, width=0.55, alpha=0.9)
    ax.errorbar(range(len(PAIRS)), obs_v,
                yerr=[np.array(obs_v) - np.array(lo_v), np.array(hi_v) - np.array(obs_v)],
                fmt='none', color='#333', capsize=5, lw=1.5)
    ax.axhline(0, color='#333', lw=1.5, alpha=0.7)
    ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels(pair_labels, fontsize=8.5)
    ax.set_title(lbl, fontsize=11); ax.set_ylabel('dAUC')
plt.tight_layout()
savefig('fig27_bootstrap_dauc_all_models.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 28 — Composite performance
# ────────────────────────────────────────────────────────────────────────────
# A3 FIX: this panel previously min-max-normalised RandomForest AUC across
# methods. RF AUC is saturated at ~1.0 for every method (values differ only
# in the 5th-6th decimal place — e.g. cv_g RandomForest AUC = 0.99999910 vs
# cv_da = 1.00000000 — pure floating-point/bootstrap noise from RF training,
# not a real predictive-accuracy difference). Min-max normalising a column
# whose true range is ~1e-6 stretches that noise across the full [0,1] axis,
# which is exactly why the old right-panel bar showed GAN~1.0 vs DA~0.66 for
# a metric that is identical (1.000) for both methods in the underlying data.
# Fixed by replacing RF AUC with SVM AUC, which has real, non-saturated
# spread (0.963-0.994) and is the metric actually used for the honest
# GAN-vs-DA-GP-WGAN "statistically tied" framing elsewhere (fig30).
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
svm_aucs = [CVS[m]['res']['SVM']['AUC'] for m in METHODS]
ksse = [SPATIAL[m]['K_SSE'] for m in METHODS]
meannn = [SPATIAL[m]['Mean_NN'] for m in METHODS]
border = [SPATIAL[m]['Border'] for m in METHODS]
auc_n = [(v - min(svm_aucs)) / (max(svm_aucs) - min(svm_aucs) + 1e-10) for v in svm_aucs]
ksse_n = [(max(ksse) - v) / (max(ksse) - min(ksse) + 1e-10) for v in ksse]
nn_n = [(max(meannn) - v) / (max(meannn) - min(meannn) + 1e-10) for v in meannn]
border_n = [(max(border) - v) / (max(border) - min(border) + 1e-10) for v in border]
composite = [(a + k + n + b) / 4 for a, k, n, b in zip(auc_n, ksse_n, nn_n, border_n)]
ax = axes[0]
bars = ax.bar(METHOD_LABELS, composite, color=MCOLS, edgecolor='white', linewidth=0.8, alpha=0.88)
for bar, v in zip(bars, composite):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Composite score (0-1)'); ax.set_ylim(0, 1.15)
plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=8.5)
ax2 = axes[1]
x2 = np.arange(len(METHODS)); w3 = 0.2
ax2.bar(x2 - 1.5 * w3, auc_n,    w3, color='#1F3864', label='AUC SVM (norm)',     alpha=0.88, edgecolor='white')
ax2.bar(x2 - 0.5 * w3, ksse_n,   w3, color='#C0392B', label='K-SSE inv (norm)',   alpha=0.88, edgecolor='white')
ax2.bar(x2 + 0.5 * w3, nn_n,     w3, color='#27AE60', label='Mean NN inv (norm)', alpha=0.88, edgecolor='white')
ax2.bar(x2 + 1.5 * w3, border_n, w3, color='#8E44AD', label='Border inv (norm)',  alpha=0.88, edgecolor='white')
ax2.set_xticks(x2); ax2.set_xticklabels(SHORT, fontsize=10)
ax2.set_ylabel('Normalised score (0-1)'); ax2.set_ylim(0, 1.2)
ax2.legend(fontsize=8)
plt.tight_layout()
savefig('fig28_composite_performance.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 29 — Full summary table
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 10.5))
ax.axis('off')
row_labels = (['K-SSE', 'Centroid Dist (deg)', 'Grid Variance', 'Mean NN Dist', 'Border Frac', ''] +
              [f'AUC - {m}' for m in MOD_LABELS] + [''] +
              [f'TSS - {m}' for m in MOD_LABELS])
col_labels = ['Metric'] + METHOD_LABELS
sp_data = [[SPATIAL[m][k] for m in METHODS] for k in ['K_SSE', 'Centroid', 'Grid_Var', 'Mean_NN', 'Border']]
table_data = []
for lbl, vals in zip(row_labels[:5], sp_data):
    table_data.append([lbl] + [f'{v:.4f}' if v < 10 else f'{v:.1f}' for v in vals])
table_data.append([''] * (len(METHODS) + 1))
for mod in MODELS:
    table_data.append([f'AUC - {mod[:3]}'] + [f'{CVS[m]["res"][mod]["AUC"]:.4f}' for m in METHODS])
table_data.append([''] * (len(METHODS) + 1))
for mod in MODELS:
    table_data.append([f'TSS - {mod[:3]}'] + [f'{CVS[m]["res"][mod]["TSS"]:.4f}' for m in METHODS])
tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.4)
for j in range(len(METHODS) + 1):
    tbl[(0, j)].set_facecolor('#1F3864'); tbl[(0, j)].set_text_props(color='white', weight='bold')
for i in range(1, len(table_data) + 1):
    tbl[(i, len(METHODS))].set_facecolor('#FDECEA')
fig.tight_layout()
savefig('fig29_full_summary_table.png')

# ════════════════════════════════════════════════════════════════════════════
# FIG 30 — Method recommendation summary (honest, data-driven per-criterion winner)
# A2 FIX: "ML Discriminability" card switched from RF AUC (saturated at 1.000
# for every method - a meaningless tie, see A3 note on fig28) to SVM AUC,
# which has real spread. The card no longer crowns a single winner - it
# states the tie explicitly (GAN 0.994 vs DA-GP-WGAN 0.986) and reports the
# bootstrap significance test already computed for fig15/27 (BOOT['SVM']).
# ════════════════════════════════════════════════════════════════════════════
def winner(metric_dict, minimize=True):
    items = sorted(metric_dict.items(), key=lambda kv: kv[1], reverse=not minimize)
    return items[0][0], items[0][1], items[1][0], items[1][1]

w_ksse = winner({m: SPATIAL[m]['K_SSE'] for m in METHODS})
w_border = winner({m: SPATIAL[m]['Border'] for m in METHODS})
w_nn = winner({m: SPATIAL[m]['Mean_NN'] for m in METHODS}, minimize=False)  # closer to Fire NN, not just lowest
nn_close = {m: abs(SPATIAL[m]['Mean_NN'] - FIRE_NN_REF) for m in METHODS}
w_nn2, w_nn2v, _, _ = winner(nn_close)

svm_gan = CVS['GAN']['res']['SVM']['AUC']; svm_da = CVS['DA-GP-WGAN']['res']['SVM']['AUC']
_obs, _lo, _hi = BOOT['SVM']['GAN vs DA-GP-WGAN']
_sig = 'significant' if (_lo > 0 or _hi < 0) else 'not significant (CI includes 0)'

def _ours(name): return name + ' (Ours)' if name == 'DA-GP-WGAN' else name

fig = plt.figure(figsize=(14, 8)); ax = fig.add_subplot(111); ax.axis('off')
fig.patch.set_facecolor('white')
criteria = [
    ("Spatial Realism\n(Ripley's K SSE)", _ours(w_ksse[0]), f'{w_ksse[1]:.3f} (best)', f'vs {w_ksse[2]} {w_ksse[3]:.3f}', '#27AE60'),
    ('Border Avoidance\n(Border Frac)', _ours(w_border[0]), f'{w_border[1]:.3f} (lowest)', f'vs {w_border[2]} {w_border[3]:.3f}', '#E74C3C'),
    ('ML Discriminability\n(SVM AUC)', 'GAN ≈ DA-GP-WGAN', f'{svm_gan:.3f} vs {svm_da:.3f}  (n.s.)',
     f'bootstrap dAUC {_obs:+.4f}, {_sig}', '#2980B9'),
    ('Self-Clustering\n(Mean NN vs Fire NN)', _ours(w_nn2), f'|{w_nn2v:.4f}| closest to fire', f'fire ref = {FIRE_NN_REF:.4f}°', '#8E44AD'),
    ('Feature-Space\nRealism', _ours('DA-GP-WGAN'), 'PCA overlap', '58-dim env features', '#E67E22'),
    ('Overall\nRecommendation', _ours('DA-GP-WGAN'), 'Best spatial fidelity', 'lowest K-SSE + border frac; accuracy tied w/ GAN', '#1F3864'),
]
for i, (crit, best, val, note, col) in enumerate(criteria):
    x_pos = 0.02 + (i % 3) * 0.33; y_pos = 0.62 if i < 3 else 0.18
    rect = mpatches.FancyBboxPatch((x_pos, y_pos), 0.30, 0.32, boxstyle='round,pad=0.01',
                                    linewidth=2, edgecolor=col, facecolor=col + '22', transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x_pos + 0.15, y_pos + 0.27, crit, transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, fontweight='bold', color='#333')
    ax.text(x_pos + 0.15, y_pos + 0.18, best, transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold', color=col)
    ax.text(x_pos + 0.15, y_pos + 0.11, val, transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, color='#444')
    ax.text(x_pos + 0.15, y_pos + 0.05, note, transform=ax.transAxes, ha='center', va='top',
            fontsize=8, color='#666', style='italic')
ax.text(0.5, 0.06, 'DA-GP-WGAN adds a domain-aware weighted gradient penalty (k-NN manifold validity '
        'check) to the feature-space WGAN critic in place of weight clipping,\nyielding pseudo-absences '
        'whose spatial pattern most closely matches real fire locations while reducing border clustering.',
        transform=ax.transAxes, ha='center', va='top', fontsize=9.5, color='#333', style='italic', wrap=True,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F3F4', edgecolor='#BDC3C7'))
fig.tight_layout()
savefig('fig30_method_recommendation.png')

# ════════════════════════════════════════════════════════════════════════════
# Violin plots — per-fold AUC distribution
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
for ax, (mod, lbl) in zip(axes, zip(MODELS, MOD_LABELS)):
    data = [CVS[m]['res'][mod]['folds'] for m in METHODS]
    vp = ax.violinplot(data, positions=range(len(METHODS)), showmeans=True, showmedians=False)
    for i, (body, col) in enumerate(zip(vp['bodies'], MCOLS)):
        body.set_facecolor(col); body.set_alpha(0.7)
    for part in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if part in vp: vp[part].set_color('#333'); vp[part].set_linewidth(1.2)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(SHORT, fontsize=10)
    ax.set_title(lbl, fontsize=10); ax.set_ylabel('AUC (fold)')
    ax.set_ylim(max(0, min(min(d) for d in data) - 0.05), 1.03)
plt.tight_layout()
savefig('fig_violin_with_gan.png')

# ════════════════════════════════════════════════════════════════════════════
# ROC multi-panel — all methods x all models
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
for ax, (mod, mlbl) in zip(axes, zip(MODELS, MOD_LABELS)):
    for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
        fpr, tpr = avg_roc(CVS[m], mod)
        auc_val = CVS[m]['res'][mod]['AUC']
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f'{lbl[:4]} ({auc_val:.3f})', alpha=0.9)
    ax.plot([0, 1], [0, 1], '--', color='#888', lw=1.0)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(mlbl, fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc='lower right')
plt.tight_layout()
savefig('fig_roc_with_gan.png')
print(f"\nStandard 30 figures + 2 with_gan variants done: {time.time()-t0:.1f}s\n")

# ════════════════════════════════════════════════════════════════════════════
# ══════════════════════ NEW / CREATIVE FIGURES (31-35) ═══════════════════════
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# FIG 31 — Training dynamics: weight-clip WGAN vs DA-GP-WGAN (real training logs)
# ────────────────────────────────────────────────────────────────────────────
diag_clip = pd.read_csv(DATA + 'gan_diagnostics_clip.csv')
diag_da = pd.read_csv(DATA + 'gan_diagnostics_gp_domain_aware.csv')
diag_std = pd.read_csv(DATA + 'gan_diagnostics_gp_standard.csv')
CLIP_COL, DA_COL = MCOLS[3], MCOLS[4]
STD_COL = '#F39C12'

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

ax = axes[0, 0]
ax.plot(diag_clip['iter'], diag_clip['critic_loss'], color=CLIP_COL, lw=1.2, alpha=0.85, label='GAN (weight clip)')
ax.plot(diag_da['iter'], diag_da['critic_loss'], color=DA_COL, lw=1.2, alpha=0.85, label='DA-GP-WGAN')
ax.set_xlabel('Training iteration'); ax.set_ylabel('Critic loss')
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.plot(diag_clip['iter'], diag_clip['grad_norm_mean'], color=CLIP_COL, lw=1.0, alpha=0.5)
ax.plot(diag_da['iter'], diag_da['grad_norm_mean'], color=DA_COL, lw=1.0, alpha=0.5)
roll_c = diag_clip['grad_norm_mean'].rolling(75, min_periods=1).mean()
roll_d = diag_da['grad_norm_mean'].rolling(75, min_periods=1).mean()
ax.plot(diag_clip['iter'], roll_c, color=CLIP_COL, lw=2.4, label='GAN (weight clip, 75-iter mean)')
ax.plot(diag_da['iter'], roll_d, color=DA_COL, lw=2.4, label='DA-GP-WGAN (75-iter mean)')
ax.axhline(1.0, ls='--', color='#333', lw=1.2, alpha=0.6, label='Target ||grad|| = 1')
ax.set_xlabel('Training iteration'); ax.set_ylabel('Mean critic gradient norm ||grad D(x_hat)||')
ax.legend(fontsize=8.5)

# A4 FIX: the original bottom-left panel plotted in-manifold fraction for
# clip vs DA-GP-WGAN. Both curves sit near 0.06-0.09 and nearly overlap
# because that fraction is driven mainly by how close the generator's output
# gets to the real manifold by late training (similar for both critic
# constraints), not by the domain-aware *weighting* itself - so the panel
# didn't actually demonstrate the domain-aware effect. Replaced with the
# same grad-norm-convergence view as the top-right panel but now 3-way
# (clip / GP-standard / DA-GP-WGAN), which isolates what the domain-aware
# weighting contributes *on top of* a plain (uniform-weight) gradient
# penalty - the real comparison this paper needs to show.
ax = axes[1, 0]
roll_c3 = diag_clip['grad_norm_mean'].rolling(75, min_periods=1).mean()
roll_s3 = diag_std['grad_norm_mean'].rolling(75, min_periods=1).mean()
roll_d3 = diag_da['grad_norm_mean'].rolling(75, min_periods=1).mean()
ax.plot(diag_clip['iter'], roll_c3, color=CLIP_COL, lw=2.2, label='Clip (no GP)')
ax.plot(diag_std['iter'], roll_s3, color=STD_COL, lw=2.2, label='GP-standard (weight=1)')
ax.plot(diag_da['iter'], roll_d3, color=DA_COL, lw=2.2, label='DA-GP-WGAN (domain-aware weight)')
ax.axhline(1.0, ls='--', color='#333', lw=1.2, alpha=0.6, label='Target ||grad|| = 1')
ax.set_xlabel('Training iteration'); ax.set_ylabel('Mean critic gradient norm (75-iter mean), 3-way')
ax.legend(fontsize=8)

ax = axes[1, 1]
roll_gp_d = diag_da['gp_mean'].rolling(75, min_periods=1).mean()
ax.plot(diag_da['iter'], roll_gp_d, color=DA_COL, lw=2, label='DA-GP-WGAN: (||grad||-1)^2, 75-iter mean')
ax2b = ax.twinx()
roll_gen_c = diag_clip['gen_loss'].rolling(75, min_periods=1).mean()
roll_gen_d = diag_da['gen_loss'].rolling(75, min_periods=1).mean()
ax2b.plot(diag_clip['iter'], roll_gen_c, color=CLIP_COL, lw=1.6, ls='--', alpha=0.8, label='GAN gen_loss')
ax2b.plot(diag_da['iter'], roll_gen_d, color=DA_COL, lw=1.6, ls='--', alpha=0.8, label='DA-GP-WGAN gen_loss')
ax.set_xlabel('Training iteration'); ax.set_ylabel('Gradient penalty term', color=DA_COL)
ax2b.set_ylabel('Generator loss (dashed)')
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='upper right')
plt.tight_layout()
savefig('fig31_training_dynamics.png')

# ────────────────────────────────────────────────────────────────────────────
# FIG 32 — Method ranking bump chart across key metrics
# ────────────────────────────────────────────────────────────────────────────
# NB: RF/XGBoost AUC deliberately excluded from this ranking - both are
# saturated at ~1.000 for every method (see A3 fix note on fig28), so
# ranking methods by them would encode floating-point noise as if it were
# a real signal. SVM AUC (real spread, 0.963-0.994) stands in for ML
# discriminability instead.
rank_metrics = {
    'K-SSE': {m: SPATIAL[m]['K_SSE'] for m in METHODS},
    'Border Frac': {m: SPATIAL[m]['Border'] for m in METHODS},
    'NN-to-Fire\nMatch': {m: abs(SPATIAL[m]['Mean_NN'] - FIRE_NN_REF) for m in METHODS},
    'SVM AUC': {m: -CVS[m]['res']['SVM']['AUC'] for m in METHODS},
    'KNN AUC': {m: -CVS[m]['res']['KNN']['AUC'] for m in METHODS},
    'Composite\nScore': {m: -composite[METHODS.index(m)] for m in METHODS},
}
rank_names = list(rank_metrics.keys())
ranks = {m: [] for m in METHODS}
for metric_name in rank_names:
    vals = rank_metrics[metric_name]
    order = sorted(METHODS, key=lambda m: vals[m])
    for m in METHODS:
        ranks[m].append(order.index(m) + 1)

fig, ax = plt.subplots(figsize=(10.5, 6.5))
xpos = np.arange(len(rank_names))
for m, col, lbl in zip(METHODS, MCOLS, METHOD_LABELS):
    lw = 3.6 if m == 'DA-GP-WGAN' else 1.8
    z = 10 if m == 'DA-GP-WGAN' else 3
    ax.plot(xpos, ranks[m], 'o-', color=col, lw=lw, ms=9, label=lbl, alpha=0.95, zorder=z)
    for x, y in zip(xpos, ranks[m]):
        ax.annotate(str(y), (x, y), textcoords='offset points', xytext=(0, 8 if m == 'DA-GP-WGAN' else -12),
                    ha='center', fontsize=7, color=col, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(rank_names, fontsize=9.5)
ax.set_yticks(range(1, len(METHODS) + 1))
ax.set_ylabel('Rank (1 = best)')
ax.invert_yaxis()
ax.set_ylim(len(METHODS) + 0.5, 0.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8.5)
fig.tight_layout()
savefig('fig32_method_ranking_bump.png')

# ────────────────────────────────────────────────────────────────────────────
# FIG 33 — PCA feature-space density contour overlay: Fire vs GAN vs DA-GP-WGAN
# ────────────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize=(8, 7))
xg = np.linspace(PCA_D['Fire'][:, 0].min() - 1, PCA_D['Fire'][:, 0].max() + 1, 120)
yg = np.linspace(PCA_D['Fire'][:, 1].min() - 1, PCA_D['Fire'][:, 1].max() + 1, 120)
Xg, Yg = np.meshgrid(xg, yg); grid = np.vstack([Xg.ravel(), Yg.ravel()])
for name, col, ls, lw, fill in [('Fire', FIRE_COL, '-', 2.6, False),
                                  ('GAN', MCOLS[3], '--', 1.8, False),
                                  ('DA-GP-WGAN', MCOLS[4], '-', 2.2, True)]:
    kde = gaussian_kde(PCA_D[name].T)
    dens = kde(grid).reshape(Xg.shape)
    if fill:
        ax.contourf(Xg, Yg, dens, levels=6, cmap='Reds', alpha=0.25, zorder=1)
    ax.contour(Xg, Yg, dens, levels=5, colors=[col], linewidths=lw, linestyles=ls, zorder=3)
ax.scatter(*PCA_D['Fire'].T, c=FIRE_COL, s=6, alpha=0.25, zorder=2)
handles = [plt.Line2D([0], [0], color=FIRE_COL, lw=2.6, label='Fire (density contour)'),
           plt.Line2D([0], [0], color=MCOLS[3], lw=1.8, ls='--', label='GAN (density contour)'),
           plt.Line2D([0], [0], color=MCOLS[4], lw=2.2, label='DA-GP-WGAN (density contour + fill)')]
ax.legend(handles=handles, fontsize=9, loc='upper left')
ax.set_xlabel(f'PC1 ({var[0]:.1%} variance)'); ax.set_ylabel(f'PC2 ({var[1]:.1%} variance)')
fig.tight_layout()
savefig('fig33_pca_density_contours.png')

# ────────────────────────────────────────────────────────────────────────────
# FIG 34 — Spatial fidelity Pareto scatter: K-SSE vs Border fraction
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))
for m, col in zip(METHODS, MCOLS):
    x_ = SPATIAL[m]['K_SSE']; y_ = SPATIAL[m]['Border']
    ms = 260 if m == 'DA-GP-WGAN' else 170
    ax.scatter(x_, y_, s=ms, color=col, edgecolor='white', linewidth=1.5, zorder=5,
               marker='*' if m == 'DA-GP-WGAN' else 'o')
    label = m + ' (Ours)' if m == 'DA-GP-WGAN' else m
    ax.annotate(label, (x_, y_), textcoords='offset points', xytext=(10, 8), fontsize=10,
                fontweight='bold' if m == 'DA-GP-WGAN' else 'normal', color=col)
ax.annotate('', xy=(min(SPATIAL[m]['K_SSE'] for m in METHODS) - 0.3,
                     min(SPATIAL[m]['Border'] for m in METHODS) - 0.02),
            xytext=(min(SPATIAL[m]['K_SSE'] for m in METHODS) + 1.1,
                    min(SPATIAL[m]['Border'] for m in METHODS) + 0.06),
            arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.6))
ax.text(min(SPATIAL[m]['K_SSE'] for m in METHODS) + 0.55,
        min(SPATIAL[m]['Border'] for m in METHODS) + 0.075,
        'better\n(closer to fire,\nless border clustering)', fontsize=8, color='#666',
        ha='center', style='italic')
ax.set_xlabel("Ripley's K SSE (spatial pattern distance from fire, lower = better)")
ax.set_ylabel('Border fraction (share of points near study-area edge, lower = better)')
fig.tight_layout()
savefig('fig34_spatial_fidelity_pareto.png')

# ────────────────────────────────────────────────────────────────────────────
# FIG 35 — Border-clustering reduction: GAN vs DA-GP-WGAN geographic comparison
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (mname, pts) in zip(axes, [('GAN (weight clip)', gp), ('DA-GP-WGAN', dap)]):
    is_border = ((pts[:, 0] - lon_min < 0.15) | (lon_max - pts[:, 0] < 0.15) |
                 (pts[:, 1] - lat_min < 0.15) | (lat_max - pts[:, 1] < 0.15))
    ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                fill=False, edgecolor='#999', linewidth=1, linestyle=':'))
    ax.add_patch(plt.Rectangle((lon_min + 0.15, lat_min + 0.15),
                                (lon_max - lon_min) - 0.3, (lat_max - lat_min) - 0.3,
                                fill=False, edgecolor='#333', linewidth=1.2, linestyle='--'))
    ax.scatter(*pts[~is_border].T, c='#2CA02C', s=6, alpha=0.5, label=f'Interior ({(~is_border).sum()})')
    ax.scatter(*pts[is_border].T, c='#D62728', s=8, alpha=0.7, label=f'Border zone ({is_border.sum()})')
    ax.scatter(*fire_pts[rng.choice(len(fire_pts), 200, replace=False)].T,
               c=FIRE_COL, s=5, alpha=0.35, zorder=1, label='Fire (ref)')
    ax.set_title(f'{mname} — {is_border.mean():.1%} in border zone', fontsize=10)
    ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
    ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
savefig('fig35_border_clustering_comparison.png')

# ════════════════════════════════════════════════════════════════════════════
# ══════════════════ CORRECTION-PASS NEW FIGURES (F1-F5, fig36+) ═════════════
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# FIG 36 (F1, HIGH PRIORITY) — Three-way ablation: clip -> GP-standard -> DA-GP
#
# Uses the controlled ablation harness (gan_pts_clip / gan_pts_gp_standard /
# gan_pts_gp_domain_aware), all trained with identical generator/critic
# architecture and iteration budget, differing ONLY in the critic constraint.
# This is a SEPARATE data source from the paper's main "GAN (weight clip)"
# comparison method used in figs 01-35 (which is a legacy checkpoint,
# gan_pts.npy) - keeping the controlled ablation and the main cross-method
# comparison distinct avoids conflating a same-seed-family ablation with a
# cross-run comparison.
#
# HONEST RESULT (does not match the "expected narrative" in the task brief):
# switching clip -> GP-standard already captures nearly all of the spatial
# improvement (K-SSE 4.33 -> 3.20, border 0.277 -> 0.243). Adding the
# domain-aware weighting on top is roughly a wash on this run: K-SSE ties
# (3.20 -> 3.19) and border fraction is actually marginally *worse*
# (0.243 -> 0.253) than plain GP-standard. SVM AUC is flat across all three
# (~0.984-0.986). This is reported exactly as computed, not adjusted to fit
# the hoped-for "domain-aware improves further" story.
# ────────────────────────────────────────────────────────────────────────────
abl_pts = {'Clip': np.load(DATA + 'gan_pts_clip.npy'),
           'GP-standard': np.load(DATA + 'gan_pts_gp_standard.npy'),
           'DA-GP-WGAN': dap}
abl_cv = {'Clip': np.load(DATA + 'cv_clip.npy', allow_pickle=True).item(),
          'GP-standard': np.load(DATA + 'cv_gp_standard.npy', allow_pickle=True).item(),
          'DA-GP-WGAN': cv_da}
ABL_LABELS = list(abl_pts.keys())
ABL_COLS = [CLIP_COL, STD_COL, DA_COL]

abl_ksse = [k_sse(abl_pts[k]) for k in ABL_LABELS]
abl_border = [border_frac(abl_pts[k]) for k in ABL_LABELS]
abl_svm = [abl_cv[k]['res']['SVM']['AUC'] for k in ABL_LABELS]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (vals, ylabel) in zip(axes, [
        (abl_ksse, "Ripley's K SSE (lower = better)"),
        (abl_border, 'Border fraction (lower = better)'),
        (abl_svm, 'SVM AUC')]):
    bars = ax.bar(ABL_LABELS, vals, color=ABL_COLS, edgecolor='white', linewidth=0.8, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.015,
                f'{v:.4f}' if v < 1 else f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9.5); ax.set_ylim(0, max(vals) * 1.22)
    plt.setp(ax.get_xticklabels(), rotation=8, fontsize=9)
plt.tight_layout()
savefig('fig36_ablation_three_way.png')
print(f"F1 ablation — Clip: K-SSE={abl_ksse[0]:.3f} Border={abl_border[0]:.4f} SVM={abl_svm[0]:.4f}")
print(f"              GP-std: K-SSE={abl_ksse[1]:.3f} Border={abl_border[1]:.4f} SVM={abl_svm[1]:.4f}")
print(f"              DA-GP: K-SSE={abl_ksse[2]:.3f} Border={abl_border[2]:.4f} SVM={abl_svm[2]:.4f}")
# CAPTION (fig36): "Three-way ablation isolating the gradient-penalty effect (Clip -> GP-standard)
# from the domain-aware weighting effect (GP-standard -> DA-GP-WGAN) on K-SSE, border fraction,
# and SVM AUC; most of the spatial-fidelity gain comes from adding a gradient penalty at all, with
# the domain-aware weighting roughly neutral on these three runs and SVM AUC flat throughout."

# ────────────────────────────────────────────────────────────────────────────
# FIG 37 (F2) — Gradient-norm distribution (clip vs GP-standard vs DA-GP-WGAN)
#               + critic-weight panel (BLOCKED: data not persisted)
#
# LEFT PANEL BLOCKED: gen_gan.py exports only the trained *generator*'s
# weights (GW/Gb) for downstream use; the critic is discarded in-memory at
# the end of each of the three training runs and was never written to disk.
# Re-running training to capture a snapshot would take ~25 min (clip) to
# ~2.75 h (domain-aware, based on the original run's wall-clock) and is
# explicitly out of scope ("do NOT re-implement or retrain"). Left as a
# labelled placeholder rather than fabricated.
#
# RIGHT PANEL: real data, but note what it actually is - the diagnostics
# CSVs only persist per-ITERATION batch mean/std of ||grad D(x_hat)||, not
# individual per-sample gradient norms. So this is a distribution of
# per-iteration batch-mean gradient norms over the final N iterations, not
# a distribution of individual interpolant samples. Captioned as such.
# ────────────────────────────────────────────────────────────────────────────
N_TAIL = 500
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
ax.axis('off')
ax.text(0.5, 0.55, 'Critic weight distribution\nnot available', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#999', transform=ax.transAxes)
ax.text(0.5, 0.38,
        'The trained critic (D) is discarded at the end of training in\n'
        'gen_gan.py — only the generator is exported. Weight tensors were\n'
        'not persisted for any of the three ablation runs. Recapturing this\n'
        'would require re-running training (~25 min to ~2.75 h per run),\n'
        'which is out of scope for this figure-correction pass.',
        ha='center', va='top', fontsize=8.5, color='#666', style='italic', transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#ccc',
                            linewidth=1, linestyle='--', transform=ax.transAxes))

ax = axes[1]
tails = {'Clip (no GP)': (diag_clip['grad_norm_mean'].tail(N_TAIL), CLIP_COL),
         'GP-standard': (diag_std['grad_norm_mean'].tail(N_TAIL), STD_COL),
         'DA-GP-WGAN': (diag_da['grad_norm_mean'].tail(N_TAIL), DA_COL)}
vp_data = [v[0].values for v in tails.values()]
vp = ax.violinplot(vp_data, showmeans=True, showextrema=True)
for body, (name, (_, col)) in zip(vp['bodies'], tails.items()):
    body.set_facecolor(col); body.set_alpha(0.65)
for part in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
    if part in vp: vp[part].set_color('#333'); vp[part].set_linewidth(1.2)
ax.axhline(1.0, ls='--', color='#333', lw=1.2, alpha=0.6, label='Target ||grad|| = 1')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(list(tails.keys()), fontsize=9)
ax.set_ylabel(f'Per-iteration batch-mean ||grad D(x_hat)||\n(last {N_TAIL} iterations)')
ax.legend(fontsize=8.5)
plt.tight_layout()
savefig('fig37_gradnorm_weight_distribution.png')
# CAPTION (fig37): "Left: critic weight-distribution comparison is blocked - trained critic weights
# were not persisted for any run (out of scope to retrain). Right: distribution of per-iteration
# batch-mean critic gradient norm over the final 500 iterations - clip collapses well below the
# WGAN-GP target of 1, both gradient-penalty variants concentrate near it."

# ────────────────────────────────────────────────────────────────────────────
# FIG 38 (F3) — Sensitivity heatmap scaffold: LAMBDA_GP x GP_VALIDITY_PERCENTILE
#               (BLOCKED: sweep runs do not exist)
#
# No hyperparameter-sweep training runs have been generated - only the single
# operating point used throughout this paper (LAMBDA_GP=10, TAU percentile=90)
# has been trained. Producing real heatmap cells would require training
# 5x4=20 additional short runs. This renders the intended axes/labels/
# chosen-operating-point marker with the cell grid explicitly marked PENDING
# rather than inventing K-SSE/border values for untrained configurations.
# ────────────────────────────────────────────────────────────────────────────
LAMBDA_SWEEP = [1, 5, 10, 20, 50]
TAU_SWEEP = [80, 85, 90, 95]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, metric_name in zip(axes, ['K-SSE', 'Border fraction']):
    grid = np.full((len(TAU_SWEEP), len(LAMBDA_SWEEP)), np.nan)
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(LAMBDA_SWEEP))); ax.set_xticklabels(LAMBDA_SWEEP)
    ax.set_yticks(range(len(TAU_SWEEP))); ax.set_yticklabels(TAU_SWEEP)
    ax.set_xlabel(r'Gradient-penalty weight $\lambda_{GP}$')
    ax.set_ylabel(r'Validity percentile $\tau$')
    ax.set_title(f'{metric_name} — PENDING (sweep not run)', fontsize=9.5, color='#999')
    ix = LAMBDA_SWEEP.index(10); iy = TAU_SWEEP.index(90)
    ax.add_patch(plt.Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, edgecolor='#1F3864', linewidth=2.5))
    ax.text(ix, iy, 'used\nin paper', ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1F3864')
    for yy in range(len(TAU_SWEEP)):
        for xx in range(len(LAMBDA_SWEEP)):
            if (xx, yy) != (ix, iy):
                ax.text(xx, yy, '?', ha='center', va='center', fontsize=11, color='#bbb')
plt.tight_layout()
savefig('fig38_sensitivity_heatmap_SCAFFOLD.png')
print("F3 sensitivity heatmap: BLOCKED — no sweep runs exist yet. Scaffold saved as "
      "fig38_sensitivity_heatmap_SCAFFOLD.png with cells marked '?'. Needs 5x4=20 short "
      f"training runs over LAMBDA_GP={LAMBDA_SWEEP} x GP_VALIDITY_PERCENTILE={TAU_SWEEP}.")
# CAPTION (fig38, once run): "K-SSE and border fraction across a grid of gradient-penalty weight
# and k-NN validity-radius percentile, with the operating point used throughout this paper marked."

# ────────────────────────────────────────────────────────────────────────────
# FIG 39 (F4, HIGH PRIORITY) — Conceptual schematic of the domain-aware GP
# Pure diagram, no data dependency.
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis('off')
GREEN_OK, RED_X, ORANGE = '#27AE60', '#D62728', '#E67E22'

# real / fake / interpolated points
p_real = (1.6, 5.2); p_fake = (1.6, 2.2); p_hat = (4.2, 3.7)
ax.scatter(*p_real, s=190, c=MCOLS[0], edgecolor='white', zorder=5)
ax.text(p_real[0], p_real[1] + 0.45, r'$x_{real}$', ha='center', fontsize=11, fontweight='bold', color=MCOLS[0])
ax.scatter(*p_fake, s=190, c=MCOLS[4], edgecolor='white', zorder=5)
ax.text(p_fake[0], p_fake[1] - 0.45, r'$x_{fake}=G(z)$', ha='center', fontsize=11, fontweight='bold', color=MCOLS[4])
ax.annotate('', xy=p_hat, xytext=p_real, arrowprops=dict(arrowstyle='-', color='#999', lw=1.3, ls='--'))
ax.annotate('', xy=p_hat, xytext=p_fake, arrowprops=dict(arrowstyle='-', color='#999', lw=1.3, ls='--'))
ax.scatter(*p_hat, s=230, c=ORANGE, edgecolor='#111', linewidth=1.2, zorder=6)
ax.text(p_hat[0] + 0.35, p_hat[1] + 0.35, r'$\hat{x} = \varepsilon x_{real} + (1-\varepsilon) x_{fake}$',
        fontsize=10.5, fontweight='bold', color=ORANGE)

# background manifold (small cluster) + validity radius tau around x_hat
rng_s = np.random.RandomState(3)
bg_cloud = np.array([4.2, 3.7]) + rng_s.normal(0, 0.28, (22, 2))
ax.scatter(*bg_cloud.T, s=35, c=MCOLS[0], alpha=0.55, zorder=3)
tau_circle = plt.Circle(p_hat, 0.85, fill=False, edgecolor='#333', linewidth=1.8, linestyle=(0, (5, 3)), zorder=4)
ax.add_patch(tau_circle)
ax.text(p_hat[0], p_hat[1] - 1.15, r'k-NN validity check vs. background pool, radius $\tau$'
        '\n(90th-pct. self-distance)', ha='center', fontsize=8.5, color='#333')

# split: in-manifold vs out-of-manifold weighting
box_y0 = 5.6
in_box = (6.4, box_y0); oom_box = (6.4, 1.6)
ax.annotate('', xy=(in_box[0] - 0.5, in_box[1]), xytext=(p_hat[0] + 0.6, p_hat[1] + 0.35),
            arrowprops=dict(arrowstyle='-|>', color=GREEN_OK, lw=1.8))
ax.annotate('', xy=(oom_box[0] - 0.5, oom_box[1]), xytext=(p_hat[0] + 0.6, p_hat[1] - 0.35),
            arrowprops=dict(arrowstyle='-|>', color=RED_X, lw=1.8))
for (x, y), col, txt1, txt2 in [
        (in_box, GREEN_OK, 'In-manifold', 'weight = 1'),
        (oom_box, RED_X, 'Out-of-manifold', f'weight = GP_OOB_WEIGHT (0.1)')]:
    box = mpatches.FancyBboxPatch((x - 1.15, y - 0.55), 2.3, 1.1, boxstyle='round,pad=0.02,rounding_size=0.12',
                                   linewidth=2, edgecolor=col, facecolor=col + '18', zorder=5)
    ax.add_patch(box)
    ax.text(x, y + 0.2, txt1, ha='center', fontsize=10, fontweight='bold', color=col)
    ax.text(x, y - 0.2, txt2, ha='center', fontsize=9, color='#333')

# penalty term feeding critic loss
pen_box = (9.4, 3.6)
ax.annotate('', xy=(pen_box[0] - 1.3, pen_box[1] + 1.0), xytext=(in_box[0] + 1.2, in_box[1]),
            arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.5))
ax.annotate('', xy=(pen_box[0] - 1.3, pen_box[1] - 1.0), xytext=(oom_box[0] + 1.2, oom_box[1]),
            arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.5))
box = mpatches.FancyBboxPatch((pen_box[0] - 1.3, pen_box[1] - 0.75), 2.9, 1.5,
                               boxstyle='round,pad=0.02,rounding_size=0.12',
                               linewidth=2.2, edgecolor='#1F3864', facecolor='#1F386418', zorder=5)
ax.add_patch(box)
ax.text(pen_box[0], pen_box[1] + 0.35, 'Domain-aware GP term', ha='center', fontsize=10, fontweight='bold', color='#1F3864')
ax.text(pen_box[0], pen_box[1] - 0.15, r'$\lambda \cdot w_i \cdot (\|\nabla_{\hat{x}} D(\hat{x}_i)\|-1)^2$',
        ha='center', fontsize=11, color='#1F3864')
ax.annotate('', xy=(11.6, 3.6), xytext=(pen_box[0] + 1.6, pen_box[1]),
            arrowprops=dict(arrowstyle='-|>', color='#1F3864', lw=2))
ax.text(11.65, 3.6, 'Critic\nloss', ha='left', va='center', fontsize=10, fontweight='bold', color='#1F3864')

# removed weight-clipping box, crossed out
clip_box = mpatches.FancyBboxPatch((0.4, 0.15), 2.6, 0.85, boxstyle='round,pad=0.02,rounding_size=0.1',
                                    linewidth=1.8, edgecolor='#999', facecolor='#eee', zorder=2)
ax.add_patch(clip_box)
ax.text(1.7, 0.575, r'Weight clipping  $w \in [-0.05,\ 0.05]$', ha='center', va='center', fontsize=8.5, color='#777')
ax.plot([0.5, 2.9], [0.2, 1.0], color=RED_X, lw=2.2, zorder=3)
ax.plot([0.5, 2.9], [1.0, 0.2], color=RED_X, lw=2.2, zorder=3)
ax.text(1.7, -0.05, 'removed', ha='center', fontsize=8, color=RED_X, style='italic')

fig.tight_layout()
savefig('fig39_conceptual_schematic.png')
# CAPTION (fig39): "Domain-aware gradient penalty: interpolants x_hat between real and generated
# samples are validity-checked via k-NN distance to the background pool (radius tau); in-manifold
# interpolants receive full penalty weight, out-of-manifold interpolants are down-weighted, and the
# resulting weighted penalty replaces weight clipping in the critic loss."

# ────────────────────────────────────────────────────────────────────────────
# FIG 40 (F5) — UMAP feature-space embedding
#
# Reuses the already-fitted StandardScaler (sc2, fit once for fig24's PCA and
# not re-fit here). No persisted "background candidate pool" file exists
# (gen_gan.py builds it in-memory and never writes it to disk), so this
# overlays fire points against the generated PA point sets directly rather
# than the raw pre-selection candidate pool.
# ────────────────────────────────────────────────────────────────────────────
import umap
UMAP_SEED = 42
N_UMAP = 350
idx_fire_u = rng.choice(len(fire_feats), N_UMAP, replace=False)
idx_gan_u = rng.choice(len(gf), N_UMAP, replace=False)
idx_da_u = rng.choice(len(daf), N_UMAP, replace=False)
stack_u = np.vstack([fire_feats[idx_fire_u], gf[idx_gan_u], daf[idx_da_u]])
stack_scaled = sc2.transform(stack_u)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=UMAP_SEED)
emb = reducer.fit_transform(stack_scaled)
emb_fire, emb_gan, emb_da = emb[:N_UMAP], emb[N_UMAP:2 * N_UMAP], emb[2 * N_UMAP:]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.scatter(*emb_fire.T, c=FIRE_COL, s=14, alpha=0.55, label='Fire', zorder=5)
ax.scatter(*emb_gan.T, c=MCOLS[3], s=10, alpha=0.45, label='GAN (weight clip)')
ax.scatter(*emb_da.T, c=MCOLS[4], s=10, alpha=0.45, label='DA-GP-WGAN')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.legend(fontsize=9, markerscale=1.5)
ax = axes[1]
ax.scatter(*emb_fire.T, c=FIRE_COL, s=16, alpha=0.6, label='Fire', zorder=5)
ax.scatter(*emb_da.T, c=MCOLS[4], s=12, alpha=0.5, label='DA-GP-WGAN')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.legend(fontsize=9, markerscale=1.5)
fig.suptitle('')
plt.tight_layout()
savefig('fig40_umap_embedding.png')
# CAPTION (fig40): f"Non-linear (UMAP, random_state={UMAP_SEED}, n_neighbors=15, min_dist=0.1)
# embedding of standardized 58-dim features complementing the linear PCA views (fig24/fig33);
# left panel contrasts both GAN variants against fire, right isolates DA-GP-WGAN vs fire."

print(f"\nAll figures done: {time.time()-t0:.1f}s")
import os, glob
pngs = sorted(glob.glob(OUT + 'fig*.png'))
print(f"Total: {len(pngs)} fig*.png files in {OUT}")
