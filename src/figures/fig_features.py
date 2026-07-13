import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree

np.random.seed(42)
COLORS = {'fire':'#d62728','heur':'#1f77b4','rand':'#2ca02c','sa':'#ff7f0e'}

df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire      = df[['LONGITUDE','LATITUDE']].values
fcols     = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats= df[fcols].values.astype(float)
fire_tree = cKDTree(fire)
heur_pts  = np.load('../../outputs/heur_pts.npy')
rand_pts  = np.load('../../outputs/rand_pts.npy')
sa_pts    = np.load('../../outputs/sa_pts.npy')

def interp(pts, k=7):
    d,ix=fire_tree.query(pts,k=k); d=np.maximum(d,1e-9)
    w=1/d**2; w/=w.sum(1,keepdims=True)
    f=np.einsum('nk,nkf->nf',w,fire_feats[ix])
    f+=np.random.normal(0,0.05*fire_feats.std(0),f.shape)
    return f

hf = interp(heur_pts)
rf = interp(rand_pts)
sf = interp(sa_pts)

KEY_FEATS = ['slope','elevation','aspect','ndvi','avg_temperature',
             'avg_precipitation','avg_windspeed','twi','valley_depth',
             'temp_1m_prior','precip_1m_prior','wind_1m_prior']

fire_df = pd.DataFrame(fire_feats, columns=fcols)
heur_df = pd.DataFrame(hf, columns=fcols)
rand_df = pd.DataFrame(rf, columns=fcols)
sa_df   = pd.DataFrame(sf, columns=fcols)

# ════════════════════════════════════════════════════════════
# FIG 12 – Box plots for key features (fire vs PA)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.ravel()
for i, feat in enumerate(KEY_FEATS):
    ax = axes[i]
    data = [fire_df[feat].values, heur_df[feat].values, sa_df[feat].values, rand_df[feat].values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black',lw=2))
    for patch, col in zip(bp['boxes'], [COLORS['fire'],COLORS['heur'],COLORS['sa'],COLORS['rand']]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_xticklabels(['Fire','Heuristic','SA','Random'], fontsize=8)
    ax.set_title(feat.replace('_',' ').title(), fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Feature Distributions: Fire vs Heuristic vs SA vs Random', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../../outputs/fig12_feature_boxplots.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 12 done')

# ════════════════════════════════════════════════════════════
# FIG 13 – Violin plots for key features
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.ravel()
for i, feat in enumerate(KEY_FEATS):
    ax = axes[i]
    dfs = [fire_df[feat].values, heur_df[feat].values, sa_df[feat].values, rand_df[feat].values]
    parts = ax.violinplot(dfs, positions=[1,2,3,4], showmedians=True, showextrema=True)
    for j,(pc,col) in enumerate(zip(parts['bodies'],[COLORS['fire'],COLORS['heur'],COLORS['sa'],COLORS['rand']])):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    for part in ['cbars','cmins','cmaxes','cmedians']:
        parts[part].set_color('black'); parts[part].set_linewidth(1.5)
    ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['Fire','Heuristic','SA','Random'], fontsize=8)
    ax.set_title(feat.replace('_',' ').title(), fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Feature Violin Plots: Fire vs Heuristic vs SA vs Random', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../../outputs/fig13_feature_violins.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 13 done')

# ════════════════════════════════════════════════════════════
# FIG 14 – Feature KDE overlay (8 selected features)
# ════════════════════════════════════════════════════════════
from scipy.stats import gaussian_kde
sel8 = ['elevation','slope','ndvi','avg_temperature','avg_precipitation',
        'avg_windspeed','temp_1m_prior','precip_1m_prior']
fig, axes = plt.subplots(2, 4, figsize=(17, 9))
axes = axes.ravel()
for i, feat in enumerate(sel8):
    ax = axes[i]
    for dset, label, col in [(fire_df, 'Fire',         COLORS['fire']),
                              (heur_df,'Heuristic PA', COLORS['heur']),
                              (sa_df,  'SA PA',        COLORS['sa']),
                              (rand_df,'Random PA',    COLORS['rand'])]:
        vals = dset[feat].dropna().values
        xs = np.linspace(vals.min(), vals.max(), 300)
        kde = gaussian_kde(vals, bw_method=0.25)
        ax.plot(xs, kde(xs), color=col, lw=2, label=label)
        ax.fill_between(xs, kde(xs), alpha=0.12, color=col)
    ax.set_title(feat.replace('_',' ').title(), fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=8); ax.grid(True, alpha=0.25)
    if i == 0: ax.legend(fontsize=8)
fig.suptitle('Feature Density Curves: Fire vs Pseudo-Absence Methods', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig14_feature_kde.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 14 done')

# ════════════════════════════════════════════════════════════
# FIG 15 – Correlation heatmap: fire vs heuristic vs random (feature means)
# ════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors
means = pd.DataFrame({
    'Fire':         fire_df[KEY_FEATS].mean(),
    'Heuristic PA': heur_df[KEY_FEATS].mean(),
    'SA PA':        sa_df[KEY_FEATS].mean(),
    'Random PA':    rand_df[KEY_FEATS].mean(),
}).T
means_norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(means_norm.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(KEY_FEATS)))
ax.set_xticklabels([f.replace('_',' ').title() for f in KEY_FEATS],
                   rotation=35, ha='right', fontsize=9)
ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['Fire','Heuristic PA','SA PA','Random PA'], fontsize=11)
plt.colorbar(im, ax=ax, label='Normalized mean value')
for r in range(4):
    for c in range(len(KEY_FEATS)):
        ax.text(c, r, f'{means_norm.values[r,c]:.2f}', ha='center', va='center',
                fontsize=8, color='black')
ax.set_title('Normalized Feature Mean: Fire vs Heuristic vs SA vs Random\n(0=min, 1=max across groups)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig15_feature_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 15 done')

# ════════════════════════════════════════════════════════════
# FIG 16 – Scatter: elevation vs temperature coloured by class
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
pairs = [('elevation','avg_temperature'), ('slope','ndvi')]
for ax, (fx, fy) in zip(axes, pairs):
    ax.scatter(rand_df[fx], rand_df[fy], s=6, alpha=0.25, color=COLORS['rand'],  label='Random PA')
    ax.scatter(sa_df[fx],   sa_df[fy],   s=6, alpha=0.25, color=COLORS['sa'],    label='SA PA')
    ax.scatter(heur_df[fx], heur_df[fy], s=6, alpha=0.25, color=COLORS['heur'],  label='Heuristic PA')
    ax.scatter(fire_df[fx], fire_df[fy], s=6, alpha=0.45, color=COLORS['fire'],  label='Fire points')
    ax.set_xlabel(fx.replace('_',' ').title(), fontsize=11)
    ax.set_ylabel(fy.replace('_',' ').title(), fontsize=11)
    tx = fx.replace('_',' ').title(); ty = fy.replace('_',' ').title()
    ax.set_title(f'{tx} vs {ty}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2); ax.grid(True, alpha=0.25)
fig.suptitle('Feature Space Scatter: Fire vs Heuristic vs SA vs Random', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig16_feature_scatter.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 16 done')

print('All feature figures saved.')
