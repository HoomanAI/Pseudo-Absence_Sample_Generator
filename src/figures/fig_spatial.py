import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import warnings; warnings.filterwarnings('ignore')

df        = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire      = df[['LONGITUDE','LATITUDE']].values
fcols     = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats= df[fcols].values.astype(float)
fire_tree = cKDTree(fire)
heur  = np.load('../../outputs/heur_pts.npy')
rand  = np.load('../../outputs/rand_pts.npy')
sa    = np.load('../../outputs/sa_pts.npy')

C = {'fire':'#d62728','heur':'#1f77b4','rand':'#2ca02c','sa':'#ff7f0e'}
ALPHA = 0.4; SZ = 7

# ══════════════════════════════════════════════════════
# FIG 1 – Map overlay (all 4 point sets)
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9,8))
ax.scatter(rand[:,0], rand[:,1],  s=SZ, alpha=ALPHA, color=C['rand'],  label=f'Random PA (n={len(rand):,})',    zorder=2)
ax.scatter(sa[:,0],   sa[:,1],    s=SZ, alpha=ALPHA, color=C['sa'],    label=f'SA PA (n={len(sa):,})',           zorder=3)
ax.scatter(heur[:,0], heur[:,1],  s=SZ, alpha=ALPHA, color=C['heur'],  label=f'Heuristic PA (n={len(heur):,})', zorder=4)
ax.scatter(fire[:,0], fire[:,1],  s=SZ+2, alpha=0.75, color=C['fire'], label=f'Fire points (n={len(fire):,})',  zorder=5)
ax.set_xlabel('Longitude (°)', fontsize=12); ax.set_ylabel('Latitude (°)', fontsize=12)
ax.set_title('Alberta Fire Points vs Pseudo-Absence Methods\n(Heuristic BP_V6 | SA | Random)', fontsize=13, fontweight='bold')
ax.legend(markerscale=2, fontsize=10, loc='upper right')
ax.set_facecolor('#f0f4f8'); fig.patch.set_facecolor('white'); ax.grid(True, alpha=0.3)
ax.annotate('10 km exclusion\nbuffer (Heuristic & SA)', xy=(-115.5,55.2), fontsize=8.5,
            color='#555', style='italic', bbox=dict(boxstyle='round,pad=0.3',fc='white',alpha=0.7))
plt.tight_layout()
plt.savefig('../../outputs/fig01_map_overlay.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 1 done')

# ══════════════════════════════════════════════════════
# FIG 2 – Side-by-side point maps (4 panels)
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharex=True, sharey=True)
datasets = [(fire, C['fire'],  f'Fire Points\n(n={len(fire):,})'),
            (heur, C['heur'],  f'Heuristic PA\n(n={len(heur):,})'),
            (sa,   C['sa'],    f'SA PA\n(n={len(sa):,})'),
            (rand, C['rand'],  f'Random PA\n(n={len(rand):,})')]
for ax, (pts, col, title) in zip(axes, datasets):
    ax.scatter(pts[:,0], pts[:,1], s=5, alpha=0.35, color=col)
    ax.set_title(title, fontsize=11, fontweight='bold', color=col)
    ax.set_xlabel('Longitude (°)', fontsize=9)
    ax.set_facecolor('#f5f5f5'); ax.grid(True, alpha=0.25)
    cx, cy = pts.mean(0)
    ax.plot(cx, cy, marker='*', ms=12, color='black', zorder=5, label='Centroid')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Latitude (°)', fontsize=10)
fig.suptitle('Spatial Distribution: Fire | Heuristic | SA | Random', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../../outputs/fig02_side_by_side_maps.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 2 done')

# ══════════════════════════════════════════════════════
# FIG 3 – Density heatmaps (4 panels)
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for ax, pts, title, cmap in zip(axes,
        [fire, heur, sa, rand],
        ['Fire Points','Heuristic PA','SA PA','Random PA'],
        ['Reds','Blues','Oranges','Greens']):
    h = ax.hist2d(pts[:,0], pts[:,1], bins=25, norm=LogNorm(), cmap=cmap)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude (°)', fontsize=9)
axes[0].set_ylabel('Latitude (°)', fontsize=9)
fig.suptitle('Point Density Heatmaps — log scale', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig03_density_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 3 done')

# ══════════════════════════════════════════════════════
# FIG 4 – Ripley's K-function
# ══════════════════════════════════════════════════════
r_vals = np.linspace(0.05, 0.8, 20)
area   = (fire[:,0].max()-fire[:,0].min())*(fire[:,1].max()-fire[:,1].min())
def kc(pts):
    t=cKDTree(pts); lam=len(pts)/area
    return np.array([(np.array(t.query_ball_point(pts,r,return_length=True))-1).mean()/lam for r in r_vals])

kf=kc(fire); kh=kc(heur); ks=kc(sa); kr=kc(rand)
k_csr = np.pi*r_vals**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax=axes[0]
ax.plot(r_vals,kf,'k-', lw=2.5, label='Fire points')
ax.plot(r_vals,kh,'-',  color=C['heur'],lw=2.2, label='Heuristic PA')
ax.plot(r_vals,ks,'-',  color=C['sa'],  lw=2.2, label='SA PA')
ax.plot(r_vals,kr,'-',  color=C['rand'],lw=2.2, label='Random PA')
ax.plot(r_vals,k_csr,'k--',lw=1.5,alpha=0.5,label='CSR')
ax.set_xlabel('Distance r (°)',fontsize=11); ax.set_ylabel("Ripley's K(r)",fontsize=11)
ax.set_title("Ripley's K Curves",fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)

ax=axes[1]
sse_h_r=(kh-kf)**2; sse_s_r=(ks-kf)**2; sse_r_r=(kr-kf)**2
ax.fill_between(r_vals,sse_h_r,alpha=0.35,color=C['heur'],label=f'Heuristic SSE={sse_h_r.sum():.2f}')
ax.fill_between(r_vals,sse_s_r,alpha=0.35,color=C['sa'],  label=f'SA SSE={sse_s_r.sum():.2f}')
ax.fill_between(r_vals,sse_r_r,alpha=0.35,color=C['rand'],label=f'Random SSE={sse_r_r.sum():.2f}')
for arr,col in [(sse_h_r,C['heur']),(sse_s_r,C['sa']),(sse_r_r,C['rand'])]:
    ax.plot(r_vals,arr,color=col,lw=2)
ax.set_xlabel('Distance r (°)',fontsize=11); ax.set_ylabel('(K_PA − K_fire)²',fontsize=11)
ax.set_title("Ripley's K SSE by Distance Band",fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
fig.suptitle("Spatial Pattern — Ripley's K Function",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig04_ripleys_k.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 4 done')

# ══════════════════════════════════════════════════════
# FIG 5 – Marginal distributions
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, coord, label in zip(axes,[1,0],['Latitude (°)','Longitude (°)']):
    ax.hist(fire[:,coord],bins=25,alpha=0.55,color=C['fire'],density=True,label='Fire')
    ax.hist(heur[:,coord],bins=25,alpha=0.45,color=C['heur'],density=True,label='Heuristic PA')
    ax.hist(sa[:,coord],  bins=25,alpha=0.45,color=C['sa'],  density=True,label='SA PA')
    ax.hist(rand[:,coord],bins=25,alpha=0.45,color=C['rand'],density=True,label='Random PA')
    ax.set_xlabel(label,fontsize=11); ax.set_ylabel('Density',fontsize=11)
    ax.set_title(f'{label} Distribution',fontsize=12,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
fig.suptitle('Marginal Spatial Distributions',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig05_lat_lon_distributions.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 5 done')

# ══════════════════════════════════════════════════════
# FIG 6 – KDE scatter (3 PA methods)
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
for ax, pts, title in zip(axes,
        [heur, sa, rand],['Heuristic PA','SA PA','Random PA']):
    xy=np.vstack([pts[:,0],pts[:,1]]); z=gaussian_kde(xy)(xy)
    sc=ax.scatter(pts[:,0],pts[:,1],c=z,cmap='viridis',s=7,alpha=0.7)
    ax.scatter(fire[:,0],fire[:,1],s=4,color=C['fire'],alpha=0.12,label='Fire pts')
    plt.colorbar(sc,ax=ax,label='KDE density')
    ax.set_title(title,fontsize=12,fontweight='bold')
    ax.set_xlabel('Longitude (°)',fontsize=10)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.25)
axes[0].set_ylabel('Latitude (°)',fontsize=10)
fig.suptitle('KDE Density Scatter: Heuristic vs SA vs Random\n(fire points overlaid)',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig06_kde_scatter.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 6 done')

print('All spatial figures saved.')
