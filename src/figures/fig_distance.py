import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree

np.random.seed(42)
C = {'fire':'#d62728','heur':'#1f77b4','rand':'#2ca02c','sa':'#ff7f0e'}

df       = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire     = df[['LONGITUDE','LATITUDE']].values
fire_tree= cKDTree(fire)
heur_pts = np.load('../../outputs/heur_pts.npy')
rand_pts  = np.load('../../outputs/rand_pts.npy')
sa_pts   = np.load('../../outputs/sa_pts.npy')
DEG2KM   = 111.0

d_heur_km = fire_tree.query(heur_pts, k=1)[0] * DEG2KM
d_rand_km  = fire_tree.query(rand_pts,  k=1)[0] * DEG2KM
d_sa_km   = fire_tree.query(sa_pts,   k=1)[0] * DEG2KM

nh = cKDTree(heur_pts).query(heur_pts, k=2)[0][:,1] * DEG2KM
nr  = cKDTree(rand_pts).query(rand_pts,  k=2)[0][:,1] * DEG2KM
ns  = cKDTree(sa_pts).query(sa_pts,   k=2)[0][:,1] * DEG2KM
nf  = cKDTree(fire).query(fire,         k=2)[0][:,1] * DEG2KM

# ══════════════════════════════════════════════════════
# FIG 17 – Min distance to fire histogram
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins  = np.linspace(0, 120, 60)
bins2 = np.linspace(0, 100, 60)

ax = axes[0]
ax.hist(d_heur_km, bins=bins, alpha=0.55, color=C['heur'], density=True, label='Heuristic PA')
ax.hist(d_sa_km,   bins=bins, alpha=0.55, color=C['sa'],   density=True, label='SA PA')
ax.hist(d_rand_km, bins=bins, alpha=0.55, color=C['rand'], density=True, label='Random PA')
ax.axvline(10, color='black', lw=2, ls='--', label='10 km buffer')
ax.set_xlabel('Min distance to nearest fire (km)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distance to Nearest Fire Point', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
for d,col,lbl in [(d_heur_km,C['heur'],'Heuristic'),(d_sa_km,C['sa'],'SA'),(d_rand_km,C['rand'],'Random')]:
    ax.hist(d, bins=bins2, alpha=0.5, color=col, label=lbl, cumulative=True, density=True)
ax.set_xlabel('Min distance to nearest fire (km)', fontsize=11)
ax.set_ylabel('Cumulative proportion', fontsize=11)
ax.set_title('CDF — Distance to Fire', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.suptitle('Distance Analysis: PA Points vs Fire Points', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig17_distance_hist.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 17 done')

# ══════════════════════════════════════════════════════
# FIG 18 – Self-clustering NN distance
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
bins3 = np.linspace(0, 30, 60)
for arr,col,lbl in [(nf,C['fire'],f'Fire (mean={nf.mean():.2f} km)'),
                     (nh,C['heur'],f'Heuristic (mean={nh.mean():.2f} km)'),
                     (ns,C['sa'],  f'SA (mean={ns.mean():.2f} km)'),
                     (nr,C['rand'],f'Random (mean={nr.mean():.2f} km)')]:
    ax.hist(arr, bins=bins3, alpha=0.55, color=col, density=True, label=lbl)
    ax.axvline(arr.mean(), color=col, lw=2, ls='--', alpha=0.8)
ax.set_xlabel('Nearest-neighbour distance (km)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Self-Clustering: NN Distance Distribution\n(closer to Fire = more realistic PA spacing)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../outputs/fig18_nn_distance.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 18 done')

# ══════════════════════════════════════════════════════
# FIG 19 – Spatial metrics summary (3 methods)
# ══════════════════════════════════════════════════════
stats = np.load('../../outputs/spatial_stats.npy')
# stats=[sse_h,sse_r,sse_s, cd_h,cd_r,cd_s, gv_h,gv_r,gv_s, nn_h,nn_r,nn_s, nn_f]
sse_h,sse_r,sse_s = stats[0],stats[1],stats[2]
cd_h, cd_r, cd_s  = stats[3],stats[4],stats[5]
gv_h, gv_r, gv_s  = stats[6],stats[7],stats[8]
nn_h2,nn_r2,nn_s2,nn_f2 = stats[9],stats[10],stats[11],stats[12]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

def bar3(ax, vals, labels, title, ylabel, invert=False):
    cols=[C['heur'],C['sa'],C['rand']]
    bars=ax.bar(labels, vals, color=cols, alpha=0.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if invert: ax.annotate('↓ better', xy=(0.98,0.97), xycoords='axes fraction',
                            ha='right', va='top', fontsize=9, color='gray', style='italic')

bar3(axes[0],[sse_h,sse_s,sse_r],["Heuristic","SA","Random"],"Ripley's K SSE","SSE (↓ better)",True)
bar3(axes[1],[cd_h, cd_s, cd_r], ["Heuristic","SA","Random"],"Centroid Distance","Distance (°)")
bar3(axes[2],[gv_h, gv_s, gv_r], ["Heuristic","SA","Random"],"Grid Variance","Variance (↓ better)",True)
bar3(axes[3],[nn_h2*DEG2KM, ns2*DEG2KM, nn_r2*DEG2KM] if False else
             [stats[9]*DEG2KM,stats[11]*DEG2KM,stats[10]*DEG2KM],
             ["Heuristic","SA","Random"],"Mean NN Distance","km")
axes[3].axhline(stats[12]*DEG2KM, color=C['fire'], ls='--', lw=1.5, label=f'Fire ref={stats[12]*DEG2KM:.2f} km')
axes[3].legend(fontsize=8)

fig.suptitle('Spatial Quality Metrics — Heuristic vs SA vs Random', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig19_spatial_summary.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 19 done')

# ══════════════════════════════════════════════════════
# FIG 20 – Temporal distribution (unchanged)
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
df['YEAR'].hist(bins=30, color=C['fire'], alpha=0.8, ax=ax, edgecolor='white')
ax.set_xlabel('Year', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title('Fire Points by Year', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)
ax = axes[1]
df['MONTH'].value_counts().sort_index().plot(kind='bar', color=C['fire'], alpha=0.8, ax=ax, edgecolor='white')
ax.set_xlabel('Month', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title('Fire Points by Month', fontsize=12, fontweight='bold')
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.set_xticklabels([months[int(x.get_text())-1] for x in ax.get_xticklabels()], rotation=45)
ax.grid(axis='y', alpha=0.3)
fig.suptitle('Temporal Distribution of Fire Points', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig20_temporal_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 20 done')

# ══════════════════════════════════════════════════════
# FIG 21 – Summary dashboard (3-way)
# ══════════════════════════════════════════════════════
ml_df = pd.read_csv('../../outputs/ml_results.csv')
def get(m, col): return float(ml_df.loc[ml_df['Model']==m, col].values[0])

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#f8f9fa')
fig.text(0.5, 0.97, 'Pseudo-Absence Evaluation Dashboard — Heuristic | SA | Random\nAlberta Fire Dataset (3,370 points, 58 features)',
         ha='center', va='top', fontsize=14, fontweight='bold')

# Highlight boxes
boxes = [
    ("Ripley K SSE", f"H={sse_h:.2f}  SA={sse_s:.2f}  R={sse_r:.2f}", "Heuristic best", C['heur']),
    ("SVM AUC",  f"H={get('SVM','AUC_Heuristic'):.4f} SA={get('SVM','AUC_SA'):.4f} R={get('SVM','AUC_Random'):.4f}",
     f"Heuristic wins p<0.001", C['heur']),
    ("KNN AUC",  f"H={get('KNN','AUC_Heuristic'):.4f} SA={get('KNN','AUC_SA'):.4f} R={get('KNN','AUC_Random'):.4f}",
     f"Heuristic best p<0.001", C['heur']),
    ("SA Grid Var", f"H={gv_h:.0f}  SA={gv_s:.0f}  R={gv_r:.1f}",
     "SA better coverage\nthan Heuristic", C['sa']),
]
for i,(title,vals,verdict,col) in enumerate(boxes):
    ax=fig.add_axes([0.03+i*0.245,0.73,0.225,0.18])
    ax.set_facecolor(col); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.5,0.78,title, ha='center',va='center',fontsize=11,fontweight='bold',color='white')
    ax.text(0.5,0.48,vals,  ha='center',va='center',fontsize=9,color='white')
    ax.text(0.5,0.18,verdict,ha='center',va='center',fontsize=9,color='white',style='italic',fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# AUC bar
ax1=fig.add_axes([0.03,0.38,0.45,0.30])
ML=['RandomForest','XGBoost','SVM','KNN']; x=np.arange(4); w=0.28
for off,key,col,lbl in zip([-w,0,w],
        ['AUC_Heuristic','AUC_SA','AUC_Random'],
        [C['heur'],C['sa'],C['rand']],
        ['Heuristic','SA','Random']):
    vals=[get(m,key) for m in ML]
    ax1.bar(x+off,vals,w,color=col,alpha=0.85,label=lbl)
ax1.set_xticks(x); ax1.set_xticklabels(ML,fontsize=9)
ax1.set_ylim(0.82,1.015); ax1.set_ylabel('AUC',fontsize=10)
ax1.set_title('AUC Comparison',fontsize=11,fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y',alpha=0.3)

# TSS bar
ax2=fig.add_axes([0.53,0.38,0.45,0.30])
for off,key,col,lbl in zip([-w,0,w],
        ['TSS_Heuristic','TSS_SA','TSS_Random'],
        [C['heur'],C['sa'],C['rand']],
        ['Heuristic','SA','Random']):
    vals=[get(m,key) for m in ML]
    ax2.bar(x+off,vals,w,color=col,alpha=0.85,label=lbl)
ax2.set_xticks(x); ax2.set_xticklabels(ML,fontsize=9)
ax2.set_ylim(0,1.1); ax2.set_ylabel('TSS',fontsize=10)
ax2.set_title('TSS Comparison',fontsize=11,fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y',alpha=0.3)

# Summary table
ax3=fig.add_axes([0.03,0.02,0.94,0.32])
ax3.axis('off')
table_data=[
    ["Metric","Heuristic","SA","Random","Best Method"],
    [f"Ripley's K SSE (↓)",f"{sse_h:.2f}",f"{sse_s:.2f}",f"{sse_r:.2f}","Heuristic"],
    [f"Grid Variance (↓)",  f"{gv_h:.0f}", f"{gv_s:.0f}", f"{gv_r:.1f}","Random < SA < Heuristic"],
    [f"Mean NN Dist (km)",
     f"{stats[9]*DEG2KM:.2f}",f"{stats[11]*DEG2KM:.2f}",f"{stats[10]*DEG2KM:.2f}",
     f"Fire ref={stats[12]*DEG2KM:.2f} km"],
    [f"RF AUC",   f"{get('RandomForest','AUC_Heuristic'):.4f}",
                   f"{get('RandomForest','AUC_SA'):.4f}",
                   f"{get('RandomForest','AUC_Random'):.4f}","All ≈ equal"],
    [f"SVM AUC",  f"{get('SVM','AUC_Heuristic'):.4f}",
                   f"{get('SVM','AUC_SA'):.4f}",
                   f"{get('SVM','AUC_Random'):.4f}","Heuristic > SA > Random"],
    [f"KNN AUC",  f"{get('KNN','AUC_Heuristic'):.4f}",
                   f"{get('KNN','AUC_SA'):.4f}",
                   f"{get('KNN','AUC_Random'):.4f}","Heuristic > SA > Random"],
    [f"KNN TSS",  f"{get('KNN','TSS_Heuristic'):.4f}",
                   f"{get('KNN','TSS_SA'):.4f}",
                   f"{get('KNN','TSS_Random'):.4f}","Heuristic best"],
]
tbl=ax3.table(cellText=table_data,cellLoc='center',loc='center',
              colWidths=[0.3,0.15,0.12,0.12,0.31])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r,c),cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor('#333333'); cell.set_text_props(color='white',fontweight='bold')
    elif c==1: cell.set_facecolor('#dce8f5')
    elif c==2: cell.set_facecolor('#ffe8cc')
    elif c==3: cell.set_facecolor('#ddf5dd')
    cell.set_edgecolor('#cccccc')
ax3.set_title('3-Method Comparison Summary', fontsize=11, fontweight='bold', pad=5)

plt.savefig('../../outputs/fig21_dashboard.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 21 done')

print('All distance & summary figures saved.')
