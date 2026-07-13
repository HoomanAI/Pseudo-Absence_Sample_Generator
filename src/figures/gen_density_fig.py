"""Feature Density Curves: Fire vs PA Methods — updated with GAN."""
import numpy as np, pandas as pd, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
import warnings; warnings.filterwarnings('ignore')

t0=time.time()
OUT='../../outputs/'

rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,
    'axes.titlesize':11,'axes.titleweight':'bold','axes.titlepad':6,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.6,
    'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9.5,'legend.framealpha':0.9,'legend.edgecolor':'#cccccc',
})

# Load data
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fcols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[fcols].values.astype(float)
hf=np.load(OUT+'heur_feats.npy')
rf_f=np.load(OUT+'rand_feats.npy')
sf=np.load(OUT+'sa_feats.npy')
gf=np.load(OUT+'gan_feats.npy')

# Map feature columns to plot titles (matching original figure)
FEAT_MAP=[
    (1, 'Elevation'),
    (0, 'Slope'),
    (7, 'Ndvi'),
    (8, 'Avg Temperature'),
    (9, 'Avg Precipitation'),
    (10, 'Avg Windspeed'),
    (16, 'Temp 1M Prior'),
    (13, 'Precip 1M Prior'),
]

# Colour / label scheme (matching original + GAN added)
GROUPS=[
    ('Fire',        fire_feats, '#E31A1C', '--', 2.2, 0.20),
    ('Heuristic PA', hf,        '#1F77B4', '-',  1.8, 0.18),
    ('SA PA',        sf,        '#FF7F0E', '-',  1.8, 0.18),
    ('Random PA',    rf_f,      '#2CA02C', '-',  1.8, 0.18),
    ('GAN PA',       gf,        '#9467BD', '-',  1.8, 0.18),
]

def kde_curve(data, n_pts=300):
    d=data[np.isfinite(data)]
    if len(d)<10: return np.array([]),np.array([])
    lo,hi=np.percentile(d,[0.5,99.5])
    bw=max((hi-lo)/60,1e-9)
    kde=gaussian_kde(d,bw_method=bw/(d.std()+1e-12))
    xs=np.linspace(lo,hi,n_pts)
    return xs,kde(xs)

fig,axes=plt.subplots(2,4,figsize=(20,10))
fig.suptitle('Feature Density Curves: Fire vs Pseudo-Absence Methods',
             fontsize=15,fontweight='bold',y=1.01)

for ax,(feat_idx,feat_title) in zip(axes.flat,FEAT_MAP):
    for name,feats,color,ls,lw,alpha in GROUPS:
        col=feats[:,feat_idx]
        xs,ys=kde_curve(col)
        if len(xs)==0: continue
        ax.plot(xs,ys,color=color,lw=lw,ls=ls,alpha=0.95 if name=='Fire' else 0.85,
                label=name if ax==axes[0,0] else '_')
        ax.fill_between(xs,ys,alpha=alpha,color=color)
    ax.set_title(feat_title)
    ax.set_ylabel('Density')
    ax.tick_params(axis='both',labelsize=8.5)
    ax.set_facecolor('#FAFAFA')

# Legend on first subplot
axes[0,0].legend(loc='upper right',fontsize=9.5,framealpha=0.92)
plt.tight_layout(rect=[0,0,1,0.99])
plt.savefig(OUT+'fig_feature_density_curves.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f"Done: {time.time()-t0:.1f}s")
