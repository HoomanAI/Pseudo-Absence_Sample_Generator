"""Regenerate 3 original figures with GAN added."""
import numpy as np, pandas as pd, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
import warnings; warnings.filterwarnings('ignore')

t0=time.time()
OUT='../../outputs/'

# ── Load all data ──────────────────────────────────────────────────────────────
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fcols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[fcols].values.astype(float)
hf=np.load(OUT+'heur_feats.npy')
rf_f=np.load(OUT+'rand_feats.npy')
sf=np.load(OUT+'sa_feats.npy')
gf=np.load(OUT+'gan_feats.npy')

cv_h=np.load(OUT+'cv_h.npy',allow_pickle=True).item()
cv_r=np.load(OUT+'cv_r.npy',allow_pickle=True).item()
cv_s=np.load(OUT+'cv_s.npy',allow_pickle=True).item()
cv_g=np.load(OUT+'cv_g.npy',allow_pickle=True).item()

# 12 features used in the original figures
FEAT_IDX= [0, 1, 2, 7, 8, 9, 10, 6, 5, 16, 13, 19]
FEAT_NAMES=['Slope','Elevation','Aspect','Ndvi','Avg Temperature','Avg Precipitation',
            'Avg Windspeed','Twi','Valley Depth','Temp 1M Prior','Precip 1M Prior','Wind 1M Prior']

# Groups: (label, data, violin-color)
GROUPS=[('Fire',      fire_feats, '#E74C3C'),
        ('Heuristic', hf,         '#5B9BD5'),
        ('SA',        sf,         '#FFA040'),
        ('Random',    rf_f,       '#5CB85C'),
        ('GAN',       gf,         '#9B59B6')]

print(f"Data loaded: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Feature Violin Plots  (3×4 grid)
# ══════════════════════════════════════════════════════════════════════════════
rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                 'axes.titlesize':10,'axes.titleweight':'bold',
                 'axes.spines.top':False,'axes.spines.right':False,
                 'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.5,
                 'xtick.labelsize':8,'ytick.labelsize':8})

fig,axes=plt.subplots(3,4,figsize=(16,12))
fig.suptitle('Feature Violin Plots: Fire vs Heuristic vs SA vs Random vs GAN',
             fontsize=13,fontweight='bold',y=1.005)
fig.patch.set_facecolor('white')

for ax,(fi,fname) in zip(axes.flat, zip(FEAT_IDX,FEAT_NAMES)):
    data=[g[1][:,fi] for g in GROUPS]
    xlabels=[g[0] for g in GROUPS]
    colors=[g[2] for g in GROUPS]
    parts=ax.violinplot(data,positions=range(len(GROUPS)),
                        showmedians=True,showextrema=True,widths=0.75)
    # Style each violin
    for i,(body,col) in enumerate(zip(parts['bodies'],colors)):
        body.set_facecolor(col); body.set_alpha(0.65); body.set_edgecolor('#333')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.8)
    parts['cmaxes'].set_color('#555'); parts['cmins'].set_color('#555')
    parts['cbars'].set_color('#555'); parts['cbars'].set_linewidth(1.0)
    # Overlay mean dot
    for i,(d,col) in enumerate(zip(data,colors)):
        ax.scatter([i],[np.mean(d)],s=18,color='white',zorder=5,edgecolors='#333',linewidth=0.8)
    ax.set_title(fname,fontsize=9.5,fontweight='bold',pad=4)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(xlabels,fontsize=8,rotation=0)
    ax.set_facecolor('#F8F9FA')
    ax.tick_params(axis='y',labelsize=8)

plt.tight_layout(rect=[0,0,1,0.99])
plt.savefig(OUT+'fig_violin_with_gan.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close('all')
print(f"Fig1 violin: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Normalized Feature Mean Heatmap  (5 rows × 12 cols)
# ══════════════════════════════════════════════════════════════════════════════
group_names=['Fire','Heuristic PA','SA PA','Random PA','GAN PA']
means=np.array([[g[1][:,fi].mean() for fi in FEAT_IDX] for g in GROUPS])
# Normalise per column (feature): 0=min, 1=max across groups
mn=means.min(0,keepdims=True); mx=means.max(0,keepdims=True)
norm=(means-mn)/(mx-mn+1e-10)

fig,ax=plt.subplots(figsize=(16,5.5))
fig.patch.set_facecolor('white')
im=ax.imshow(norm,cmap='coolwarm_r',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(12)); ax.set_xticklabels(FEAT_NAMES,rotation=30,ha='right',fontsize=10)
ax.set_yticks(range(5)); ax.set_yticklabels(group_names,fontsize=11)
ax.set_title('Normalized Feature Mean: Fire vs Heuristic vs SA vs Random vs GAN\n(0=min, 1=max across groups)',
             fontsize=13,fontweight='bold',pad=10)
# Annotate cells
for i in range(5):
    for j in range(12):
        v=norm[i,j]
        ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9.5,fontweight='bold',
                color='white' if (v<0.25 or v>0.80) else '#222')
cbar=plt.colorbar(im,ax=ax,shrink=0.85,label='Normalized mean value')
cbar.ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(OUT+'fig_heatmap_with_gan.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close('all')
print(f"Fig2 heatmap: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ROC Curves with ±1 std shading  (4 panels now, one per method)
# ══════════════════════════════════════════════════════════════════════════════
CVS_LIST=[('Heuristic PA',cv_h,'#00BFFF'),
          ('SA PA',        cv_s,'#FFA040'),
          ('Random PA',    cv_r,'#5CB85C'),
          ('GAN PA',       cv_g,'#BB86FC')]
MODELS=['RandomForest','XGBoost','SVM','KNN']
MOD_COLS={'RandomForest':'#FF69B4','XGBoost':'#FFA500','SVM':'#9B59B6','KNN':'#00CED1'}
MOD_FILL={'RandomForest':'#FF69B420','XGBoost':'#FFA50020','SVM':'#9B59B620','KNN':'#00CED120'}

def roc_mean_std(cv,model,n=100):
    fprs,tprs=cv['roc'][model]
    grid=np.linspace(0,1,n)
    interp=[np.interp(grid,f,t) for f,t in zip(fprs,tprs) if len(f)>1]
    if not interp: return grid,grid,np.zeros(n)
    arr=np.array(interp)
    return grid,arr.mean(0),arr.std(0)

fig,axes=plt.subplots(1,4,figsize=(20,6.5))
fig.patch.set_facecolor('#0D0D0D')
fig.suptitle('ROC Curves: Spatial Block CV (shaded = ±1 std)',
             fontsize=14,fontweight='bold',color='white',y=1.01)

for ax,(pa_label,cv,title_col) in zip(axes,CVS_LIST):
    ax.set_facecolor('#0D0D0D')
    ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#AAA',labelsize=9)
    ax.set_xlabel('FPR',color='#CCC',fontsize=10)
    ax.set_ylabel('TPR',color='#CCC',fontsize=10)
    ax.set_title(f'ROC — {pa_label}',color=title_col,fontsize=12,fontweight='bold',pad=8)
    ax.plot([0,1],[0,1],'--',color='#666',lw=1.2,label='Chance',zorder=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.grid(True,alpha=0.15,color='#888',linewidth=0.5)
    for mod in MODELS:
        auc=cv['res'][mod]['AUC']
        fpr_g,mean_tpr,std_tpr=roc_mean_std(cv,mod)
        col=MOD_COLS[mod]
        ax.plot(fpr_g,mean_tpr,color=col,lw=2.0,
                label=f'{mod} (AUC={auc:.4f})',zorder=3)
        ax.fill_between(fpr_g,
                        np.clip(mean_tpr-std_tpr,0,1),
                        np.clip(mean_tpr+std_tpr,0,1.05),
                        alpha=0.18,color=col,zorder=2)
    leg=ax.legend(loc='lower right',fontsize=8.5,framealpha=0.25,
                  edgecolor='#555',labelcolor='white')

plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(OUT+'fig_roc_with_gan.png',dpi=150,bbox_inches='tight',
            facecolor='#0D0D0D')
plt.close('all')
print(f"Fig3 ROC: {time.time()-t0:.1f}s")
print("All 3 figures done.")
