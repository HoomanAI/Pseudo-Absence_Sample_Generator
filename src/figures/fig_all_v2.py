"""All figures: spatial maps, ML bars/ROC/violin/scatter, distance dist."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

OUT='../../outputs/'
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire=df[['LONGITUDE','LATITUDE']].values
fire_tree=cKDTree(fire)

heur=np.load(OUT+'heur_pts.npy'); rand=np.load(OUT+'rand_pts.npy')
sa=np.load(OUT+'sa_pts.npy');     gan=np.load(OUT+'gan_pts.npy')
sp=pd.read_csv(OUT+'spatial_results.csv')

cvH=np.load(OUT+'cv_h.npy',allow_pickle=True).item()
cvR=np.load(OUT+'cv_r.npy',allow_pickle=True).item()
cvS=np.load(OUT+'cv_s.npy',allow_pickle=True).item()
cvG=np.load(OUT+'cv_g.npy',allow_pickle=True).item()
rH=cvH['res']; rR=cvR['res']; rS=cvS['res']; rG=cvG['res']
rocH=cvH['roc']; rocR=cvR['roc']; rocS=cvS['roc']; rocG=cvG['roc']
prH=cvH['pr'];   prR=cvR['pr'];   prS=cvS['pr'];   prG=cvG['pr']

ML=list(rH.keys())
C={'fire':'#d62728','heur':'#1f77b4','rand':'#2ca02c','sa':'#ff7f0e','gan':'#9467bd'}
MC={'RandomForest':'#e377c2','XGBoost':'#ff7f0e','SVM':'#9467bd','KNN':'#17becf'}
LBLS={'heur':'Heuristic BP_V6','rand':'Random','sa':'SA','gan':'GAN'}

# ── FIG 1: Spatial scatter (4 panels) ────────────────────────────────────────
fig,axes=plt.subplots(1,4,figsize=(20,6),sharex=True,sharey=True)
for ax,(pts,key) in zip(axes,[(heur,'heur'),(sa,'sa'),(gan,'gan'),(rand,'rand')]):
    ax.scatter(pts[:,0],pts[:,1],c=C[key],s=2,alpha=0.4,label=LBLS[key])
    ax.scatter(fire[:,0],fire[:,1],c=C['fire'],s=3,alpha=0.6,label='Fire')
    ax.set_title(LBLS[key],fontsize=11,fontweight='bold',color=C[key])
    ax.set_xlabel('Longitude'); ax.grid(alpha=0.2)
axes[0].set_ylabel('Latitude')
fig.suptitle('Alberta Fire Points vs Pseudo-Absence Methods (n=5500 each)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig01_spatial_scatter.png',dpi=150,bbox_inches='tight'); plt.close()
print('Fig1 done')

# ── FIG 2: K-function SSE bar ─────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,5))
keys=['heur','sa','gan','rand']; lbls=[LBLS[k] for k in keys]
vals=[sp.loc[sp.Method==m,'K_SSE'].values[0] for m in ['Heuristic','SA','GAN','Random']]
cols=[C[k] for k in keys]
bars=ax.bar(lbls,vals,color=cols,alpha=0.85,edgecolor='white',linewidth=1.2)
for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+0.05,f'{v:.3f}',ha='center',fontsize=10,fontweight='bold')
ax.set_ylabel("Ripley's K SSE (lower = more fire-like)",fontsize=11)
ax.set_title("Ripley's K-function SSE by PA Method",fontsize=12,fontweight='bold')
ax.grid(axis='y',alpha=0.35); plt.tight_layout()
plt.savefig(OUT+'fig02_ksse_bar.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig2 done')

# ── FIG 3: Spatial metrics radar / bar grid ───────────────────────────────────
metrics=['K_SSE','Centroid','Grid_Var','Mean_NN']
mlbls=["K-func SSE","Centroid Dist","Grid Variance","Mean NN Dist"]
methods_ord=['Heuristic','SA','GAN','Random']
cols_ord=[C['heur'],C['sa'],C['gan'],C['rand']]
fig,axes=plt.subplots(1,4,figsize=(18,5))
for ax,met,mlbl in zip(axes,metrics,mlbls):
    vals=[sp.loc[sp.Method==m,met].values[0] for m in methods_ord]
    bars=ax.bar(methods_ord,vals,color=cols_ord,alpha=0.85,edgecolor='white',linewidth=1.2)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v*1.01,f'{v:.4g}',ha='center',fontsize=8,fontweight='bold')
    ax.set_title(mlbl,fontsize=11,fontweight='bold'); ax.grid(axis='y',alpha=0.3)
    ax.set_xticklabels(methods_ord,rotation=15,ha='right',fontsize=9)
fig.suptitle('Spatial Evaluation Metrics (n=5500 per method)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig03_spatial_metrics.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig3 done')

# ── FIG 4: Distance-to-fire distributions ─────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,5))
for pts,key in [(heur,'heur'),(sa,'sa'),(gan,'gan'),(rand,'rand')]:
    d,_=fire_tree.query(pts,k=1)
    ax.hist(d*111,bins=50,density=True,alpha=0.55,color=C[key],label=LBLS[key])
ax.set_xlabel('Distance to nearest fire point (km)',fontsize=11)
ax.set_ylabel('Density',fontsize=11); ax.legend(fontsize=10)
ax.set_title('Distance to Nearest Fire — PA Methods',fontsize=12,fontweight='bold')
ax.grid(alpha=0.3); ax.axvline(10,color='gray',ls='--',lw=1.5,label='D_MIN=10km')
plt.tight_layout(); plt.savefig(OUT+'fig04_fire_dist.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig4 done')

# ── FIG 5: AUC grouped bar ────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(13,6))
x=np.arange(len(ML)); w=0.21
for offset,res,col,lbl in zip([-1.5*w,-0.5*w,0.5*w,1.5*w],
        [rH,rS,rG,rR],[C['heur'],C['sa'],C['gan'],C['rand']],
        ['Heuristic PA','SA PA','GAN PA','Random PA']):
    vals=[res[m]['AUC'] for m in ML]; errs=[res[m]['AUC_std'] for m in ML]
    bars=ax.bar(x+offset,vals,w,yerr=errs,capsize=4,color=col,label=lbl,alpha=0.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,f'{v:.3f}',
                ha='center',va='bottom',fontsize=6.5,color=col,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(ML,fontsize=11)
ax.set_ylabel('AUC (spatial block CV, 5-fold)',fontsize=11)
ax.set_title('AUC — Heuristic vs SA vs GAN vs Random PA',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.set_ylim(0.82,1.015); ax.grid(axis='y',alpha=0.35)
ax.axhline(1.0,color='gray',lw=1,ls='--',alpha=0.5)
plt.tight_layout(); plt.savefig(OUT+'fig05_auc_bar.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig5 done')

# ── FIG 6: TSS grouped bar ────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(13,6))
for offset,res,col,lbl in zip([-1.5*w,-0.5*w,0.5*w,1.5*w],
        [rH,rS,rG,rR],[C['heur'],C['sa'],C['gan'],C['rand']],
        ['Heuristic PA','SA PA','GAN PA','Random PA']):
    vals=[res[m]['TSS'] for m in ML]; errs=[res[m]['TSS_std'] for m in ML]
    bars=ax.bar(x+offset,vals,w,yerr=errs,capsize=4,color=col,label=lbl,alpha=0.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,max(bar.get_height(),0)+0.008,f'{v:.3f}',
                ha='center',va='bottom',fontsize=6.5,color=col,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(ML,fontsize=11)
ax.set_ylabel('TSS (Sens + Spec − 1)',fontsize=11)
ax.set_title('TSS — Heuristic vs SA vs GAN vs Random PA',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.set_ylim(0,1.15); ax.grid(axis='y',alpha=0.35)
plt.tight_layout(); plt.savefig(OUT+'fig06_tss_bar.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig6 done')

# ── FIG 7: ROC curves (4 panels) ─────────────────────────────────────────────
common_fpr=np.linspace(0,1,200)
fig,axes=plt.subplots(1,4,figsize=(20,5))
for ax,roc_d,res,lbl,pc in zip(axes,
        [rocH,rocS,rocG,rocR],[rH,rS,rG,rR],
        ['Heuristic PA','SA PA','GAN PA','Random PA'],
        [C['heur'],C['sa'],C['gan'],C['rand']]):
    for nm,mc in MC.items():
        if nm not in roc_d: continue
        fps_l,tps_l=roc_d[nm]; itprs=[]
        for fpr_f,tpr_f in zip(fps_l,tps_l):
            f=interp1d(fpr_f,tpr_f,bounds_error=False,fill_value=(0,1))
            itprs.append(f(common_fpr))
        mt=np.mean(itprs,0); st=np.std(itprs,0)
        ax.plot(common_fpr,mt,color=mc,lw=2.2,label=f'{nm} ({res[nm]["AUC"]:.3f})')
        ax.fill_between(common_fpr,mt-st,mt+st,alpha=0.12,color=mc)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_xlabel('FPR',fontsize=10); ax.set_ylabel('TPR',fontsize=10)
    ax.set_title(lbl,fontsize=11,fontweight='bold',color=pc)
    ax.legend(fontsize=7,loc='lower right'); ax.grid(True,alpha=0.3)
fig.suptitle('ROC Curves: Spatial Block CV (shaded ±1σ)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig07_roc.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig7 done')

# ── FIG 8: Bootstrap ΔAUC CI ──────────────────────────────────────────────────
def bdelta(a,b,n=2000):
    np.random.seed(42)
    d=np.array([np.mean(np.random.choice(a,len(a),True))-np.mean(np.random.choice(b,len(b),True)) for _ in range(n)])
    return float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

pairs=[('H−R',rH,rR,C['heur']),('SA−R',rS,rR,C['sa']),('GAN−R',rG,rR,C['gan']),('H−GAN',rH,rG,'#8c564b')]
fig,axes=plt.subplots(1,4,figsize=(18,5),sharey=True)
for ax,(plbl,ra,rb,pc) in zip(axes,pairs):
    dm_list,lo_list,hi_list,cols=[],[],[],[]
    for m in ML:
        dm,lo,hi=bdelta(ra[m]['folds'],rb[m]['folds'])
        dm_list.append(dm); lo_list.append(lo); hi_list.append(hi)
        cols.append('#2ca02c' if dm>0 else '#d62728')
    yp=np.arange(len(ML))
    elo=np.maximum(np.array(dm_list)-np.array(lo_list),0)
    ehi=np.maximum(np.array(hi_list)-np.array(dm_list),0)
    ax.barh(yp,dm_list,xerr=[elo,ehi],capsize=4,color=cols,alpha=0.8,height=0.5)
    ax.axvline(0,color='black',lw=1.5,ls='--')
    for i,(m,dm) in enumerate(zip(ML,dm_list)):
        ax.text(max(dm,0)+0.001,i,f'Δ={dm:+.3f}',va='center',fontsize=8)
    ax.set_yticks(yp); ax.set_yticklabels(ML,fontsize=9)
    ax.set_xlabel(f'ΔAUC ({plbl})',fontsize=10); ax.set_title(plbl,fontsize=11,fontweight='bold')
    ax.grid(axis='x',alpha=0.3)
fig.suptitle('Bootstrap ΔAUC 95% CI (2000 resamples)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig08_delta_auc.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig8 done')

# ── FIG 9: AUC/TSS fold violin ───────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,metric in zip(axes,['AUC','TSS']):
    for i,m in enumerate(ML):
        for res,col,off,lbl in [(rH,C['heur'],-0.3,'H'),(rS,C['sa'],-0.1,'SA'),(rG,C['gan'],0.1,'GAN'),(rR,C['rand'],0.3,'R')]:
            folds=res[m]['folds'] if metric=='AUC' else [res[m]['TSS']]
            vp=ax.violinplot([folds],positions=[i+off],widths=0.18,showmedians=True)
            for pc in vp['bodies']: pc.set_facecolor(col); pc.set_alpha(0.55)
            for pt in ['cbars','cmins','cmaxes','cmedians']:
                if pt in vp: vp[pt].set_color(col)
            ax.scatter(np.full(len(folds),i+off),folds,color=col,s=20,zorder=3)
    ax.set_xticks(np.arange(len(ML))); ax.set_xticklabels(ML,fontsize=9)
    ax.set_ylabel(metric,fontsize=11); ax.grid(axis='y',alpha=0.3)
    ax.set_title(f'{metric} Fold Distribution',fontsize=12,fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=C[k],label=l) for k,l in [('heur','Heuristic'),('sa','SA'),('gan','GAN'),('rand','Random')]],fontsize=9)
fig.suptitle('AUC & TSS Variability (Spatial Block CV)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig09_violin.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig9 done')

# ── FIG 10: Observed vs Predicted (calibration) scatter ─────────────────────
from sklearn.calibration import calibration_curve
fig,axes=plt.subplots(4,4,figsize=(18,16))
method_data=[(prH,'Heuristic',C['heur']),(prS,'SA',C['sa']),(prG,'GAN',C['gan']),(prR,'Random',C['rand'])]
for col_i,(pr,mlbl,mc) in enumerate(method_data):
    for row_i,nm in enumerate(ML):
        ax=axes[row_i,col_i]
        yte=pr[nm]['yte']; prob=pr[nm]['prob']
        # Scatter: jittered observed vs predicted
        jitter=np.random.uniform(-0.04,0.04,len(yte))
        ax.scatter(prob,yte+jitter,c=mc,alpha=0.08,s=3,rasterized=True)
        # Calibration curve
        if len(np.unique(yte))>1:
            frac_pos,mean_pred=calibration_curve(yte,prob,n_bins=10,strategy='quantile')
            ax.plot(mean_pred,frac_pos,'o-',color='black',lw=2,ms=4,label='Calibration')
        ax.plot([0,1],[0,1],'r--',lw=1.2,label='Perfect')
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.1,1.1)
        if row_i==0: ax.set_title(mlbl,fontsize=10,fontweight='bold',color=mc)
        if col_i==0: ax.set_ylabel(nm,fontsize=9,fontweight='bold')
        if row_i==3: ax.set_xlabel('Predicted P(fire)',fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
fig.suptitle('Observed vs Predicted — Calibration Plots (all methods × models)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig10_obs_pred.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig10 done')

# ── FIG 11: Predicted probability density per class ──────────────────────────
fig,axes=plt.subplots(4,4,figsize=(18,14))
for col_i,(pr,mlbl,mc) in enumerate(method_data):
    for row_i,nm in enumerate(ML):
        ax=axes[row_i,col_i]
        yte=pr[nm]['yte']; prob=pr[nm]['prob']
        ax.hist(prob[yte==1],bins=30,density=True,alpha=0.6,color='#d62728',label='Fire')
        ax.hist(prob[yte==0],bins=30,density=True,alpha=0.6,color=mc,label='Pseudo-abs')
        ax.set_xlim(0,1)
        if row_i==0: ax.set_title(mlbl,fontsize=10,fontweight='bold',color=mc)
        if col_i==0: ax.set_ylabel(nm,fontsize=9,fontweight='bold')
        if row_i==3: ax.set_xlabel('Predicted P(fire)',fontsize=8)
        ax.legend(fontsize=6,loc='upper center'); ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
fig.suptitle('Predicted Probability Distribution by Class (Fire vs PA)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig11_prob_dist.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig11 done')

# ── FIG 12: Spatial density hexbin ───────────────────────────────────────────
fig,axes=plt.subplots(1,4,figsize=(20,5))
for ax,(pts,lbl,col) in zip(axes,[(heur,'Heuristic',C['heur']),(sa,'SA',C['sa']),(gan,'GAN',C['gan']),(rand,'Random',C['rand'])]):
    hx=ax.hexbin(pts[:,0],pts[:,1],gridsize=22,cmap='Blues',mincnt=1)
    ax.scatter(fire[:,0],fire[:,1],c='red',s=2,alpha=0.5,label='Fire')
    ax.set_title(lbl,fontsize=11,fontweight='bold',color=col)
    ax.set_xlabel('Longitude'); ax.grid(alpha=0.15)
    plt.colorbar(hx,ax=ax,shrink=0.8,label='Count')
axes[0].set_ylabel('Latitude')
fig.suptitle('Spatial Density of Pseudo-Absences (n=5500 each)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig12_density_hex.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig12 done')

# ── FIG 13: AUC heatmap table ─────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,4))
methods=['Heuristic','SA','GAN','Random']
data=np.array([[rH[m]['AUC'] for m in ML],[rS[m]['AUC'] for m in ML],[rG[m]['AUC'] for m in ML],[rR[m]['AUC'] for m in ML]])
im=ax.imshow(data,cmap='YlOrRd',aspect='auto',vmin=0.88,vmax=1.0)
ax.set_xticks(range(len(ML))); ax.set_xticklabels(ML,fontsize=11)
ax.set_yticks(range(4)); ax.set_yticklabels(methods,fontsize=11)
for i in range(4):
    for j in range(len(ML)):
        ax.text(j,i,f'{data[i,j]:.4f}',ha='center',va='center',fontsize=11,fontweight='bold',
                color='white' if data[i,j]>0.96 else 'black')
plt.colorbar(im,ax=ax,shrink=0.8,label='AUC')
ax.set_title('AUC Heatmap — PA Method × ML Model',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig13_auc_heatmap.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig13 done')

# ── FIG 14: TSS heatmap ───────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,4))
data_t=np.array([[rH[m]['TSS'] for m in ML],[rS[m]['TSS'] for m in ML],[rG[m]['TSS'] for m in ML],[rR[m]['TSS'] for m in ML]])
im=ax.imshow(data_t,cmap='YlGn',aspect='auto',vmin=0.4,vmax=1.0)
ax.set_xticks(range(len(ML))); ax.set_xticklabels(ML,fontsize=11)
ax.set_yticks(range(4)); ax.set_yticklabels(methods,fontsize=11)
for i in range(4):
    for j in range(len(ML)):
        ax.text(j,i,f'{data_t[i,j]:.3f}',ha='center',va='center',fontsize=11,fontweight='bold',
                color='white' if data_t[i,j]>0.8 else 'black')
plt.colorbar(im,ax=ax,shrink=0.8,label='TSS')
ax.set_title('TSS Heatmap — PA Method × ML Model',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'fig14_tss_heatmap.png',dpi=150,bbox_inches='tight'); plt.close(); print('Fig14 done')
print('All figures done.')
