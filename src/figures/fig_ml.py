import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from scipy.interpolate import interp1d

np.random.seed(42)
C  = {'fire':'#d62728','heur':'#1f77b4','rand':'#2ca02c','sa':'#ff7f0e'}
MC = {'RandomForest':'#e377c2','XGBoost':'#ff7f0e','SVM':'#9467bd','KNN':'#17becf'}

df        = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire      = df[['LONGITUDE','LATITUDE']].values
fcols     = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats= df[fcols].values.astype(float)
fire_tree = cKDTree(fire)

heur_pts = np.load('../../outputs/heur_pts.npy')
rand_pts  = np.load('../../outputs/rand_pts.npy')
sa_pts    = np.load('../../outputs/sa_pts.npy')

def interp(pts,k=7):
    d,ix=fire_tree.query(pts,k=k); d=np.maximum(d,1e-9)
    w=1/d**2; w/=w.sum(1,keepdims=True)
    f=np.einsum('nk,nkf->nf',w,fire_feats[ix])
    f+=np.random.normal(0,0.05*fire_feats.std(0),f.shape); return f

hf=interp(heur_pts); rf=interp(rand_pts); sf=interp(sa_pts)

def sfolds(coords,n=5):
    lat=pd.qcut(coords[:,1],n,labels=False,duplicates='drop')
    lon=pd.qcut(coords[:,0],n,labels=False,duplicates='drop')
    bl=lat*n+lon; u=np.unique(bl); np.random.shuffle(u)
    g=np.array_split(u,n); fid=np.zeros(len(coords),int)
    for fi,gr in enumerate(g):
        for b in gr: fid[bl==b]=fi
    return fid

def run_cv(ab_pts,ab_feats,return_curves=False):
    X=np.vstack([fire_feats,ab_feats]); y=np.array([1]*len(fire)+[0]*len(ab_pts))
    coords=np.vstack([fire,ab_pts]); fid=sfolds(coords)
    models={'RandomForest':RandomForestClassifier(100,random_state=42,n_jobs=-1),
            'XGBoost':XGBClassifier(n_estimators=100,objective='binary:logistic',eval_metric='logloss',verbosity=0,random_state=42),
            'KNN':KNeighborsClassifier(9,n_jobs=-1),
            'SVM':SVC(kernel='rbf',probability=True,random_state=42)}
    out={}; roc_d={}
    for nm,clf in models.items():
        aucs,tsss,fps_all,tps_all=[],[],[],[]
        for fold in range(5):
            tr,te=fid!=fold,fid==fold
            if te.sum()<10 or len(np.unique(y[te]))<2: continue
            Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
            if nm=='SVM':
                i0=np.where(ytr==0)[0]; i1=np.where(ytr==1)[0]
                s=np.concatenate([np.random.choice(i0,min(500,len(i0)),False),
                                   np.random.choice(i1,min(500,len(i1)),False)])
                Xtr,ytr=Xtr[s],ytr[s]
            if nm in ('SVM','KNN'):
                sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
            clf.fit(Xtr,ytr); p=clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte,p))
            pred=(p>=0.5).astype(int)
            tp=((pred==1)&(yte==1)).sum(); tn=((pred==0)&(yte==0)).sum()
            fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
            tsss.append(tp/(tp+fn+1e-9)+tn/(tn+fp+1e-9)-1)
            if return_curves:
                fpr,tpr,_=roc_curve(yte,p); fps_all.append(fpr); tps_all.append(tpr)
        out[nm]={'AUC':np.mean(aucs),'AUC_std':np.std(aucs),
                 'TSS':np.mean(tsss),'TSS_std':np.std(tsss),'folds':aucs}
        if return_curves: roc_d[nm]=(fps_all,tps_all)
    return (out,roc_d) if return_curves else out

print('Running CV …')
rh,roc_h=run_cv(heur_pts,hf,return_curves=True)
rr,roc_r=run_cv(rand_pts,rf,return_curves=True)
rs,roc_s=run_cv(sa_pts,  sf,return_curves=True)
print('Done.')

ML = list(rh.keys())

# ══════════════════════════════════════════════════════
# FIG 7 – AUC grouped bar chart (3 methods)
# ══════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(11,6))
x=np.arange(len(ML)); w=0.26
for offset,res,col,lbl in zip([-w,0,w],
        [rh,rs,rr],[C['heur'],C['sa'],C['rand']],
        ['Heuristic PA','SA PA','Random PA']):
    vals=[res[m]['AUC'] for m in ML]; errs=[res[m]['AUC_std'] for m in ML]
    bars=ax.bar(x+offset,vals,w,yerr=errs,capsize=5,color=col,label=lbl,alpha=0.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
                f'{v:.4f}',ha='center',va='bottom',fontsize=7.5,color=col,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(ML,fontsize=11)
ax.set_ylabel('AUC (spatial block CV, 5-fold)',fontsize=11)
ax.set_title('AUC — Heuristic vs SA vs Random Pseudo-Absences\n(error bars = ±1 std)',fontsize=12,fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0.82,1.015); ax.grid(axis='y',alpha=0.35)
ax.axhline(1.0,color='gray',lw=1,ls='--',alpha=0.5)
plt.tight_layout()
plt.savefig('../../outputs/fig07_auc_bar.png',dpi=150,bbox_inches='tight')
plt.close(); print('Fig 7 done')

# ══════════════════════════════════════════════════════
# FIG 8 – TSS grouped bar chart
# ══════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(11,6))
for offset,res,col,lbl in zip([-w,0,w],
        [rh,rs,rr],[C['heur'],C['sa'],C['rand']],
        ['Heuristic PA','SA PA','Random PA']):
    vals=[res[m]['TSS'] for m in ML]; errs=[res[m]['TSS_std'] for m in ML]
    bars=ax.bar(x+offset,vals,w,yerr=errs,capsize=5,color=col,label=lbl,alpha=0.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,max(bar.get_height(),0)+0.01,
                f'{v:.3f}',ha='center',va='bottom',fontsize=7.5,color=col,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(ML,fontsize=11)
ax.set_ylabel('TSS (Sens + Spec − 1)',fontsize=11)
ax.set_title('TSS — Heuristic vs SA vs Random Pseudo-Absences',fontsize=12,fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0,1.1); ax.grid(axis='y',alpha=0.35)
ax.axhline(1.0,color='gray',lw=1,ls='--',alpha=0.5)
plt.tight_layout()
plt.savefig('../../outputs/fig08_tss_bar.png',dpi=150,bbox_inches='tight')
plt.close(); print('Fig 8 done')

# ══════════════════════════════════════════════════════
# FIG 9 – ΔAUC Bootstrap CI (pairwise: H−R, SA−R, H−SA)
# ══════════════════════════════════════════════════════
def bdelta(a,b,n=2000):
    np.random.seed(42)
    d=np.array([np.mean(np.random.choice(a,len(a),True))-np.mean(np.random.choice(b,len(b),True)) for _ in range(n)])
    return float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

pairs=[('H−R',rh,rr,C['heur']),('SA−R',rs,rr,C['sa']),('H−SA',rh,rs,'#8c564b')]
fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
for ax,(pair_lbl,ra,rb,pc) in zip(axes,pairs):
    dm_list,lo_list,hi_list,colors=[],[],[],[]
    for m in ML:
        dm,lo,hi=bdelta(ra[m]['folds'],rb[m]['folds'])
        dm_list.append(dm); lo_list.append(lo); hi_list.append(hi)
        colors.append('#2ca02c' if dm>0 else '#d62728')
    yp=np.arange(len(ML))
    err_lo = np.maximum(np.array(dm_list)-np.array(lo_list), 0)
    err_hi = np.maximum(np.array(hi_list)-np.array(dm_list), 0)
    ax.barh(yp,dm_list,xerr=[err_lo, err_hi],
            capsize=5,color=colors,alpha=0.8,height=0.5)
    ax.axvline(0,color='black',lw=1.5,ls='--')
    for i,(m,dm,lo,hi) in enumerate(zip(ML,dm_list,lo_list,hi_list)):
        ax.text(max(hi,0)+0.001,i,f'Δ={dm:+.4f}',va='center',fontsize=8)
    ax.set_yticks(yp); ax.set_yticklabels(ML,fontsize=10)
    ax.set_xlabel(f'ΔAUC ({pair_lbl})',fontsize=10)
    ax.set_title(pair_lbl,fontsize=12,fontweight='bold')
    ax.grid(axis='x',alpha=0.3)
fig.suptitle('Bootstrap ΔAUC 95% CI (2000 resamples)\n+ favors left method, − favors right',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig09_delta_auc_ci.png',dpi=150,bbox_inches='tight')
plt.close(); print('Fig 9 done')

# ══════════════════════════════════════════════════════
# FIG 10 – ROC curves (3 panels)
# ══════════════════════════════════════════════════════
common_fpr=np.linspace(0,1,200)
fig,axes=plt.subplots(1,3,figsize=(18,6))
for ax,roc_data,res,lbl,pc in zip(axes,
        [roc_h,roc_s,roc_r],[rh,rs,rr],
        ['Heuristic PA','SA PA','Random PA'],
        [C['heur'],C['sa'],C['rand']]):
    for nm,mc in MC.items():
        if nm not in roc_data: continue
        fps_l,tps_l=roc_data[nm]; itprs=[]
        for fpr_f,tpr_f in zip(fps_l,tps_l):
            f=interp1d(fpr_f,tpr_f,bounds_error=False,fill_value=(0,1))
            itprs.append(f(common_fpr))
        mt=np.mean(itprs,0); st=np.std(itprs,0)
        ax.plot(common_fpr,mt,color=mc,lw=2.2,label=f'{nm} (AUC={res[nm]["AUC"]:.4f})')
        ax.fill_between(common_fpr,mt-st,mt+st,alpha=0.12,color=mc)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Chance')
    ax.set_xlabel('FPR',fontsize=10); ax.set_ylabel('TPR',fontsize=10)
    ax.set_title(f'ROC — {lbl}',fontsize=11,fontweight='bold',color=pc)
    ax.legend(fontsize=8,loc='lower right'); ax.grid(True,alpha=0.3)
fig.suptitle('ROC Curves: Spatial Block CV (shaded = ±1 std)',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig10_roc_curves.png',dpi=150,bbox_inches='tight')
plt.close(); print('Fig 10 done')

# ══════════════════════════════════════════════════════
# FIG 11 – Fold-level AUC violin
# ══════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,metric in zip(axes,['AUC','TSS']):
    for i,m in enumerate(ML):
        fh=rh[m]['folds'] if metric=='AUC' else [rh[m]['TSS']]
        fs=rs[m]['folds'] if metric=='AUC' else [rs[m]['TSS']]
        fr=rr[m]['folds'] if metric=='AUC' else [rr[m]['TSS']]
        for folds,col,off in [(fh,C['heur'],-0.27),(fs,C['sa'],0),(fr,C['rand'],0.27)]:
            vp=ax.violinplot([folds],positions=[i+off],widths=0.24,showmedians=True)
            for pc in vp['bodies']: pc.set_facecolor(col); pc.set_alpha(0.55)
            for part in ['cbars','cmins','cmaxes','cmedians']:
                if part in vp: vp[part].set_color(col)
            ax.scatter(np.full(len(folds),i+off),folds,color=col,s=25,zorder=3)
    ax.set_xticks(np.arange(len(ML))); ax.set_xticklabels(ML,fontsize=10)
    ax.set_ylabel(metric,fontsize=11); ax.grid(axis='y',alpha=0.3)
    ax.set_title(f'{metric} Distribution Across Folds',fontsize=12,fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=C[k],label=l)
                        for k,l in [('heur','Heuristic'),('sa','SA'),('rand','Random')]],
              fontsize=9)
fig.suptitle('AUC & TSS Fold Variability (Spatial Block CV)',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('../../outputs/fig11_fold_violin.png',dpi=150,bbox_inches='tight')
plt.close(); print('Fig 11 done')

print('All ML figures saved.')
