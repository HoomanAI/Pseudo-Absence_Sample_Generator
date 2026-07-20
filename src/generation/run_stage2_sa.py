"""Stage 2 (SA edition): ML evaluation – Random vs Heuristic vs SA"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

try: from xgboost import XGBClassifier; HAS_XGB=True
except: HAS_XGB=False

np.random.seed(42)
df        = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts  = df[['LONGITUDE','LATITUDE']].values
feat_cols = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats= df[feat_cols].values
fire_tree = cKDTree(fire_pts)

heur_pts = np.load('../../outputs/heur_pts.npy')
rand_pts  = np.load('../../outputs/rand_pts.npy')
sa_pts    = np.load('../../outputs/sa_pts.npy')

def interp(pts, k=7):
    d,ix = fire_tree.query(pts,k=k); d=np.maximum(d,1e-9)
    w=1/d**2; w/=w.sum(1,keepdims=True)
    f=np.einsum('nk,nkf->nf',w,fire_feats[ix].reshape(len(pts),k,-1))
    f+=np.random.normal(0,0.05*fire_feats.std(0),f.shape)
    return f

def block_folds(coords, n=5):
    lb=pd.qcut(coords[:,1],n,labels=False,duplicates='drop')
    lo=pd.qcut(coords[:,0],n,labels=False,duplicates='drop')
    bl=lb*n+lo; uniq=np.unique(bl); np.random.shuffle(uniq)
    grps=np.array_split(uniq,n); fid=np.zeros(len(coords),int)
    for fi,g in enumerate(grps):
        for b in g: fid[bl==b]=fi
    return fid

def run_ml(ab_pts, ab_feats, label):
    print(f"\n  [{label}]")
    X=np.vstack([fire_feats,ab_feats]); y=np.array([1]*len(fire_pts)+[0]*len(ab_pts))
    coords=np.vstack([fire_pts,ab_pts]); fid=block_folds(coords)
    models={'RandomForest':RandomForestClassifier(100,random_state=42,n_jobs=-1),
            'SVM':SVC(kernel='rbf',probability=True,random_state=42),
            'KNN':KNeighborsClassifier(9,n_jobs=-1)}
    if HAS_XGB:
        models['XGBoost']=XGBClassifier(n_estimators=100,objective='binary:logistic',
                                         eval_metric='logloss',verbosity=0,random_state=42)
    out={}
    for nm,clf in models.items():
        aucs,tsss=[],[]
        for fold in range(5):
            tr,te=fid!=fold,fid==fold
            if te.sum()<10 or len(np.unique(y[te]))<2: continue
            Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
            if nm=='SVM':
                i0=np.where(ytr==0)[0]; i1=np.where(ytr==1)[0]
                s=np.concatenate([np.random.choice(i0,min(600,len(i0)),False),
                                   np.random.choice(i1,min(600,len(i1)),False)])
                Xtr,ytr=Xtr[s],ytr[s]
            if nm in ('SVM','KNN'):
                sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
            clf.fit(Xtr,ytr)
            p=clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte,p))
            pred=(p>=0.5).astype(int)
            tp=((pred==1)&(yte==1)).sum(); tn=((pred==0)&(yte==0)).sum()
            fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
            tsss.append(tp/(tp+fn+1e-9)+tn/(tn+fp+1e-9)-1)
        out[nm]={'AUC':np.mean(aucs),'AUC_std':np.std(aucs),
                 'TSS':np.mean(tsss),'TSS_std':np.std(tsss),'folds':aucs}
        print(f"    {nm:14}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  TSS={np.mean(tsss):.4f}")
    return out

def bootstrap_delta(a, b, n=2000):
    np.random.seed(42)
    d=np.array([np.mean(np.random.choice(a,len(a),True))-
                np.mean(np.random.choice(b,len(b),True)) for _ in range(n)])
    ci=(np.percentile(d,2.5),np.percentile(d,97.5))
    pval=2*min((d<=0).mean(),(d>=0).mean())
    return float(d.mean()),ci,float(pval)

print("Interpolating features …")
hf=interp(heur_pts); rf=interp(rand_pts); sf=interp(sa_pts)
np.save('../../outputs/heur_feats.npy', hf)
np.save('../../outputs/rand_feats.npy', rf)
np.save('../../outputs/sa_feats.npy',   sf)

res_h = run_ml(heur_pts, hf, 'Heuristic')
res_r = run_ml(rand_pts, rf, 'Random')
res_s = run_ml(sa_pts,   sf, 'SA')

# Build comparison table
rows=[]
for m in res_h:
    d_hr,ci_hr,pv_hr=bootstrap_delta(res_h[m]['folds'],res_r[m]['folds'])
    d_hs,ci_hs,pv_hs=bootstrap_delta(res_h[m]['folds'],res_s[m]['folds'])
    d_sr,ci_sr,pv_sr=bootstrap_delta(res_s[m]['folds'],res_r[m]['folds'])
    rows.append({
        'Model':m,
        'AUC_Heuristic':round(res_h[m]['AUC'],4),'AUC_Heuristic_std':round(res_h[m]['AUC_std'],4),
        'AUC_Random':round(res_r[m]['AUC'],4),    'AUC_Random_std':round(res_r[m]['AUC_std'],4),
        'AUC_SA':round(res_s[m]['AUC'],4),         'AUC_SA_std':round(res_s[m]['AUC_std'],4),
        'ΔAUC_H_vs_R':round(d_hr,4),'CI_H_R_lo':round(ci_hr[0],4),'CI_H_R_hi':round(ci_hr[1],4),'p_H_R':round(pv_hr,4),
        'ΔAUC_H_vs_S':round(d_hs,4),'CI_H_S_lo':round(ci_hs[0],4),'CI_H_S_hi':round(ci_hs[1],4),'p_H_S':round(pv_hs,4),
        'ΔAUC_S_vs_R':round(d_sr,4),'CI_S_R_lo':round(ci_sr[0],4),'CI_S_R_hi':round(ci_sr[1],4),'p_S_R':round(pv_sr,4),
        'TSS_Heuristic':round(res_h[m]['TSS'],4),
        'TSS_Random':round(res_r[m]['TSS'],4),
        'TSS_SA':round(res_s[m]['TSS'],4),
    })

ml_df=pd.DataFrame(rows)
print("\n── COMPARISON TABLE ──")
print(ml_df[['Model','AUC_Heuristic','AUC_Random','AUC_SA',
             'ΔAUC_H_vs_R','p_H_R','ΔAUC_H_vs_S','p_H_S','ΔAUC_S_vs_R','p_S_R',
             'TSS_Heuristic','TSS_Random','TSS_SA']].to_string(index=False))

ml_df.to_csv('../../outputs/ml_results.csv', index=False)
print("\nStage 2 (SA edition) done.")
