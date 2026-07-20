"""Stage 2: ML evaluation"""
import numpy as np
import pandas as pd
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
df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts  = df[['LONGITUDE','LATITUDE']].values
feat_cols = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats = df[feat_cols].values
fire_tree  = cKDTree(fire_pts)

heur_pts = np.load('../../outputs/heur_pts.npy')
rand_pts  = np.load('../../outputs/rand_pts.npy')

# IDW feature interpolation
def interp(new_pts, k=7):
    d,idx = fire_tree.query(new_pts,k=k)
    d = np.maximum(d,1e-9); w=1/d**2; w/=w.sum(1,keepdims=True)
    feats = np.einsum('nk,nkf->nf',w,fire_feats[idx].reshape(len(new_pts),k,-1))
    feats += np.random.normal(0,0.05*fire_feats.std(0),feats.shape)
    return feats

# Spatial block CV
def block_folds(coords,n=5):
    lb = pd.qcut(coords[:,1],n,labels=False,duplicates='drop')
    lo = pd.qcut(coords[:,0],n,labels=False,duplicates='drop')
    bl = lb*n+lo; uniq=np.unique(bl); np.random.shuffle(uniq)
    grps=np.array_split(uniq,n); fid=np.zeros(len(coords),dtype=int)
    for fi,g in enumerate(grps):
        for b in g: fid[bl==b]=fi
    return fid

def run_ml(ab_pts, ab_feats):
    X=np.vstack([fire_feats,ab_feats]); y=np.array([1]*len(fire_pts)+[0]*len(ab_pts))
    coords=np.vstack([fire_pts,ab_pts]); fid=block_folds(coords)
    models={'RandomForest': RandomForestClassifier(100,random_state=42,n_jobs=-1),
            'SVM':          SVC(kernel='rbf',probability=True,random_state=42),
            'KNN':          KNeighborsClassifier(9,n_jobs=-1)}
    if HAS_XGB: models['XGBoost']=XGBClassifier(n_estimators=100,objective='binary:logistic',eval_metric='logloss',verbosity=0,random_state=42)
    out={}
    for name,clf in models.items():
        aucs,tsss=[],[]
        for fold in range(5):
            tr,te=fid!=fold,fid==fold
            if te.sum()<10 or len(np.unique(y[te]))<2: continue
            Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
            if name=='SVM':  # subsample for speed
                idx0=np.where(ytr==0)[0]; idx1=np.where(ytr==1)[0]
                s0=idx0[np.random.choice(len(idx0),min(600,len(idx0)),False)]
                s1=idx1[np.random.choice(len(idx1),min(600,len(idx1)),False)]
                si=np.concatenate([s0,s1]); Xtr,ytr=Xtr[si],ytr[si]
            if name in ('SVM','KNN'):
                sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
            clf.fit(Xtr,ytr)
            prob=clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte,prob))
            pred=(prob>=0.5).astype(int)
            tp=((pred==1)&(yte==1)).sum(); tn=((pred==0)&(yte==0)).sum()
            fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
            tsss.append(tp/(tp+fn+1e-9)+tn/(tn+fp+1e-9)-1)
        out[name]={'AUC':np.mean(aucs),'AUC_std':np.std(aucs),
                   'TSS':np.mean(tsss),'TSS_std':np.std(tsss),'folds':aucs}
        print(f"  {name:14}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  TSS={np.mean(tsss):.4f}")
    return out

def bootstrap_delta(a,b,n=1000):
    d=np.array([np.mean(np.random.choice(a,len(a),True))-np.mean(np.random.choice(b,len(b),True)) for _ in range(n)])
    ci=(np.percentile(d,2.5),np.percentile(d,97.5))
    pval=2*min((d<=0).mean(),(d>=0).mean())
    return float(d.mean()),ci,float(pval)

print("Interpolating features...")
hf=interp(heur_pts); rf=interp(rand_pts)

print("\n[Heuristic]")
res_h=run_ml(heur_pts,hf)
print("\n[Random]")
res_r=run_ml(rand_pts,rf)

rows=[]
for m in res_h:
    d,ci,pv=bootstrap_delta(res_h[m]['folds'],res_r[m]['folds'])
    w="Heuristic✓" if d>0 and pv<0.05 else ("Random✓" if d<0 and pv<0.05 else "Tie")
    rows.append({'Model':m,'AUC_Heuristic':round(res_h[m]['AUC'],4),'AUC_Heuristic_std':round(res_h[m]['AUC_std'],4),
                 'AUC_Random':round(res_r[m]['AUC'],4),'AUC_Random_std':round(res_r[m]['AUC_std'],4),
                 'ΔAUC':round(d,4),'CI_lo':round(ci[0],4),'CI_hi':round(ci[1],4),'p_value':round(pv,4),
                 'TSS_Heuristic':round(res_h[m]['TSS'],4),'TSS_Random':round(res_r[m]['TSS'],4),'Winner':w})
ml_df=pd.DataFrame(rows)
print("\nCOMPARISON TABLE:")
print(ml_df[['Model','AUC_Heuristic','AUC_Random','ΔAUC','p_value','TSS_Heuristic','TSS_Random','Winner']].to_string(index=False))
ml_df.to_csv('../../outputs/ml_results.csv',index=False)
np.save('../../outputs/svm_results.npy',np.array([res_h.get('SVM',{}).get('AUC',0),res_r.get('SVM',{}).get('AUC',0)]))
print("\nStage 2 done.")
