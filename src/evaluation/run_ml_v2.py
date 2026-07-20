"""ML evaluation for all 4 PA methods (5500 pts each). Saves ml_results.csv + .npy arrays."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire = df[['LONGITUDE','LATITUDE']].values
fcols = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats = df[fcols].values.astype(float)
fire_tree = cKDTree(fire)
OUT = '../../outputs/'

heur_pts  = np.load(OUT+'heur_pts.npy');  heur_feats = np.load(OUT+'heur_feats.npy')
rand_pts  = np.load(OUT+'rand_pts.npy');  rand_feats = np.load(OUT+'rand_feats.npy')
sa_pts    = np.load(OUT+'sa_pts.npy');    sa_feats   = np.load(OUT+'sa_feats.npy')
gan_pts   = np.load(OUT+'gan_pts.npy');   gan_feats  = np.load(OUT+'gan_feats.npy')

def sfolds(coords, n=5):
    lat = pd.qcut(coords[:,1],n,labels=False,duplicates='drop')
    lon = pd.qcut(coords[:,0],n,labels=False,duplicates='drop')
    bl = lat*n+lon; u = np.unique(bl); np.random.shuffle(u)
    g = np.array_split(u,n); fid = np.zeros(len(coords),int)
    for fi,gr in enumerate(g):
        for b in gr: fid[bl==b]=fi
    return fid

def run_cv(ab_pts, ab_feats, return_curves=False):
    X = np.vstack([fire_feats, ab_feats]); y = np.array([1]*len(fire)+[0]*len(ab_pts))
    coords = np.vstack([fire, ab_pts]); fid = sfolds(coords)
    models = {
        'RandomForest': RandomForestClassifier(100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, objective='binary:logistic',
                                  eval_metric='logloss', verbosity=0, random_state=42),
        'KNN': KNeighborsClassifier(9, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    out = {}; roc_d = {}; proba_d = {}
    for nm, clf in models.items():
        aucs,tsss,fps_all,tps_all=[],[],[],[]; all_yte=[]; all_prob=[]
        for fold in range(5):
            tr,te = fid!=fold, fid==fold
            if te.sum()<10 or len(np.unique(y[te]))<2: continue
            Xtr,ytr,Xte,yte = X[tr],y[tr],X[te],y[te]
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
            all_yte.extend(yte.tolist()); all_prob.extend(p.tolist())
            if return_curves:
                fpr,tpr,_=roc_curve(yte,p); fps_all.append(fpr); tps_all.append(tpr)
        out[nm]={'AUC':np.mean(aucs),'AUC_std':np.std(aucs),
                 'TSS':np.mean(tsss),'TSS_std':np.std(tsss),'folds':aucs}
        proba_d[nm]={'yte':np.array(all_yte),'prob':np.array(all_prob)}
        if return_curves: roc_d[nm]=(fps_all,tps_all)
    return (out,roc_d,proba_d) if return_curves else (out,{},proba_d)

print('Running CV for all 4 methods ...')
rh,roc_h,pr_h = run_cv(heur_pts, heur_feats, return_curves=True); print('Heuristic done')
rr,roc_r,pr_r = run_cv(rand_pts,  rand_feats, return_curves=True); print('Random done')
rs,roc_s,pr_s = run_cv(sa_pts,    sa_feats,   return_curves=True); print('SA done')
rg,roc_g,pr_g = run_cv(gan_pts,   gan_feats,  return_curves=True); print('GAN done')

ML = list(rh.keys())

# Bootstrap delta
def bdelta(a,b,n=2000):
    np.random.seed(42)
    d=np.array([np.mean(np.random.choice(a,len(a),True))-np.mean(np.random.choice(b,len(b),True)) for _ in range(n)])
    return float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

rows=[]
for m in ML:
    dm_hr,lo_hr,hi_hr=bdelta(rh[m]['folds'],rr[m]['folds'])
    dm_sr,lo_sr,hi_sr=bdelta(rs[m]['folds'],rr[m]['folds'])
    dm_gr,lo_gr,hi_gr=bdelta(rg[m]['folds'],rr[m]['folds'])
    p_hr=1-np.mean(np.array([np.mean(np.random.choice(rh[m]['folds'],len(rh[m]['folds']),True))-
                              np.mean(np.random.choice(rr[m]['folds'],len(rr[m]['folds']),True))>0 for _ in range(2000)]))
    p_gr=1-np.mean(np.array([np.mean(np.random.choice(rg[m]['folds'],len(rg[m]['folds']),True))-
                              np.mean(np.random.choice(rr[m]['folds'],len(rr[m]['folds']),True))>0 for _ in range(2000)]))
    rows.append({'Model':m,
        'AUC_H':round(rh[m]['AUC'],4),'AUC_R':round(rr[m]['AUC'],4),
        'AUC_SA':round(rs[m]['AUC'],4),'AUC_GAN':round(rg[m]['AUC'],4),
        'TSS_H':round(rh[m]['TSS'],3),'TSS_R':round(rr[m]['TSS'],3),
        'TSS_SA':round(rs[m]['TSS'],3),'TSS_GAN':round(rg[m]['TSS'],3),
        'dAUC_H-R':round(dm_hr,4),'p_H-R':round(p_hr,3),
        'dAUC_GAN-R':round(dm_gr,4),'p_GAN-R':round(p_gr,3)})

ml_df=pd.DataFrame(rows)
print(ml_df[['Model','AUC_H','AUC_SA','AUC_GAN','AUC_R','TSS_H','TSS_SA','TSS_GAN','TSS_R']].to_string(index=False))
ml_df.to_csv(OUT+'ml_results.csv',index=False)

np.save(OUT+'roc_h.npy', roc_h, allow_pickle=True)
np.save(OUT+'roc_r.npy', roc_r, allow_pickle=True)
np.save(OUT+'roc_s.npy', roc_s, allow_pickle=True)
np.save(OUT+'roc_g.npy', roc_g, allow_pickle=True)
np.save(OUT+'pr_h.npy',  pr_h,  allow_pickle=True)
np.save(OUT+'pr_r.npy',  pr_r,  allow_pickle=True)
np.save(OUT+'pr_s.npy',  pr_s,  allow_pickle=True)
np.save(OUT+'pr_g.npy',  pr_g,  allow_pickle=True)
np.save(OUT+'ml_res.npy', {'H':rh,'R':rr,'SA':rs,'GAN':rg}, allow_pickle=True)
print('ML results saved.')
