"""ML CV for SA + GAN."""
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire=df[['LONGITUDE','LATITUDE']].values
fcols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[fcols].values.astype(float)
OUT='../../outputs/'

def sfolds(coords,n=5):
    lat=pd.qcut(coords[:,1],n,labels=False,duplicates='drop')
    lon=pd.qcut(coords[:,0],n,labels=False,duplicates='drop')
    bl=lat*n+lon; u=np.unique(bl); np.random.shuffle(u)
    g=np.array_split(u,n); fid=np.zeros(len(coords),int)
    for fi,gr in enumerate(g):
        for b in gr: fid[bl==b]=fi
    return fid

def run_cv(ab_pts,ab_feats):
    X=np.vstack([fire_feats,ab_feats]); y=np.array([1]*len(fire)+[0]*len(ab_pts))
    coords=np.vstack([fire,ab_pts]); fid=sfolds(coords)
    models={'RandomForest':RandomForestClassifier(80,random_state=42,n_jobs=-1),
            'XGBoost':XGBClassifier(n_estimators=80,objective='binary:logistic',eval_metric='logloss',verbosity=0,random_state=42),
            'KNN':KNeighborsClassifier(9,n_jobs=-1),
            'SVM':SVC(kernel='rbf',probability=True,random_state=42)}
    out={}; roc_d={}; proba_d={}
    for nm,clf in models.items():
        aucs,tsss,fps_all,tps_all=[],[],[],[]; ayte=[]; aprob=[]
        for fold in range(5):
            tr,te=fid!=fold,fid==fold
            if te.sum()<10 or len(np.unique(y[te]))<2: continue
            Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
            if nm=='SVM':
                i0=np.where(ytr==0)[0]; i1=np.where(ytr==1)[0]
                s=np.concatenate([np.random.choice(i0,min(500,len(i0)),False),np.random.choice(i1,min(500,len(i1)),False)])
                Xtr,ytr=Xtr[s],ytr[s]
            if nm in('SVM','KNN'):
                sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
            clf.fit(Xtr,ytr); p=clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte,p))
            pred=(p>=0.5).astype(int)
            tp=((pred==1)&(yte==1)).sum(); tn=((pred==0)&(yte==0)).sum()
            fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
            tsss.append(tp/(tp+fn+1e-9)+tn/(tn+fp+1e-9)-1)
            fpr,tpr,_=roc_curve(yte,p); fps_all.append(fpr); tps_all.append(tpr)
            ayte.extend(yte.tolist()); aprob.extend(p.tolist())
        out[nm]={'AUC':np.mean(aucs),'AUC_std':np.std(aucs),'TSS':np.mean(tsss),'TSS_std':np.std(tsss),'folds':aucs}
        roc_d[nm]=(fps_all,tps_all); proba_d[nm]={'yte':np.array(ayte),'prob':np.array(aprob)}
    return out,roc_d,proba_d

sa_pts=np.load(OUT+'sa_pts.npy'); sa_feats=np.load(OUT+'sa_feats.npy')
gan_pts=np.load(OUT+'gan_pts.npy'); gan_feats=np.load(OUT+'gan_feats.npy')
rs,roc_s,pr_s=run_cv(sa_pts,sa_feats);   print('SA done')
rg,roc_g,pr_g=run_cv(gan_pts,gan_feats); print('GAN done')
np.save(OUT+'cv_s.npy',{'res':rs,'roc':roc_s,'pr':pr_s},allow_pickle=True)
np.save(OUT+'cv_g.npy',{'res':rg,'roc':roc_g,'pr':pr_g},allow_pickle=True)
print('Part2 saved.')
