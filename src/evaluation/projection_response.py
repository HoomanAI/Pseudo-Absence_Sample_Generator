"""Project 0..13 reclassified features and measure downstream response.

Resumable: one row per arm/seed/subset/p/model is appended after every cell.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import evaluate_arms as ea

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
DATA=ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv'
RETRAIN=HERE/'retrain'; DEFAULT_OUT=HERE/'results'/'projection_response.csv'
MODELS={'LogReg':lambda:LogisticRegression(max_iter=2000),'NaiveBayes':lambda:GaussianNB(),'CART':lambda:DecisionTreeClassifier(max_depth=6,min_samples_leaf=20,random_state=42)}

def score(X,y,coords):
    np.random.seed(42); folds=ea.sfolds(coords); out={}
    for name,maker in MODELS.items():
        aucs=[]; tsss=[]
        for f in range(5):
            tr,te=folds!=f,folds==f
            sc=StandardScaler().fit(X[tr]); clf=maker().fit(sc.transform(X[tr]),y[tr]); prob=clf.predict_proba(sc.transform(X[te]))[:,1]
            aucs.append(roc_auc_score(y[te],prob)); pred=prob>=.5; yt=y[te]; tsss.append(((pred& (yt==1)).sum()/max((yt==1).sum(),1))+(((~pred)&(yt==0)).sum()/max((yt==0).sum(),1))-1)
        out[name]=(float(np.mean(aucs)),float(np.mean(tsss)))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=DEFAULT_OUT); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    df=pd.read_csv(DATA).dropna(); meta=['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']; fc=[c for c in df if c not in meta]; ff=df[fc].to_numpy(float); fire=df[['LONGITUDE','LATITUDE']].to_numpy(); disc=[j for j,c in enumerate(fc) if df[c].nunique()<=12]
    assert len(disc)==13, f'expected 13 reclassified dimensions, found {len(disc)}'; levels={j:np.unique(ff[:,j]) for j in disc}; man=pd.read_csv(RETRAIN/'manifest.csv')
    subsets=[]
    for p in range(14):
        n=1 if p in (0,13) else 5
        for sid in range(n): subsets.append((p,sid,np.random.default_rng(42000+p*10+sid).choice(disc,p,replace=False)))
    total=len(man)*len(subsets); print(f'{len(man)} arm-seed cells x {len(subsets)} subsets = {total} evaluation cells ({total*3*5} classifier fits)')
    if args.dry_run:return
    old=pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(); done=set(zip(old.arm,old.seed,old.subset_id,old.p)) if len(old) else set(); rows=old.to_dict('records')
    for rr in man.itertuples():
        run=RETRAIN/f'{rr.label}_seed{rr.seed}'; pa=np.load(run/f'feats_{rr.arm}.npy'); pts=np.load(run/f'pts_{rr.arm}.npy')
        for p,sid,chosen in subsets:
            if (rr.label,rr.seed,sid,p) in done: continue
            B=pa.copy()
            for j in chosen: B[:,j]=levels[j][np.argmin(np.abs(B[:,[j]]-levels[j][None,:]),axis=1)]
            X=np.vstack([ff,B]); y=np.r_[np.ones(len(ff),int),np.zeros(len(B),int)]; coords=np.vstack([fire,pts]); tell=ea.tell_auc(X,y,levels); vals=score(X,y,coords)
            for model,(auc,tss) in vals.items(): rows.append({'arm':rr.label,'seed':rr.seed,'subset_id':sid,'p':p,'model':model,'AUC':auc,'TSS':tss,'tell_AUC':tell})
            pd.DataFrame(rows).to_csv(args.out,index=False); print(rr.label,rr.seed,p,sid,flush=True)
if __name__=='__main__': main()
