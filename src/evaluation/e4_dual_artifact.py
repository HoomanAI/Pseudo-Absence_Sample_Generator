"""E4: 2x2 audit of lattice and local-geometry construction artifacts."""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'Revision Code'/'results'; RETRAIN=ROOT/'Revision Code'/'retrain'
DATA=ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv'
sys.path.insert(0,str(ROOT/'Revision Code'))
import construction_audit as ca
import evaluate_arms as ea
SEED=42; CONDITIONS=['baseline','lattice','bw','lattice_bw']; MODELS=['LogReg','NaiveBayes','CART','KNN','SVM']

def paths(row):
    d=RETRAIN/f'{row.label}_seed{int(row.seed)}'; return d/f'pts_{row.arm}.npy',d/f'feats_{row.arm}.npy'

def main():
    df=pd.read_csv(DATA).dropna(); fire=df[['LONGITUDE','LATITUDE']].to_numpy(); fc=[c for c in df if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]; ff=df[fc].to_numpy(float)
    disc=[j for j,c in enumerate(fc) if df[c].nunique()<=ca.DISC_MAX_LEVELS]; cont=[j for j in range(len(fc)) if j not in disc]; levels={j:np.unique(ff[:,j]) for j in disc}
    man=pd.read_csv(RETRAIN/'manifest.csv'); first=man.iloc[0]; pp,fp=paths(first); lag0=cKDTree(fire).query(np.load(pp),k=1)[0]
    eq=ca.assert_bw_chunk_equivalence(fire,ff,lag0,cont,n_check=500,seed=SEED); print('chunk equivalence',eq,flush=True)
    checkpoint=RESULTS/'e4_checkpoint.csv'
    rows=pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() else []
    done={(r['arm'],r['condition'],int(r['seed'])) for r in rows}
    for cell,row in man.iterrows():
        if all((row.label,c,int(row.seed)) in done for c in CONDITIONS):
            print(f"skip complete cell {row.label} s{row.seed}",flush=True); continue
        pp,fp=paths(row); pts=np.load(pp); pa=np.load(fp); pa_snap=ca.snap_to_lattice(pa,levels); pa_lag=cKDTree(fire).query(pts,k=1)[0]
        # Same arm-seed-specific empirical lag distribution for C3 and C4.
        ffbw,fdist=ca.bw_matched_fire(fire,ff,pa_lag,cont,rng=np.random.default_rng(SEED))
        flag=fdist.mean(1)
        for cond in CONDITIONS:
            if (row.label,cond,int(row.seed)) in done:
                print(f"skip {row.label} s{row.seed} {cond}",flush=True); continue
            use_snap=cond in ('lattice','lattice_bw'); use_bw=cond in ('bw','lattice_bw')
            P=pa_snap if use_snap else pa; F=ffbw if use_bw else ff
            X=np.vstack([F,P]); y=np.r_[np.ones(len(F)),np.zeros(len(P))]; coords=np.vstack([fire,pts])
            L=ca.lattice_dev(X,levels); lattice_score=L.sum(1); lattice_auc=float(roc_auc_score(1-y,lattice_score)); onrate=float(np.mean(np.all(L[len(F):]==0,axis=1)))
            M=ca.geometry_meta(X); m1f,m1p=M[:len(F),0],M[len(F):,0]
            np.random.seed(SEED); geom=ca.run_cv(M,y,coords,subset=['RandomForest'])['RandomForest']['AUC']
            np.random.seed(SEED); cv,_=ea.run_cv(X,y,coords)
            base={'arm':row.label,'condition':cond,'seed':int(row.seed),'lattice_tell_AUC':lattice_auc,'geometry_tell_AUC':geom,
                  'm1_fire_median':float(np.median(m1f)),'m1_pa_median':float(np.median(m1p)),'m1_fire_over_pa':float(np.median(m1f)/np.median(m1p)),
                  'pa_lag_mean_deg':float(pa_lag.mean()),'fire_lag_mean_deg':float(flag.mean()) if use_bw else float(cKDTree(fire).query(fire,k=8)[0][:,1:].mean()),
                  'on_lattice_rate':onrate,'on_lattice_count':int(np.all(L[len(F):]==0,axis=1).sum()),
                  'lattice_probe_var_fire':float(np.var(lattice_score[:len(F)])),'lattice_probe_var_pa':float(np.var(lattice_score[len(F):])),
                  'max_snap_feature_diff':float(np.max(np.abs(pa-pa_snap)))}
            for model,(auc,auc_sd,tss,tss_sd) in cv.items():
                z=base|{'model':model,'AUC':auc,'fold_AUC_sd':auc_sd,'TSS':tss,'fold_TSS_sd':tss_sd}; rows.append(z)
            pd.DataFrame(rows).to_csv(checkpoint,index=False)
            print(f"{cell+1:02d}/24 {row.label} s{row.seed} {cond}: lattice={lattice_auc:.5f} geometry={geom:.5f} CART={cv['CART'][0]:.4f}",flush=True)
    out=pd.DataFrame(rows); out.to_csv(RESULTS/'e4_dual_artifact_wide.csv',index=False)
    idc=['arm','condition','seed','model']; valuec=[c for c in out.columns if c not in idc]
    out.melt(id_vars=idc,value_vars=valuec,var_name='metric',value_name='value').to_csv(RESULTS/'e4_dual_artifact.csv',index=False)
    # Summaries plus prespecified Welch tests across conditions and arms within C4.
    metric_cols=['lattice_tell_AUC','geometry_tell_AUC','m1_fire_over_pa','pa_lag_mean_deg','fire_lag_mean_deg','AUC','TSS']
    summ=out.groupby(['arm','condition','model'])[metric_cols].agg(['mean','std']).reset_index(); summ.columns=['_'.join(x).rstrip('_') for x in summ.columns]
    tests=[]
    for arm in man.label.unique():
        for model in MODELS:
            for metric in ['geometry_tell_AUC','m1_fire_over_pa','AUC','TSS']:
                for c1,c2 in [('baseline','lattice'),('baseline','bw'),('lattice','lattice_bw'),('bw','lattice_bw')]:
                    a=out[(out.arm==arm)&(out.model==model)&(out.condition==c1)][metric]; b=out[(out.arm==arm)&(out.model==model)&(out.condition==c2)][metric]; t,p=ttest_ind(a,b,equal_var=False)
                    tests.append({'test_scope':'condition','arm_a':arm,'arm_b':'','condition_a':c1,'condition_b':c2,'model':model,'metric':metric,'t':t,'p':p,'decision':'significant' if p<.05 else 'n.s.'})
    arms=list(man.label.unique())
    for i,a1 in enumerate(arms):
        for a2 in arms[i+1:]:
            for model in MODELS:
                for metric in ['geometry_tell_AUC','m1_fire_over_pa','AUC','TSS']:
                    a=out[(out.arm==a1)&(out.model==model)&(out.condition=='lattice_bw')][metric]; b=out[(out.arm==a2)&(out.model==model)&(out.condition=='lattice_bw')][metric]; t,p=ttest_ind(a,b,equal_var=False)
                    tests.append({'test_scope':'arms_C4','arm_a':a1,'arm_b':a2,'condition_a':'lattice_bw','condition_b':'lattice_bw','model':model,'metric':metric,'t':t,'p':p,'decision':'significant' if p<.05 else 'n.s.'})
    summ.insert(0,'record_type','summary')
    testdf=pd.DataFrame(tests); testdf.insert(0,'record_type','welch_test')
    pd.concat([summ,testdf],ignore_index=True,sort=False).to_csv(RESULTS/'e4_summary.csv',index=False)
    # Native AP invariance audit across every output measure.
    aps=['dagp_strict','dagp_base','dagp_relaxed']; inv=[]
    cols=['lattice_tell_AUC','geometry_tell_AUC','m1_fire_median','m1_pa_median','m1_fire_over_pa','AUC','TSS']
    for arm in aps:
        arm_feature_equal=True
        for rr in man[man.label==arm].itertuples():
            _,afp=paths(rr); A=np.load(afp); arm_feature_equal &= np.array_equal(A,ca.snap_to_lattice(A,levels))
        for c1,c2 in [('baseline','lattice'),('bw','lattice_bw')]:
            a=out[(out.arm==arm)&(out.condition==c1)].sort_values(['seed','model']); b=out[(out.arm==arm)&(out.condition==c2)].sort_values(['seed','model'])
            inv.append({'arm':arm,'comparison':f'{c1}=={c2}','max_abs_diff':float(np.max(np.abs(a[cols].to_numpy()-b[cols].to_numpy()))),'feature_array_equal':arm_feature_equal})
    pd.DataFrame(inv).to_csv(RESULTS/'e4_ap_invariance.csv',index=False)
    (RESULTS/'e4_equivalence.json').write_text(json.dumps(eq,indent=2),encoding='utf-8')
    print('complete',flush=True)
if __name__=='__main__': main()
