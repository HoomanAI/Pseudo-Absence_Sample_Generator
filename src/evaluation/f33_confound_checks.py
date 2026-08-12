"""Diagnose whether F33 correlation differences track lattice or dispersion."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RETRAIN=HERE/'retrain'; OUT=HERE/'results'
df=pd.read_csv(ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv').dropna(); meta=['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']; fc=[c for c in df if c not in meta]; F=df[fc].to_numpy(float); disc=np.array([df[c].nunique()<=12 for c in fc]); man=pd.read_csv(RETRAIN/'manifest.csv'); C=np.corrcoef(F,rowvar=False)
rows=[]
for arm,g in man.groupby('label',sort=False):
    vals=[]
    for r in g.itertuples():
        A=np.load(RETRAIN/f'{r.label}_seed{r.seed}'/f'feats_{r.arm}.npy'); D=C-np.corrcoef(A,rowvar=False); centroid=A.mean(0); vals.append({'full':np.linalg.norm(D,'fro'),'reclassified':np.linalg.norm(D[np.ix_(disc,disc)],'fro'),'continuous':np.linalg.norm(D[np.ix_(~disc,~disc)],'fro'),'mean_centroid_distance':np.linalg.norm(A-centroid,axis=1).mean(),'total_variance':A.var(axis=0,ddof=1).sum()})
    z=pd.DataFrame(vals).mean(); rows.append({'arm':arm,**z.to_dict()})
out=pd.DataFrame(rows); out.to_csv(OUT/'f33_confound_checks.csv',index=False)
rho_d,p_d=spearmanr(out['full'],out['mean_centroid_distance']); rho_v,p_v=spearmanr(out['full'],out['total_variance'])
(OUT/'f33_confound_summary.txt').write_text(f'Spearman full-vs-centroid-distance: rho={rho_d:.6f}, p={p_d:.6g}\nSpearman full-vs-total-variance: rho={rho_v:.6f}, p={p_v:.6g}\n',encoding='utf-8')
print(out.to_string(index=False)); print((OUT/'f33_confound_summary.txt').read_text())
