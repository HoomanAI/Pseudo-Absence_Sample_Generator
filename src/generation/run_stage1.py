"""Stage 1: PA generation + spatial eval (fast)"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv')
df = df.dropna()
fire_pts = df[['LONGITUDE','LATITUDE']].values
feature_cols = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats = df[feature_cols].values

N = len(fire_pts)
DEG_PER_KM = 1/111.0
D_MIN = 10*DEG_PER_KM
LAM   = 20*DEG_PER_KM
fire_tree = cKDTree(fire_pts)

print(f"Loaded {N} fire points, {len(feature_cols)} features")

# Random PA
rand_pts = np.column_stack([
    np.random.uniform(fire_pts[:,0].min(), fire_pts[:,0].max(), N),
    np.random.uniform(fire_pts[:,1].min(), fire_pts[:,1].max(), N)
])

# BP_V6
n_cells = 15
lon_min,lon_max = fire_pts[:,0].min()-0.3, fire_pts[:,0].max()+0.3
lat_min,lat_max = fire_pts[:,1].min()-0.3, fire_pts[:,1].max()+0.3
lon_edges = np.linspace(lon_min,lon_max,n_cells+1)
lat_edges  = np.linspace(lat_min,lat_max,n_cells+1)
lon_idx = np.clip(np.digitize(fire_pts[:,0],lon_edges)-1,0,n_cells-1)
lat_idx = np.clip(np.digitize(fire_pts[:,1],lat_edges)-1,0,n_cells-1)
density = np.zeros((n_cells,n_cells))
for i,j in zip(lon_idx,lat_idx): density[i,j]+=1
F_max = max(density.max(),1)

cands = np.column_stack([np.random.uniform(lon_min,lon_max,120000),
                          np.random.uniform(lat_min,lat_max,120000)])
dists,_ = fire_tree.query(cands,k=1)
cands = cands[dists>=D_MIN]; dists=dists[dists>=D_MIN]
p = 1-np.exp(-(dists-D_MIN)/LAM)
acc = np.random.uniform(0,1,len(cands))<p
cands,dists = cands[acc],dists[acc]
ci = np.clip(np.digitize(cands[:,0],lon_edges)-1,0,n_cells-1)
cj = np.clip(np.digitize(cands[:,1],lat_edges)-1,0,n_cells-1)
w  = 0.3+0.7*density[ci,cj]/F_max; w/=w.sum()
n_pool = 12000 if len(cands)>12000 else len(cands)
ch = np.random.choice(len(cands),n_pool,replace=False,p=w)
pool = cands[ch]
print(f"BP_V6 pool: {len(pool)}")

# CS_V5
d_fire,_   = fire_tree.query(pool,k=1)
centroid    = fire_pts.mean(0)
d_cent      = np.sqrt(((pool-centroid)**2).sum(1))
scores      = d_fire/(1+0.5*d_cent/(d_cent.mean()+1e-9))
lat33,lat66 = np.percentile(fire_pts[:,1],[33,66])
regions     = np.where(pool[:,1]>lat66,2,np.where(pool[:,1]>lat33,1,0))
n_per = N//3; selected=[]
for r in [0,1,2]:
    idx_r = np.where(regions==r)[0]
    if len(idx_r): selected.extend(idx_r[np.argsort(scores[idx_r])[-n_per:]].tolist())
sel = np.array(selected)
if len(sel)<N:
    extras=[i for i in np.argsort(scores)[::-1] if i not in set(sel.tolist())][:N-len(sel)]
    sel=np.concatenate([sel,extras])
heur_pts = pool[sel[:N]]
print(f"CS_V5: {len(heur_pts)}")

# Spatial metrics
def k_sse(pts_test,r_vals=np.linspace(0.05,0.8,15)):
    area=(fire_pts[:,0].max()-fire_pts[:,0].min())*(fire_pts[:,1].max()-fire_pts[:,1].min())
    def kc(pts):
        t=cKDTree(pts); lam=len(pts)/area
        return np.array([(np.array(t.query_ball_point(pts,r,return_length=True))-1).mean()/lam for r in r_vals])
    return float(np.sum((kc(pts_test)-kc(fire_pts))**2))

def centroid_d(p): return float(np.sqrt(((p.mean(0)-fire_pts.mean(0))**2).sum()))
def grid_var(pts,n=10):
    le=np.linspace(pts[:,0].min(),pts[:,0].max(),n+1)
    ae=np.linspace(pts[:,1].min(),pts[:,1].max(),n+1)
    return float(np.var([(((pts[:,0]>=le[i])&(pts[:,0]<le[i+1])&(pts[:,1]>=ae[j])&(pts[:,1]<ae[j+1])).sum()) for i in range(n) for j in range(n)]))
def nn_d(pts): t=cKDTree(pts); d,_=t.query(pts,k=2); return float(d[:,1].mean())

print("\nSpatial evaluation...")
sse_h,sse_r = k_sse(heur_pts), k_sse(rand_pts)
cd_h, cd_r  = centroid_d(heur_pts), centroid_d(rand_pts)
gv_h, gv_r  = grid_var(heur_pts), grid_var(rand_pts)
nn_h, nn_r  = nn_d(heur_pts), nn_d(rand_pts)
nn_f        = nn_d(fire_pts)

sp_df = pd.DataFrame({
    'Metric': ["Ripley's K SSE","Centroid Distance","Grid Variance","Mean NN Distance"],
    'Heuristic': [round(sse_h,2),round(cd_h,4),round(gv_h,2),round(nn_h,4)],
    'Random':    [round(sse_r,2),round(cd_r,4),round(gv_r,2),round(nn_r,4)],
    'Fire_Ref':  ['0.00','0.0000','—',round(nn_f,4)]
})
print(sp_df.to_string(index=False))

# Save
np.save('../../outputs/heur_pts.npy', heur_pts)
np.save('../../outputs/rand_pts.npy', rand_pts)
np.save('../../outputs/spatial_stats.npy',
        np.array([sse_h,sse_r,cd_h,cd_r,gv_h,gv_r,nn_h,nn_r,nn_f]))
sp_df.to_csv('../../outputs/spatial_results.csv',index=False)
pd.DataFrame(heur_pts,columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/heuristic_pseudo_absences.csv',index=False)
pd.DataFrame(rand_pts,columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/random_pseudo_absences.csv',index=False)
print("\nStage 1 done.")
