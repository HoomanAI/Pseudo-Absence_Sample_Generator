"""Generate 5500 Random + BP_V6 pseudo-absences (fast)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)
N_TARGET=5500

def _epoly(cx,cy,sx,sy,n=60):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return Polygon(zip(cx+sx*np.cos(t),cy+sy*np.sin(t)))
_WP=[_epoly(-115.36,55.43,0.69,0.115),_epoly(-115.49,55.78,0.25,0.090),
     _epoly(-113.19,55.27,0.11,0.095),_epoly(-113.27,54.73,0.10,0.075),
     _epoly(-114.70,55.33,0.07,0.055),_epoly(-113.10,55.45,0.07,0.055)]
_WB=[p.bounds for p in _WP]
def in_water_arr(pts):
    r=np.zeros(len(pts),dtype=bool)
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        inbox=(pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
        for idx in np.where(inbox)[0]:
            if poly.contains(Point(pts[idx,0],pts[idx,1])): r[idx]=True
    return r

df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts=df[['LONGITUDE','LATITUDE']].values
fcols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[fcols].values.astype(float)
fire_tree=cKDTree(fire_pts)
DEG=1/111.0; D_MIN=10*DEG; LAM=20*DEG
lon_min,lon_max=fire_pts[:,0].min()-0.3,fire_pts[:,0].max()+0.3
lat_min,lat_max=fire_pts[:,1].min()-0.3,fire_pts[:,1].max()+0.3

def idw_feats(pts,k=7):
    d,ix=fire_tree.query(pts,k=k); d=np.maximum(d,1e-9)
    w=1/d**2; w/=w.sum(1,keepdims=True)
    f=np.einsum('nk,nkf->nf',w,fire_feats[ix])
    f+=np.random.normal(0,0.05*fire_feats.std(0),f.shape); return f

# RANDOM
print("[Random]")
pool=[]
while len(pool)<N_TARGET:
    b=np.column_stack([np.random.uniform(lon_min,lon_max,N_TARGET*4),
                       np.random.uniform(lat_min,lat_max,N_TARGET*4)])
    pool.extend(b[~in_water_arr(b)].tolist())
rand_pts=np.array(pool[:N_TARGET])
rand_feats=idw_feats(rand_pts)
print(f"  {len(rand_pts)} pts")

# BP_V6 vectorised
print("[BP_V6]")
GRID_RES=0.5
lon_edges=np.arange(lon_min,lon_max+GRID_RES,GRID_RES)
lat_edges=np.arange(lat_min,lat_max+GRID_RES,GRID_RES)
n_lon,n_lat=len(lon_edges)-1,len(lat_edges)-1
fi=np.clip(np.digitize(fire_pts[:,0],lon_edges)-1,0,n_lon-1)
fj=np.clip(np.digitize(fire_pts[:,1],lat_edges)-1,0,n_lat-1)
fc=np.zeros((n_lon,n_lat),int)
for ii,jj in zip(fi,fj): fc[ii,jj]+=1
Fmax=fc.max(); alloc=np.ceil(fc/Fmax*N_TARGET).astype(int)
print(f"  Grid {n_lon}x{n_lat} alloc_sum={alloc.sum()}")

pool_sz=N_TARGET*25
cands=np.column_stack([np.random.uniform(lon_min,lon_max,pool_sz),
                        np.random.uniform(lat_min,lat_max,pool_sz)])
dists,_=fire_tree.query(cands,k=1)
ok=cands[dists>=D_MIN]; okd=dists[dists>=D_MIN]
land=~in_water_arr(ok); ok=ok[land]; okd=okd[land]
p_acc=1-np.exp(-(okd-D_MIN)/LAM)
keep=np.random.uniform(0,1,len(ok))<=p_acc; ok=ok[keep]; okd=okd[keep]
print(f"  Pool after filters: {len(ok)}")
gi=np.clip(np.digitize(ok[:,0],lon_edges)-1,0,n_lon-1)
gj=np.clip(np.digitize(ok[:,1],lat_edges)-1,0,n_lat-1)
heur_pool=[]
for i in range(n_lon):
    for j in range(n_lat):
        ni=alloc[i,j]
        if ni==0: continue
        mask=(gi==i)&(gj==j); cell_pts=ok[mask]
        if len(cell_pts)>0:
            n_take=min(ni,len(cell_pts))
            heur_pool.extend(cell_pts[np.random.choice(len(cell_pts),n_take,replace=False)].tolist())
# top-up
shortage=N_TARGET-len(heur_pool)
if shortage>0 and len(ok)>shortage:
    extra=ok[np.random.choice(len(ok),shortage+100,replace=False)][:shortage]
    heur_pool.extend(extra.tolist())
np.random.shuffle(heur_pool)
heur_pts=np.array(heur_pool[:N_TARGET])
heur_feats=idw_feats(heur_pts)
print(f"  {len(heur_pts)} pts")

OUT='../../outputs/'
np.save(OUT+'rand_pts.npy',rand_pts);  np.save(OUT+'rand_feats.npy',rand_feats)
np.save(OUT+'heur_pts.npy',heur_pts);  np.save(OUT+'heur_feats.npy',heur_feats)
pd.DataFrame(rand_pts,columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'random_pseudo_absences.csv',index=False)
pd.DataFrame(heur_pts,columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'heuristic_pseudo_absences.csv',index=False)
print("rand+heur saved.")
