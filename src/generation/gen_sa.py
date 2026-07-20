"""Generate 5500 SA pseudo-absences."""
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
def in_water1(lon,lat):
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        if x0<=lon<=x1 and y0<=lat<=y1 and poly.contains(Point(lon,lat)): return True
    return False

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

def generate_sa(n_target,n_iter=60000,T_init=1.0,T_final=0.001,n_cells=12,step_f=0.25):
    print(f"[SA] n={n_target} iters={n_iter}")
    le=np.linspace(lon_min,lon_max,n_cells+1)
    ae=np.linspace(lat_min,lat_max,n_cells+1)
    def cell(pt):
        return (int(np.clip(np.digitize(pt[0],le)-1,0,n_cells-1)),
                int(np.clip(np.digitize(pt[1],ae)-1,0,n_cells-1)))
    pts=[]; att=0
    while len(pts)<n_target:
        b=np.column_stack([np.random.uniform(lon_min,lon_max,25000),
                           np.random.uniform(lat_min,lat_max,25000)])
        d,_=fire_tree.query(b,k=1); v=b[d>=D_MIN]; v=v[~in_water_arr(v)]
        pts.extend(v[:n_target-len(pts)].tolist()); att+=1
        if att>50: break
    pts=np.array(pts[:n_target])
    counts=np.zeros(n_cells*n_cells); cell_ids=[]
    for p in pts:
        ci,cj=cell(p); cid=ci*n_cells+cj; counts[cid]+=1; cell_ids.append(cid)
    cell_ids=np.array(cell_ids,dtype=int)
    E=float(np.var(counts))
    cooling=(T_final/T_init)**(1.0/n_iter); T=T_init; acc=0
    lr=lon_max-lon_min; ar=lat_max-lat_min
    for i in range(n_iter):
        idx=np.random.randint(n_target); op=pts[idx]; oc=cell_ids[idx]
        np2=np.array([op[0]+np.random.normal(0,T*lr*step_f),
                      op[1]+np.random.normal(0,T*ar*step_f)])
        if not(lon_min<=np2[0]<=lon_max and lat_min<=np2[1]<=lat_max):
            T*=cooling; continue
        d2,_=fire_tree.query(np2.reshape(1,2),k=1)
        if d2[0]<D_MIN: T*=cooling; continue
        if in_water1(np2[0],np2[1]): T*=cooling; continue
        nci,ncj=cell(np2); ncid=nci*n_cells+ncj
        if ncid!=oc:
            counts[oc]-=1; counts[ncid]+=1
            En=float(np.var(counts)); dE=En-E
            if dE<0 or np.random.random()<np.exp(-dE/(T+1e-30)):
                pts[idx]=np2; cell_ids[idx]=ncid; E=En; acc+=1
            else: counts[oc]+=1; counts[ncid]-=1
        else: pts[idx]=np2; acc+=1
        T*=cooling
        if (i+1)%15000==0:
            print(f"  iter {i+1} T={T:.5f} E={E:.2f} acc={acc/(i+1)*100:.1f}%")
    print(f"  done E={E:.2f} acc={100*acc/n_iter:.1f}%")
    return np.array(pts)

sa_pts=generate_sa(N_TARGET)
wn=in_water_arr(sa_pts).sum()
print(f"[SA] {len(sa_pts)} pts | water={wn}")
sa_feats=idw_feats(sa_pts)

OUT='../../outputs/'
np.save(OUT+'sa_pts.npy',sa_pts); np.save(OUT+'sa_feats.npy',sa_feats)
pd.DataFrame(sa_pts,columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'sa_pseudo_absences.csv',index=False)
print("SA saved.")
