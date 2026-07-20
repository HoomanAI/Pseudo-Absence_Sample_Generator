"""
Stage 1: PA generation using BP_V6 (Alberta) + SA + Random, then spatial evaluation.

BP_V6 (adapted from manuscript for Alberta):
  - 0.5-degree grid over fire bbox
  - Per-cell allocation: N_i = ceil(F_i / F_max * N_total)
  - Hard 10-km rejection buffer
  - Probabilistic acceptance: P_accept = 1 - exp(-(d - 10km) / 20km)
  - Water mask via shapely ellipse polygons (bbox-prefiltered for speed)
  - No CS selection needed (1:1 pool generated directly)

SA: minimise grid variance (spatial coverage), same fire buffer + water constraints.
Random: uniform bbox sampling with water filter.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPolygon, Polygon
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)

# ---- Water mask (bbox-prefiltered for speed) ----
def _epoly(clon, clat, slon, slat, n=60):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return Polygon(zip(clon + slon*np.cos(t), clat + slat*np.sin(t)))

_WATER_POLYS = [
    _epoly(-115.36, 55.43, 0.69, 0.115),  # Lesser Slave Lake
    _epoly(-115.49, 55.78, 0.25, 0.090),  # Utikuma Lake
    _epoly(-113.19, 55.27, 0.11, 0.095),  # Calling Lake
    _epoly(-113.27, 54.73, 0.10, 0.075),  # Baptiste Lake
    _epoly(-114.70, 55.33, 0.07, 0.055),  # Freeman Lake
    _epoly(-113.10, 55.45, 0.07, 0.055),  # Heart Lake
]
_WATER_BOUNDS = [p.bounds for p in _WATER_POLYS]  # (minx, miny, maxx, maxy)
WATER_MASK = MultiPolygon(_WATER_POLYS)

def in_water_arr(pts):
    """Vectorised water check: bbox prefilter then shapely contains."""
    result = np.zeros(len(pts), dtype=bool)
    for poly, (x0, y0, x1, y1) in zip(_WATER_POLYS, _WATER_BOUNDS):
        inbox = (pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
        for idx in np.where(inbox)[0]:
            if poly.contains(Point(pts[idx,0], pts[idx,1])):
                result[idx] = True
    return result

def in_water1(lon, lat):
    for poly, (x0, y0, x1, y1) in zip(_WATER_POLYS, _WATER_BOUNDS):
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            if poly.contains(Point(lon, lat)):
                return True
    return False

print("Water mask: 6 lakes (bbox-prefiltered)")

# ---- Load data ----
df        = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts  = df[['LONGITUDE','LATITUDE']].values
feat_cols = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats = df[feat_cols].values
fire_tree  = cKDTree(fire_pts)

N          = len(fire_pts)
DEG_PER_KM = 1/111.0
D_MIN      = 10 * DEG_PER_KM    # hard rejection radius (10 km)
LAM        = 20 * DEG_PER_KM    # probabilistic scale (20 km)

lon_min, lon_max = fire_pts[:,0].min()-0.3, fire_pts[:,0].max()+0.3
lat_min, lat_max = fire_pts[:,1].min()-0.3, fire_pts[:,1].max()+0.3

print(f"Loaded {N} fire pts | lon [{lon_min:.2f},{lon_max:.2f}] lat [{lat_min:.2f},{lat_max:.2f}]")

# ============================================================
# 1. RANDOM PA  (water-filtered)
# ============================================================
rand_pool = []
while len(rand_pool) < N:
    batch = np.column_stack([np.random.uniform(lon_min, lon_max, N*4),
                              np.random.uniform(lat_min, lat_max, N*4)])
    rand_pool.extend(batch[~in_water_arr(batch)].tolist())
rand_pts = np.array(rand_pool[:N])
print(f"\n[Random PA] {len(rand_pts)} pts (water-filtered)")

# ============================================================
# 2. HEURISTIC PA — BP_V6 (Alberta adaptation, manuscript Sec 3.4.6)
#
#   Grid: 0.5-degree cells over fire bbox
#   Per-cell allocation: N_i = ceil(F_i / F_max * N_total)
#   Acceptance: hard reject d < D_MIN; else P = 1 - exp(-(d-D_MIN)/LAM)
#   No CS selection step (pool size = N_total directly)
# ============================================================
GRID_RES = 0.5   # degrees (matches manuscript)

lon_edges = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
lat_edges  = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
n_lon = len(lon_edges) - 1
n_lat = len(lat_edges) - 1

# Fire count per cell
fi = np.clip(np.digitize(fire_pts[:,0], lon_edges) - 1, 0, n_lon-1)
fj = np.clip(np.digitize(fire_pts[:,1], lat_edges) - 1, 0, n_lat-1)
fire_counts = np.zeros((n_lon, n_lat), dtype=int)
for ii, jj in zip(fi, fj):
    fire_counts[ii, jj] += 1

F_max  = fire_counts.max()
# Per-cell allocation (Eq. 7 in manuscript)
alloc  = np.ceil(fire_counts / F_max * N).astype(int)

print(f"\n[BP_V6] grid {n_lon}x{n_lat} cells | F_max={F_max} | sum(alloc)={alloc.sum()}")

heur_pool = []
for i in range(n_lon):
    for j in range(n_lat):
        ni = alloc[i, j]
        if ni == 0:
            continue
        lo_lon, hi_lon = lon_edges[i], lon_edges[i+1]
        lo_lat, hi_lat = lat_edges[j], lat_edges[j+1]
        accepted = []
        attempts = 0
        while len(accepted) < ni and attempts < 100:
            bsz   = max(5000, ni * 20)
            batch = np.column_stack([np.random.uniform(lo_lon, hi_lon, bsz),
                                     np.random.uniform(lo_lat, hi_lat, bsz)])
            # Hard fire-buffer filter
            dists, _ = fire_tree.query(batch, k=1)
            ok  = batch[dists >= D_MIN]
            okd = dists[dists >= D_MIN]
            # Water mask
            land = ~in_water_arr(ok)
            ok   = ok[land]; okd = okd[land]
            # Probabilistic acceptance (Eq. 10-11)
            p_acc = 1 - np.exp(-(okd - D_MIN) / LAM)
            keep  = np.random.uniform(0, 1, len(ok)) <= p_acc
            ok    = ok[keep]
            accepted.extend(ok[:ni - len(accepted)].tolist())
            attempts += 1
        heur_pool.extend(accepted)

np.random.shuffle(heur_pool)
heur_pts = np.array(heur_pool[:N])
print(f"[BP_V6] pool={len(heur_pool)} | selected={len(heur_pts)}")

# ============================================================
# 3. SIMULATED ANNEALING PA (water-masked, same constraints as BP_V6)
# ============================================================
def generate_sa_pa(n_target, n_iter=55000, T_init=1.0, T_final=0.001,
                   n_cells_sa=12, step_fraction=0.25):
    print(f"\n[SA PA] n_iter={n_iter} T_init={T_init} T_final={T_final}")
    le = np.linspace(lon_min, lon_max, n_cells_sa+1)
    ae = np.linspace(lat_min, lat_max, n_cells_sa+1)
    def cell(pt):
        return (int(np.clip(np.digitize(pt[0],le)-1,0,n_cells_sa-1)),
                int(np.clip(np.digitize(pt[1],ae)-1,0,n_cells_sa-1)))
    # Init
    pts=[]; att=0
    while len(pts)<n_target:
        batch=np.column_stack([np.random.uniform(lon_min,lon_max,20000),
                                np.random.uniform(lat_min,lat_max,20000)])
        d,_=fire_tree.query(batch,k=1)
        valid=batch[d>=D_MIN]
        valid=valid[~in_water_arr(valid)]
        pts.extend(valid[:n_target-len(pts)].tolist())
        att+=1
        if att>40: break
    pts=np.array(pts[:n_target])
    counts=np.zeros(n_cells_sa*n_cells_sa)
    cell_ids=[]
    for p in pts:
        ci,cj=cell(p); cid=ci*n_cells_sa+cj; counts[cid]+=1; cell_ids.append(cid)
    cell_ids=np.array(cell_ids,dtype=int)
    E=float(np.var(counts))
    cooling=(T_final/T_init)**(1.0/n_iter); T=T_init; accepted=0
    lon_r=lon_max-lon_min; lat_r=lat_max-lat_min
    for i in range(n_iter):
        idx=np.random.randint(n_target); old_pt=pts[idx]; old_cid=cell_ids[idx]
        new_pt=np.array([old_pt[0]+np.random.normal(0,T*lon_r*step_fraction),
                         old_pt[1]+np.random.normal(0,T*lat_r*step_fraction)])
        if not(lon_min<=new_pt[0]<=lon_max and lat_min<=new_pt[1]<=lat_max):
            T*=cooling; continue
        d_new,_=fire_tree.query(new_pt.reshape(1,2),k=1)
        if d_new[0]<D_MIN: T*=cooling; continue
        if in_water1(new_pt[0],new_pt[1]): T*=cooling; continue
        new_ci,new_cj=cell(new_pt); new_cid=new_ci*n_cells_sa+new_cj
        if new_cid!=old_cid:
            counts[old_cid]-=1; counts[new_cid]+=1
            E_new=float(np.var(counts)); dE=E_new-E
            if dE<0 or np.random.random()<np.exp(-dE/(T+1e-30)):
                pts[idx]=new_pt; cell_ids[idx]=new_cid; E=E_new; accepted+=1
            else:
                counts[old_cid]+=1; counts[new_cid]-=1
        else:
            pts[idx]=new_pt; accepted+=1
        T*=cooling
        if (i+1)%10000==0:
            print(f"  iter {i+1:6d} T={T:.5f} E={E:8.2f} acc={accepted/(i+1)*100:.1f}%")
    print(f"  done: acc={100*accepted/n_iter:.1f}% E={E:.2f}")
    return np.array(pts)

sa_pts=generate_sa_pa(N)
wn=in_water_arr(sa_pts).sum()
print(f"[SA PA] {len(sa_pts)} pts | water check: {wn} (should be 0)")

# ============================================================
# 4. SPATIAL EVALUATION
# ============================================================
def k_sse(pts_test):
    r_vals=np.linspace(0.05,0.8,15)
    area=(fire_pts[:,0].max()-fire_pts[:,0].min())*(fire_pts[:,1].max()-fire_pts[:,1].min())
    def kc(pts):
        t=cKDTree(pts); lam=len(pts)/area
        return np.array([(np.array(t.query_ball_point(pts,r,return_length=True))-1).mean()/lam for r in r_vals])
    return float(np.sum((kc(pts_test)-kc(fire_pts))**2))

def centroid_d(p): return float(np.sqrt(((p.mean(0)-fire_pts.mean(0))**2).sum()))

def grid_var_metric(pts, n=10):
    le=np.linspace(pts[:,0].min(),pts[:,0].max(),n+1)
    ae=np.linspace(pts[:,1].min(),pts[:,1].max(),n+1)
    return float(np.var([((pts[:,0]>=le[i])&(pts[:,0]<le[i+1])&(pts[:,1]>=ae[j])&(pts[:,1]<ae[j+1])).sum()
                         for i in range(n) for j in range(n)]))

def nn_d(pts):
    t=cKDTree(pts); d,_=t.query(pts,k=2); return float(d[:,1].mean())

print("\nSpatial evaluation ...")
sse_h,sse_r,sse_s = k_sse(heur_pts),k_sse(rand_pts),k_sse(sa_pts)
cd_h,cd_r,cd_s   = centroid_d(heur_pts),centroid_d(rand_pts),centroid_d(sa_pts)
gv_h,gv_r,gv_s  = grid_var_metric(heur_pts),grid_var_metric(rand_pts),grid_var_metric(sa_pts)
nn_h,nn_r,nn_s   = nn_d(heur_pts),nn_d(rand_pts),nn_d(sa_pts)
nn_f             = nn_d(fire_pts)

sp_df=pd.DataFrame({
    'Metric':    ['K SSE','Centroid Dist','Grid Var','Mean NN'],
    'Heuristic': [round(sse_h,2),round(cd_h,4),round(gv_h,2),round(nn_h,4)],
    'Random':    [round(sse_r,2),round(cd_r,4),round(gv_r,2),round(nn_r,4)],
    'SA':        [round(sse_s,2),round(cd_s,4),round(gv_s,2),round(nn_s,4)],
    'Fire_Ref':  ['0.00','0.0000','--',round(nn_f,4)]
})
print(sp_df.to_string(index=False))

# ---- Save ----
np.save('../../outputs/heur_pts.npy',  heur_pts)
np.save('../../outputs/rand_pts.npy',  rand_pts)
np.save('../../outputs/sa_pts.npy',    sa_pts)
np.save('../../outputs/spatial_stats.npy',
        np.array([sse_h,sse_r,sse_s,cd_h,cd_r,cd_s,gv_h,gv_r,gv_s,nn_h,nn_r,nn_s,nn_f]))
sp_df.to_csv('../../outputs/spatial_results.csv',index=False)
pd.DataFrame(heur_pts,columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/heuristic_pseudo_absences.csv',index=False)
pd.DataFrame(rand_pts,columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/random_pseudo_absences.csv',index=False)
pd.DataFrame(sa_pts,columns=['LONGITUDE','LATITUDE']).to_csv(
    '../../outputs/sa_pseudo_absences.csv',index=False)
print("\nStage 1 (BP_V6 + SA + Random) complete.")
