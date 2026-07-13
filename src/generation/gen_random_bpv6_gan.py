"""
Generate 5500 pseudo-absences each for: Random, BP_V6 (vectorised), GAN (numpy WGAN).
Saves: rand_pts.npy, heur_pts.npy, gan_pts.npy  (+ *_feats.npy for each)
"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from shapely.geometry import Point, Polygon
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

N_TARGET = 5500

# ── Water mask ────────────────────────────────────────────────────────────────
def _epoly(cx, cy, sx, sy, n=60):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return Polygon(zip(cx + sx*np.cos(t), cy + sy*np.sin(t)))

_WP = [_epoly(-115.36,55.43,0.69,0.115), _epoly(-115.49,55.78,0.25,0.090),
       _epoly(-113.19,55.27,0.11,0.095), _epoly(-113.27,54.73,0.10,0.075),
       _epoly(-114.70,55.33,0.07,0.055), _epoly(-113.10,55.45,0.07,0.055)]
_WB = [p.bounds for p in _WP]

def in_water_arr(pts):
    r = np.zeros(len(pts), dtype=bool)
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        inbox = (pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
        for idx in np.where(inbox)[0]:
            if poly.contains(Point(pts[idx,0],pts[idx,1])): r[idx]=True
    return r

def in_water1(lon,lat):
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        if x0<=lon<=x1 and y0<=lat<=y1 and poly.contains(Point(lon,lat)): return True
    return False

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts  = df[['LONGITUDE','LATITUDE']].values
fcols     = [c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats= df[fcols].values.astype(float)
fire_tree = cKDTree(fire_pts)
N_FIRE    = len(fire_pts)
DEG       = 1/111.0; D_MIN = 10*DEG; LAM = 20*DEG
lon_min,lon_max = fire_pts[:,0].min()-0.3, fire_pts[:,0].max()+0.3
lat_min,lat_max = fire_pts[:,1].min()-0.3, fire_pts[:,1].max()+0.3
print(f"Fire pts: {N_FIRE} | bbox lon[{lon_min:.2f},{lon_max:.2f}] lat[{lat_min:.2f},{lat_max:.2f}]")

def idw_feats(pts, k=7):
    d,ix = fire_tree.query(pts,k=k); d = np.maximum(d,1e-9)
    w = 1/d**2; w /= w.sum(1,keepdims=True)
    f = np.einsum('nk,nkf->nf',w,fire_feats[ix])
    f += np.random.normal(0,0.05*fire_feats.std(0),f.shape); return f

# ── 1. RANDOM ─────────────────────────────────────────────────────────────────
print("\n[Random] generating 5500 ...")
pool=[]
while len(pool)<N_TARGET:
    b=np.column_stack([np.random.uniform(lon_min,lon_max,N_TARGET*4),
                       np.random.uniform(lat_min,lat_max,N_TARGET*4)])
    pool.extend(b[~in_water_arr(b)].tolist())
rand_pts = np.array(pool[:N_TARGET])
rand_feats= idw_feats(rand_pts)
print(f"  Done: {len(rand_pts)} pts")

# ── 2. BP_V6 vectorised ───────────────────────────────────────────────────────
print("\n[BP_V6] vectorised sampling ...")
GRID_RES=0.5
lon_edges=np.arange(lon_min,lon_max+GRID_RES,GRID_RES)
lat_edges =np.arange(lat_min,lat_max+GRID_RES,GRID_RES)
n_lon,n_lat=len(lon_edges)-1,len(lat_edges)-1
fi=np.clip(np.digitize(fire_pts[:,0],lon_edges)-1,0,n_lon-1)
fj=np.clip(np.digitize(fire_pts[:,1],lat_edges)-1,0,n_lat-1)
fire_counts=np.zeros((n_lon,n_lat),int)
for ii,jj in zip(fi,fj): fire_counts[ii,jj]+=1
F_max=fire_counts.max()
alloc=np.ceil(fire_counts/F_max*N_TARGET).astype(int)
print(f"  Grid {n_lon}x{n_lat} | F_max={F_max} | sum(alloc)={alloc.sum()}")

# Global candidate pool
pool_sz = N_TARGET*45
cands=np.column_stack([np.random.uniform(lon_min,lon_max,pool_sz),
                        np.random.uniform(lat_min,lat_max,pool_sz)])
dists,_=fire_tree.query(cands,k=1)
ok=cands[dists>=D_MIN]; okd=dists[dists>=D_MIN]
ok=ok[~in_water_arr(ok)]; okd=okd[~in_water_arr(cands[dists>=D_MIN])]
# re-align after water filter
ok2=cands[dists>=D_MIN]; okd2=dists[dists>=D_MIN]
land=~in_water_arr(ok2); ok2=ok2[land]; okd2=okd2[land]
p_acc=1-np.exp(-(okd2-D_MIN)/LAM)
keep=np.random.uniform(0,1,len(ok2))<=p_acc
ok2=ok2[keep]; okd2=okd2[keep]
print(f"  Global pool after filters: {len(ok2)}")

gi=np.clip(np.digitize(ok2[:,0],lon_edges)-1,0,n_lon-1)
gj=np.clip(np.digitize(ok2[:,1],lat_edges)-1,0,n_lat-1)
heur_pool=[]
for i in range(n_lon):
    for j in range(n_lat):
        ni=alloc[i,j]
        if ni==0: continue
        mask=(gi==i)&(gj==j)
        cell_pts=ok2[mask]
        if len(cell_pts)>0:
            n_take=min(ni,len(cell_pts))
            idx=np.random.choice(len(cell_pts),n_take,replace=False)
            heur_pool.extend(cell_pts[idx].tolist())
# Top-up if short
shortage=N_TARGET-len(heur_pool)
if shortage>0:
    extras=ok2[np.random.choice(len(ok2),shortage*2,replace=False)]
    heur_pool.extend(extras[:shortage].tolist())
np.random.shuffle(heur_pool)
heur_pts=np.array(heur_pool[:N_TARGET])
heur_feats=idw_feats(heur_pts)
print(f"  Selected: {len(heur_pts)} pts")

# ── 3. NUMPY WGAN ─────────────────────────────────────────────────────────────
print("\n[GAN] Building background pool for training ...")
bg_pool=[]
for _ in range(8):
    b=np.column_stack([np.random.uniform(lon_min,lon_max,30000),
                       np.random.uniform(lat_min,lat_max,30000)])
    d,_=fire_tree.query(b,k=1)
    valid=b[d>=D_MIN]; valid=valid[~in_water_arr(valid)]
    bg_pool.extend(valid.tolist())
    if len(bg_pool)>=20000: break
bg_pts=np.array(bg_pool[:20000])
bg_feats_raw=idw_feats(bg_pts)

# PCA (58 → 10 components) for conditioning
scaler_f=StandardScaler().fit(bg_feats_raw)
bg_feats_sc=scaler_f.transform(bg_feats_raw)
pca=PCA(n_components=10,random_state=42).fit(bg_feats_sc)
bg_cond=pca.transform(bg_feats_sc)
scaler_c=StandardScaler().fit(bg_cond)
bg_cond=scaler_c.transform(bg_cond)

# Normalise coords → [-1,1]
def norm_coords(pts):
    lon_n = 2*(pts[:,0]-lon_min)/(lon_max-lon_min)-1
    lat_n = 2*(pts[:,1]-lat_min)/(lat_max-lat_min)-1
    return np.column_stack([lon_n,lat_n])
def denorm_coords(nc):
    lon=(nc[:,0]+1)/2*(lon_max-lon_min)+lon_min
    lat=(nc[:,1]+1)/2*(lat_max-lat_min)+lat_min
    return np.column_stack([lon,lat])

bg_coords_n = norm_coords(bg_pts)

Z_DIM=32; COND_DIM=10; H=64; LR=5e-5; CLIP_C=0.01; N_CRIT=5
rng=np.random.RandomState(7)

def init_mlp(sizes):
    Ws=[rng.randn(sizes[i],sizes[i+1])*0.02 for i in range(len(sizes)-1)]
    bs=[np.zeros(sizes[i+1]) for i in range(len(sizes)-1)]
    ms_W=[np.zeros_like(w) for w in Ws]; ms_b=[np.zeros_like(b) for b in bs]
    return Ws,bs,ms_W,ms_b

def fwd(x, Ws, bs, out='linear'):
    pre=[]; post=[x]
    for i,(W,b) in enumerate(zip(Ws,bs)):
        z=post[-1]@W+b; pre.append(z)
        if i<len(Ws)-1: post.append(np.where(z>0,z,0.2*z))
        else:
            if out=='tanh': post.append(np.tanh(z))
            else: post.append(z)
    return pre,post

def bwd(pre,post,dout,Ws,out='linear'):
    dWs=[]; dbs=[]; d=dout; N=dout.shape[0]
    for i in range(len(Ws)-1,-1,-1):
        z=pre[i]; a=post[i+1]
        if i==len(Ws)-1:
            if out=='tanh': d=d*(1-a**2)
        else: d=d*np.where(z>0,1.0,0.2)
        dWs.insert(0,post[i].T@d/N); dbs.insert(0,d.mean(0))
        d=d@Ws[i].T
    return dWs,dbs,d

def rmsprop_step(Ws,bs,dWs,dbs,ms_W,ms_b,lr=LR,rho=0.99):
    for i in range(len(Ws)):
        ms_W[i]=rho*ms_W[i]+(1-rho)*dWs[i]**2
        ms_b[i]=rho*ms_b[i]+(1-rho)*dbs[i]**2
        Ws[i]-=lr*dWs[i]/(np.sqrt(ms_W[i])+1e-8)
        bs[i]-=lr*dbs[i]/(np.sqrt(ms_b[i])+1e-8)

GW,Gb,Gm_W,Gm_b=init_mlp([Z_DIM+COND_DIM,H,H//2,2])
DW,Db,Dm_W,Dm_b=init_mlp([2+COND_DIM,H,H//2,1])

EPOCHS=400; BATCH=256; M=len(bg_pts)
print(f"  Training WGAN: {EPOCHS} epochs, batch={BATCH}, bg_pool={M}")
for ep in range(EPOCHS):
    idx=np.random.permutation(M)
    for start in range(0,M-BATCH,BATCH):
        bi=idx[start:start+BATCH]
        rc=bg_coords_n[bi]; rf=bg_cond[bi]
        for _ in range(N_CRIT):
            # Critic real
            pre_r,post_r=fwd(np.hstack([rc,rf]),DW,Db)
            dWr,dbr,_=bwd(pre_r,post_r,-np.ones((BATCH,1))/BATCH,DW)
            # Critic fake
            z=np.random.randn(BATCH,Z_DIM)
            cf_samp=bg_cond[np.random.randint(M,size=BATCH)]
            pre_g,post_g=fwd(np.hstack([z,cf_samp]),GW,Gb,'tanh')
            fc=post_g[-1]
            pre_f,post_f=fwd(np.hstack([fc,cf_samp]),DW,Db)
            dWf,dbf,_=bwd(pre_f,post_f,np.ones((BATCH,1))/BATCH,DW)
            rmsprop_step(DW,Db,[a+b for a,b in zip(dWr,dWf)],[a+b for a,b in zip(dbr,dbf)],Dm_W,Dm_b)
            for i in range(len(DW)): DW[i]=np.clip(DW[i],-CLIP_C,CLIP_C); Db[i]=np.clip(Db[i],-CLIP_C,CLIP_C)
        # Generator
        z=np.random.randn(BATCH,Z_DIM)
        cf_samp=bg_cond[np.random.randint(M,size=BATCH)]
        pre_g,post_g=fwd(np.hstack([z,cf_samp]),GW,Gb,'tanh')
        fc=post_g[-1]
        pre_d,post_d=fwd(np.hstack([fc,cf_samp]),DW,Db)
        dWd,dbd,d_in=bwd(pre_d,post_d,-np.ones((BATCH,1))/BATCH,DW)
        dfc=d_in[:,:2]
        dWg,dbg,_=bwd(pre_g,post_g,dfc,GW,'tanh')
        rmsprop_step(GW,Gb,dWg,dbg,Gm_W,Gm_b)
    if (ep+1)%100==0:
        _,pf=fwd(np.hstack([np.random.randn(1000,Z_DIM),bg_cond[np.random.randint(M,size=1000)]]),GW,Gb,'tanh')
        print(f"  ep {ep+1}/{EPOCHS} | G sample range lon[{pf[-1][:,0].min():.2f},{pf[-1][:,0].max():.2f}]")

# Generate GAN pseudo-absences
print("  Generating GAN points ...")
GAN_GEN=120000
z_all=np.random.randn(GAN_GEN,Z_DIM)
cf_all=bg_cond[np.random.randint(M,size=GAN_GEN)]
_,pg=fwd(np.hstack([z_all,cf_all]),GW,Gb,'tanh')
gan_raw=denorm_coords(pg[-1])
# Clip to bbox
gan_raw=gan_raw[(gan_raw[:,0]>=lon_min)&(gan_raw[:,0]<=lon_max)&
                (gan_raw[:,1]>=lat_min)&(gan_raw[:,1]<=lat_max)]
d_gan,_=fire_tree.query(gan_raw,k=1)
gan_raw=gan_raw[d_gan>=D_MIN]
gan_raw=gan_raw[~in_water_arr(gan_raw)]
print(f"  After filters: {len(gan_raw)} candidates")
if len(gan_raw)<N_TARGET:
    # top-up with background pool points
    extra=bg_pts[np.random.choice(len(bg_pts),N_TARGET-len(gan_raw)+200,replace=True)]
    gan_raw=np.vstack([gan_raw,extra])
np.random.shuffle(gan_raw)
gan_pts=gan_raw[:N_TARGET]
gan_feats=idw_feats(gan_pts)
print(f"  GAN pts: {len(gan_pts)}")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT='../../outputs/'
np.save(OUT+'rand_pts.npy',  rand_pts);  np.save(OUT+'rand_feats.npy',  rand_feats)
np.save(OUT+'heur_pts.npy',  heur_pts);  np.save(OUT+'heur_feats.npy',  heur_feats)
np.save(OUT+'gan_pts.npy',   gan_pts);   np.save(OUT+'gan_feats.npy',   gan_feats)
pd.DataFrame(heur_pts,columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'heuristic_pseudo_absences.csv',index=False)
pd.DataFrame(rand_pts,columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'random_pseudo_absences.csv',index=False)
pd.DataFrame(gan_pts, columns=['LONGITUDE','LATITUDE']).to_csv(OUT+'gan_pseudo_absences.csv',index=False)
print("\nScript 1 done. rand/heur/gan saved.")
