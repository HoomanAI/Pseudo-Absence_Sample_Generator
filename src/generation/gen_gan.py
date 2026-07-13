"""
Feature-Space WGAN for pseudo-absence generation.
G(z[32]) → 58-dim env-feature vector trained on non-fire (background) IDW features.
Spatial mapping: generated feature vectors → kNN in PCA-20 space → background location.
Eliminates border clustering: no coordinate generation, only feature generation.
"""
import numpy as np, pandas as pd, time, os
import torch, torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point, Polygon
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42); torch.manual_seed(42); t0=time.time()
N_TARGET=5500

# -- Water polygons ------------------------------------------------------------
def _ep(cx,cy,sx,sy,n=60):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return Polygon(zip(cx+sx*np.cos(t),cy+sy*np.sin(t)))
_WP=[_ep(-115.36,55.43,0.69,0.115),_ep(-115.49,55.78,0.25,0.090),
     _ep(-113.19,55.27,0.11,0.095),_ep(-113.27,54.73,0.10,0.075),
     _ep(-114.70,55.33,0.07,0.055),_ep(-113.10,55.45,0.07,0.055)]
_WB=[p.bounds for p in _WP]
def in_water(pts):
    r=np.zeros(len(pts),dtype=bool)
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        m=(pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
        for i in np.where(m)[0]:
            if poly.contains(Point(pts[i,0],pts[i,1])): r[i]=True
    return r

# -- Load data -----------------------------------------------------------------
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts=df[['LONGITUDE','LATITUDE']].values
fcols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[fcols].values.astype(float)
fire_tree=cKDTree(fire_pts)
DEG=1/111.0; D_MIN=10*DEG
lon_min,lon_max=fire_pts[:,0].min()-0.2, fire_pts[:,0].max()+0.2
lat_min,lat_max=fire_pts[:,1].min()-0.2, fire_pts[:,1].max()+0.2
print(f"Study area: lon[{lon_min:.2f},{lon_max:.2f}] lat[{lat_min:.2f},{lat_max:.2f}]")

def idw_feats(pts, k=7, noise_seed=99):
    d,ix=fire_tree.query(pts,k=k); d=np.maximum(d,1e-9)
    w=1/d**2; w/=w.sum(1,keepdims=True)
    f=np.einsum('nk,nkf->nf',w,fire_feats[ix])
    f+=np.random.RandomState(noise_seed).normal(0,0.05*fire_feats.std(0),f.shape)
    return f

# -- Background pool -----------------------------------------------------------
print(f"Building background pool... t={time.time()-t0:.1f}s")
rng0=np.random.RandomState(0)
lons=np.linspace(lon_min+0.04,lon_max-0.04,150)
lats=np.linspace(lat_min+0.04,lat_max-0.04,110)
gx,gy=np.meshgrid(lons,lats)
grid_pts=np.column_stack([gx.ravel(),gy.ravel()])
grid_pts+=rng0.uniform(-0.03,0.03,grid_pts.shape)
rand_pts=np.column_stack([rng0.uniform(lon_min,lon_max,15000),
                           rng0.uniform(lat_min,lat_max,15000)])
all_cands=np.vstack([grid_pts, rand_pts])
dd,_=fire_tree.query(all_cands,k=1)
cands=all_cands[dd>=D_MIN]; cands=cands[~in_water(cands)]
bg_pts=cands[:15000]; N_BG=len(bg_pts)
print(f"  bg pool: {N_BG} pts, t={time.time()-t0:.1f}s")

# -- 58-dim IDW features -------------------------------------------------------
bg_feats=idw_feats(bg_pts); n_feat=bg_feats.shape[1]
sc=StandardScaler().fit(bg_feats); bg_sc=sc.transform(bg_feats)
pca20=PCA(n_components=20,random_state=42).fit(bg_sc)
bg_pca=pca20.transform(bg_sc)
bg_pca_sq=(bg_pca**2).sum(1)
print(f"  n_feat={n_feat}, PCA(20) var={pca20.explained_variance_ratio_.sum():.3f}, t={time.time()-t0:.1f}s")

# -- Domain-aware validity structure (for gradient-penalty weighting) ----------
GP_VALIDITY_PERCENTILE=90
knn_real=NearestNeighbors(n_neighbors=7).fit(bg_sc)
d_self,_=knn_real.kneighbors(bg_sc)
d_self_k=d_self[:,-1]
tau=np.percentile(d_self_k,GP_VALIDITY_PERCENTILE)
print(f"  GP validity radius tau={tau:.4f} (p{GP_VALIDITY_PERCENTILE} of 7-NN dist), t={time.time()-t0:.1f}s")

# -- WGAN in 58-dim feature space ----------------------------------------------
Z=32; H=32; CLIP=0.05; N_C=3; LR=5e-5; BATCH=256; N_ITER=4500
CRITIC_CONSTRAINT=os.environ.get('CRITIC_CONSTRAINT','gp_domain_aware')  # 'clip' | 'gp_standard' | 'gp_domain_aware'
LAMBDA_GP=10.0; GP_OOB_WEIGHT=0.1; GP_MODE='soft'  # 'soft' down-weights OOB interpolants by GP_OOB_WEIGHT, 'hard' zeroes them
ADAM_LR=1e-4; ADAM_BETAS=(0.0,0.9)
rng=np.random.RandomState(42)

def lrelu(x,a=0.2): return np.where(x>0,x,a*x)

def fwd(x,W,b):
    # kept for downstream (line ~146: Ggen,_=fwd(z_gen,GW,Gb)) — GW/Gb are exported
    # from the trained torch Generator below, in this same (in,out) list format.
    acts,zs=[x],[]
    for i,(w,bi) in enumerate(zip(W,b)):
        z=acts[-1]@w+bi; zs.append(z)
        acts.append(z if i==len(W)-1 else lrelu(z))
    return acts,zs

class _MLP(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.linears=nn.ModuleList([nn.Linear(sizes[i],sizes[i+1]) for i in range(len(sizes)-1)])
        for i,lin in enumerate(self.linears):
            nn.init.normal_(lin.weight,0.0,np.sqrt(2.0/sizes[i]))
            nn.init.zeros_(lin.bias)
    def forward(self,x):
        for i,lin in enumerate(self.linears):
            x=lin(x)
            if i<len(self.linears)-1: x=torch.nn.functional.leaky_relu(x,0.2)
        return x

G=_MLP([Z,H,H,n_feat]); D=_MLP([n_feat,H,H,1])
if CRITIC_CONSTRAINT=='clip':
    optD=torch.optim.RMSprop(D.parameters(),lr=LR,alpha=0.99)
    optG=torch.optim.RMSprop(G.parameters(),lr=LR,alpha=0.99)
else:
    optD=torch.optim.Adam(D.parameters(),lr=ADAM_LR,betas=ADAM_BETAS)
    optG=torch.optim.Adam(G.parameters(),lr=ADAM_LR,betas=ADAM_BETAS)

bg_sc_t=torch.tensor(bg_sc,dtype=torch.float32)
gp_log=[]
print(f"Training WGAN ({N_ITER} iters, H={H}, constraint={CRITIC_CONSTRAINT})... t={time.time()-t0:.1f}s")
for it in range(N_ITER):
    it_gn=[]; it_gp=[]; it_im=[]; it_cl=[]
    for _ in range(N_C):
        ri=rng.randint(0,N_BG,BATCH); real=bg_sc_t[ri]
        z=torch.tensor(rng.randn(BATCH,Z),dtype=torch.float32)
        with torch.no_grad(): fake=G(z)
        d_real=D(real); d_fake=D(fake)
        critic_core=-d_real.mean()+d_fake.mean()

        eps=torch.rand(BATCH,1)
        x_hat=(eps*real+(1-eps)*fake).requires_grad_(True)
        d_hat=D(x_hat)
        grads=torch.autograd.grad(d_hat,x_hat,grad_outputs=torch.ones_like(d_hat),
                                   create_graph=True,retain_graph=True)[0]
        grad_norm=grads.norm(2,dim=1)
        gp_per_sample=(grad_norm-1.0)**2

        knn_dist=knn_real.kneighbors(x_hat.detach().numpy(),n_neighbors=7)[0][:,-1]
        in_manifold=torch.tensor((knn_dist<=tau).astype(np.float32))

        if CRITIC_CONSTRAINT=='gp_domain_aware':
            weight=in_manifold if GP_MODE=='hard' else in_manifold+(1-in_manifold)*GP_OOB_WEIGHT
        else:
            weight=torch.ones_like(in_manifold)
        gradient_penalty=(weight*gp_per_sample).sum()/weight.sum().clamp(min=1e-8)

        critic_loss=critic_core if CRITIC_CONSTRAINT=='clip' else critic_core+LAMBDA_GP*gradient_penalty

        optD.zero_grad(); critic_loss.backward(); optD.step()
        if CRITIC_CONSTRAINT=='clip':
            with torch.no_grad():
                for lin in D.linears: lin.weight.clamp_(-CLIP,CLIP)

        it_gn.append(grad_norm.detach()); it_gp.append(gp_per_sample.detach())
        it_im.append(in_manifold.detach()); it_cl.append(float(critic_loss.detach()))

    z=torch.tensor(rng.randn(BATCH,Z),dtype=torch.float32)
    fake=G(z); gen_loss=-D(fake).mean()
    optG.zero_grad(); gen_loss.backward(); optG.step()

    gn_cat=torch.cat(it_gn); gp_cat=torch.cat(it_gp); im_cat=torch.cat(it_im)
    if not (np.isfinite(it_cl).all() and np.isfinite(float(gen_loss))):
        raise FloatingPointError(f"Non-finite loss at iter {it+1} (constraint={CRITIC_CONSTRAINT})")
    gp_log.append(dict(iter=it+1,
        grad_norm_mean=float(gn_cat.mean()), grad_norm_std=float(gn_cat.std()),
        gp_mean=float(gp_cat.mean()), gp_std=float(gp_cat.std()),
        in_manifold_frac=float(im_cat.mean()),
        critic_loss=float(np.mean(it_cl)), gen_loss=float(gen_loss.detach())))

    if (it+1)%1500==0:
        with torch.no_grad():
            z_t=torch.tensor(rng.randn(1000,Z),dtype=torch.float32); Gt=G(z_t)
            real_t=bg_sc_t[rng.randint(0,N_BG,1000)]
            wd=D(real_t).mean().item()-D(Gt).mean().item()
            bias=np.abs(Gt.numpy().mean(0)-bg_sc.mean(0)).mean()
        print(f"  iter {it+1:5d} | W-dist={wd:.4f} feat_bias={bias:.4f} constraint={CRITIC_CONSTRAINT} t={time.time()-t0:.1f}s")

pd.DataFrame(gp_log).to_csv(f'../../outputs/gan_diagnostics_{CRITIC_CONSTRAINT}.csv',index=False)

# Export trained torch Generator weights into the GW,Gb numpy list format that
# fwd()/downstream code (line ~146 onward) already expects, so nothing past
# this point needs to change.
GW=[lin.weight.detach().numpy().T.copy() for lin in G.linears]
Gb=[lin.bias.detach().numpy().copy() for lin in G.linears]

# -- Generate feature vectors → map to bg locations ----------------------------
N_GEN=25000
print(f"Generating {N_GEN} feature vectors... t={time.time()-t0:.1f}s")
z_gen=rng.randn(N_GEN,Z); Ggen,_=fwd(z_gen,GW,Gb); gen_sc=Ggen[-1]
gen_pca=pca20.transform(gen_sc)

print(f"kNN mapping (k=5)... t={time.time()-t0:.1f}s")
freq=np.zeros(N_BG,dtype=np.float32)
K_NN=5; BSZ=1000
for i in range(0,N_GEN,BSZ):
    batch=gen_pca[i:i+BSZ]
    bsq=((batch**2).sum(1))[:,None]
    cross=batch@bg_pca.T
    d2=bsq+bg_pca_sq[None,:]-2*cross          # (BSZ, N_BG)
    # soft-kNN: top-K nearest, weighted by 1/sqrt(d2+eps) — fully vectorised
    tk=np.argpartition(d2,K_NN,axis=1)[:,:K_NN]  # (BSZ,K)
    dk=np.take_along_axis(d2,tk,axis=1)           # (BSZ,K) distances
    wk=1.0/(np.sqrt(np.maximum(dk,1e-6)))         # (BSZ,K) soft weights
    np.add.at(freq, tk.ravel(), wk.ravel())        # vectorised scatter
n_hit=(freq>0).sum()
print(f"  Unique bg locs hit: {n_hit}/{N_BG}, t={time.time()-t0:.1f}s")

# Border distance penalty: down-weight points near bbox edges
lon_r=lon_max-lon_min; lat_r=lat_max-lat_min
bdist=np.minimum(
    np.minimum(bg_pts[:,0]-lon_min, lon_max-bg_pts[:,0])/lon_r,
    np.minimum(bg_pts[:,1]-lat_min, lat_max-bg_pts[:,1])/lat_r)
border_w=1.0-np.exp(-bdist/0.08)  # 0 at edge, →1 in interior (half-weight at ~0.08 of range)

# Combined weight: GAN frequency × border penalty
combined=freq*border_w
# Cap extremes and normalise
cap=np.percentile(combined[combined>0],95)
combined=np.minimum(combined,cap)
combined=np.maximum(combined,0)
# Select 5500 from ALL bg points (combined>0)
valid=np.where(combined>0)[0]
probs=combined[valid]/combined[valid].sum()
sel=rng.choice(len(valid),size=min(N_TARGET,len(valid)),replace=(N_TARGET>len(valid)),p=probs)
selected=valid[sel]
if len(selected)<N_TARGET:
    extra=rng.choice(N_BG,size=N_TARGET-len(selected),replace=False)
    selected=np.concatenate([selected,extra])
gan_pts=bg_pts[selected]

# IDW features for selected points
d,ix=fire_tree.query(gan_pts,k=7); d=np.maximum(d,1e-9)
w=1/d**2; w/=w.sum(1,keepdims=True)
gan_feats=np.einsum('nk,nkf->nf',w,fire_feats[ix])
gan_feats+=np.random.RandomState(77).normal(0,0.05*fire_feats.std(0),gan_feats.shape)

# -- Diagnostics ---------------------------------------------------------------
dd_fire,_=fire_tree.query(gan_pts,k=1)
nn_tree=cKDTree(gan_pts); dd_nn,_=nn_tree.query(gan_pts,k=2); dd_nn=dd_nn[:,1]
lon_b=np.linspace(lon_min,lon_max,11); lat_b=np.linspace(lat_min,lat_max,11)
li=np.clip(np.digitize(gan_pts[:,0],lon_b[:-1])-1,0,9)
lj=np.clip(np.digitize(gan_pts[:,1],lat_b[:-1])-1,0,9)
counts=np.bincount(li*10+lj,minlength=100)
border=((gan_pts[:,0]-lon_min<0.15)|(lon_max-gan_pts[:,0]<0.15)|
        (gan_pts[:,1]-lat_min<0.15)|(lat_max-gan_pts[:,1]<0.15))
print(f"\n-- GAN diagnostics ------------------------------------------")
print(f"  Total pts: {len(gan_pts)}")
print(f"  In water: {in_water(gan_pts).sum()}")
print(f"  Min dist to fire: {dd_fire.min()*111:.1f} km")
print(f"  Mean NN dist: {dd_nn.mean():.4f}deg")
print(f"  Grid variance (10x10): {counts.var():.1f}")
print(f"  Lon [{gan_pts[:,0].min():.3f}, {gan_pts[:,0].max():.3f}]")
print(f"  Lat [{gan_pts[:,1].min():.3f}, {gan_pts[:,1].max():.3f}]")
print(f"  Border fraction (<0.15deg): {border.mean():.3f}")
print(f"  Total time: {time.time()-t0:.1f}s")
print(f"-------------------------------------------------------------")

pts_path=f'../../outputs/gan_pts_{CRITIC_CONSTRAINT}.npy'; feats_path=f'../../outputs/gan_feats_{CRITIC_CONSTRAINT}.npy'
np.save(pts_path, gan_pts)
np.save(feats_path, gan_feats)
print(f"Saved {pts_path} {gan_pts.shape}, {feats_path} {gan_feats.shape}")
