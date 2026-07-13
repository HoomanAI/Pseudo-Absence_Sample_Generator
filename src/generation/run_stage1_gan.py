"""
Stage 1: PA generation — Random | BP_V6 | SA | GAN (WGAN, NumPy)
N_PA = 5500 pseudo-absences per method.
"""
import numpy as np, pandas as pd, time
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, MultiPolygon
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)
N_PA = 5500  # target pseudo-absences per method

# ── Water mask ────────────────────────────────────────────────────────────────
def _epoly(cx,cy,sx,sy,n=60):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return Polygon(zip(cx+sx*np.cos(t),cy+sy*np.sin(t)))

_WP=[_epoly(-115.36,55.43,0.69,0.115),_epoly(-115.49,55.78,0.25,0.090),
     _epoly(-113.19,55.27,0.11,0.095),_epoly(-113.27,54.73,0.10,0.075),
     _epoly(-114.70,55.33,0.07,0.055),_epoly(-113.10,55.45,0.07,0.055)]
_WB=[p.bounds for p in _WP]

def in_water_arr(pts):
    r=np.zeros(len(pts),bool)
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        ib=(pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
        for i in np.where(ib)[0]:
            if poly.contains(Point(pts[i,0],pts[i,1])): r[i]=True
    return r

def in_water1(lo,la):
    for poly,(x0,y0,x1,y1) in zip(_WP,_WB):
        if x0<=lo<=x1 and y0<=la<=y1 and poly.contains(Point(lo,la)): return True
    return False

print("Water mask: 6 lakes ready")

# ── Load fire data ────────────────────────────────────────────────────────────
df=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire=df[['LONGITUDE','LATITUDE']].values
feat_cols=[c for c in df.columns if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]
fire_feats=df[feat_cols].values.astype(float)
fire_tree=cKDTree(fire)
N_FIRE=len(fire)
DEG=1/111.0; D_MIN=10*DEG; LAM=20*DEG
lon_min,lon_max=fire[:,0].min()-0.3,fire[:,0].max()+0.3
lat_min,lat_max=fire[:,1].min()-0.3,fire[:,1].max()+0.3
print(f"Loaded {N_FIRE} fire pts | N_PA={N_PA} | bbox lon[{lon_min:.2f},{lon_max:.2f}] lat[{lat_min:.2f},{lat_max:.2f}]")

# ── 1. Random PA ──────────────────────────────────────────────────────────────
pool=[]
while len(pool)<N_PA:
    b=np.column_stack([np.random.uniform(lon_min,lon_max,N_PA*5),
                       np.random.uniform(lat_min,lat_max,N_PA*5)])
    pool.extend(b[~in_water_arr(b)].tolist())
rand_pts=np.array(pool[:N_PA])
print(f"[Random] {len(rand_pts)} pts")

# ── 2. BP_V6 PA ──────────────────────────────────────────────────────────────
GRID_RES=0.5
lon_e=np.arange(lon_min,lon_max+GRID_RES,GRID_RES)
lat_e=np.arange(lat_min,lat_max+GRID_RES,GRID_RES)
n_lo,n_la=len(lon_e)-1,len(lat_e)-1
fi=np.clip(np.digitize(fire[:,0],lon_e)-1,0,n_lo-1)
fj=np.clip(np.digitize(fire[:,1],lat_e)-1,0,n_la-1)
fc=np.zeros((n_lo,n_la),int)
for a,b in zip(fi,fj): fc[a,b]+=1
F_max=fc.max()
alloc=np.ceil(fc/F_max*N_PA).astype(int)
print(f"[BP_V6] grid {n_lo}x{n_la} | F_max={F_max} | sum(alloc)={alloc.sum()}")
heur_pool=[]
for i in range(n_lo):
  for j in range(n_la):
    ni=alloc[i,j]
    if ni==0: continue
    lo0,hi0=lon_e[i],lon_e[i+1]; la0,hi1=lat_e[j],lat_e[j+1]
    acc=[]; att=0
    while len(acc)<ni and att<100:
        bsz=max(5000,ni*20)
        bt=np.column_stack([np.random.uniform(lo0,hi0,bsz),np.random.uniform(la0,hi1,bsz)])
        d,_=fire_tree.query(bt,k=1); ok=bt[d>=D_MIN]; okd=d[d>=D_MIN]
        land=~in_water_arr(ok); ok=ok[land]; okd=okd[land]
        p=1-np.exp(-(okd-D_MIN)/LAM); keep=np.random.uniform(0,1,len(ok))<=p
        ok=ok[keep]; acc.extend(ok[:ni-len(acc)].tolist()); att+=1
    heur_pool.extend(acc)
np.random.shuffle(heur_pool)
# top up with random if pool short
if len(heur_pool)<N_PA:
    print(f"  BP_V6 pool={len(heur_pool)} < {N_PA}, topping up with random buffer points")
    top=[]
    while len(top)<N_PA-len(heur_pool):
        bt=np.column_stack([np.random.uniform(lon_min,lon_max,10000),
                            np.random.uniform(lat_min,lat_max,10000)])
        d,_=fire_tree.query(bt,k=1); bt=bt[d>=D_MIN]; bt=bt[~in_water_arr(bt)]
        top.extend(bt.tolist())
    heur_pool.extend(top[:N_PA-len(heur_pool)])
heur_pts=np.array(heur_pool[:N_PA])
print(f"[BP_V6] pool={len(heur_pool)} | selected={len(heur_pts)}")

# ── 3. SA PA ─────────────────────────────────────────────────────────────────
def gen_sa(n_target,n_iter=55000,T0=1.0,Tf=0.001,nc=12,sf=0.25):
    print(f"[SA] n_target={n_target} n_iter={n_iter}")
    le=np.linspace(lon_min,lon_max,nc+1); ae=np.linspace(lat_min,lat_max,nc+1)
    def cid(p):
        return int(np.clip(np.digitize(p[0],le)-1,0,nc-1))*nc+int(np.clip(np.digitize(p[1],ae)-1,0,nc-1))
    pts=[]; att=0
    while len(pts)<n_target:
        bt=np.column_stack([np.random.uniform(lon_min,lon_max,20000),np.random.uniform(lat_min,lat_max,20000)])
        d,_=fire_tree.query(bt,k=1); bt=bt[d>=D_MIN]; bt=bt[~in_water_arr(bt)]
        pts.extend(bt[:n_target-len(pts)].tolist()); att+=1
        if att>40: break
    pts=np.array(pts[:n_target])
    cts=np.zeros(nc*nc); cids=np.array([cid(p) for p in pts],int)
    for c in cids: cts[c]+=1
    E=float(np.var(cts))
    cool=(Tf/T0)**(1.0/n_iter); T=T0; acc=0
    lr=lon_max-lon_min; ar=lat_max-lat_min
    for i in range(n_iter):
        idx=np.random.randint(n_target); op=pts[idx]; oc=cids[idx]
        np_=np.array([op[0]+np.random.normal(0,T*lr*sf),op[1]+np.random.normal(0,T*ar*sf)])
        if not(lon_min<=np_[0]<=lon_max and lat_min<=np_[1]<=lat_max): T*=cool; continue
        dn,_=fire_tree.query(np_.reshape(1,2),k=1)
        if dn[0]<D_MIN: T*=cool; continue
        if in_water1(np_[0],np_[1]): T*=cool; continue
        nc_=cid(np_)
        if nc_!=oc:
            cts[oc]-=1; cts[nc_]+=1; En=float(np.var(cts)); dE=En-E
            if dE<0 or np.random.random()<np.exp(-dE/(T+1e-30)):
                pts[idx]=np_; cids[idx]=nc_; E=En; acc+=1
            else: cts[oc]+=1; cts[nc_]-=1
        else: pts[idx]=np_; acc+=1
        T*=cool
        if (i+1)%10000==0: print(f"  iter {i+1} T={T:.5f} E={E:.2f} acc={acc/(i+1)*100:.1f}%")
    print(f"  done E={E:.2f} acc={100*acc/n_iter:.1f}%")
    return pts
sa_pts=gen_sa(N_PA)
wn=in_water_arr(sa_pts).sum()
print(f"[SA] {len(sa_pts)} pts | water={wn}")

# ── 4. WGAN (NumPy) ───────────────────────────────────────────────────────────
class Dense:
    def __init__(self,id,od):
        sc=np.sqrt(2.0/(id+od))
        self.W=np.random.randn(id,od)*sc; self.b=np.zeros(od)
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
    def fwd(self,x): self.x=x; return x@self.W+self.b
    def bwd(self,g):
        n=len(g); self.gW=self.x.T@g/n; self.gb=g.mean(0)
        return g@self.W.T
    def clip(self,c=0.05): np.clip(self.W,-c,c,out=self.W); np.clip(self.b,-c,c,out=self.b)
    def adam(self,lr,t,b1=0.5,b2=0.9,e=1e-8):
        for p,g,m,v in[(self.W,self.gW,self.mW,self.vW),(self.b,self.gb,self.mb,self.vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            p-=lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+e)

def lrelu(x,a=0.2): return np.where(x>0,x,a*x)
def dlr(x,a=0.2): return np.where(x>0,1.,a)
def dtanh(x): return 1-np.tanh(x)**2

class Generator:
    def __init__(self,z=64): self.z=z; self.L=[Dense(z,128),Dense(128,64),Dense(64,2)]
    def fwd(self,z):
        h1=self.L[0].fwd(z); a1=lrelu(h1)
        h2=self.L[1].fwd(a1); a2=lrelu(h2)
        h3=self.L[2].fwd(a2); out=np.tanh(h3)
        self.c=(h1,h2,h3); return out
    def bwd(self,g):
        h1,h2,h3=self.c; g=g*dtanh(h3)
        g=self.L[2].bwd(g); g=g*dlr(h2)
        g=self.L[1].bwd(g); g=g*dlr(h1)
        self.L[0].bwd(g)
    def step(self,lr,t):
        for l in self.L: l.adam(lr,t)
    def sample(self,n): return self.fwd(np.random.randn(n,self.z))

class Critic:
    def __init__(self): self.L=[Dense(2,64),Dense(64,128),Dense(128,1)]
    def fwd(self,x):
        h1=self.L[0].fwd(x); a1=lrelu(h1)
        h2=self.L[1].fwd(a1); a2=lrelu(h2)
        out=self.L[2].fwd(a2); self.c=(h1,h2); return out
    def bwd(self,g):
        h1,h2=self.c
        g=self.L[2].bwd(g); g=g*dlr(h2)
        g=self.L[1].bwd(g); g=g*dlr(h1)
        return self.L[0].bwd(g)
    def clip(self,c=0.05):
        for l in self.L: l.clip(c)
    def step(self,lr,t):
        for l in self.L: l.adam(lr,t)

def train_wgan(data_norm, epochs=600, bs=64, z=64, n_crit=5, clip_c=0.05, lr=5e-4):
    np.random.seed(0)
    G=Generator(z); D=Critic()
    N=len(data_norm); t_g=0; t_d=0; hist=[]
    t0=time.time()
    for ep in range(epochs):
        idx=np.random.permutation(N)
        for s in range(0,N-bs,bs):
            real=data_norm[idx[s:s+bs]]
            # Critic steps
            for _ in range(n_crit):
                t_d+=1
                fake=G.sample(bs)
                x_all=np.vstack([real,fake])
                D.fwd(x_all)
                g=np.vstack([-np.ones((bs,1))/bs, np.ones((bs,1))/bs])
                D.bwd(g); D.step(lr,t_d); D.clip(clip_c)
            # Generator step
            t_g+=1
            fake=G.fwd(np.random.randn(bs,z))
            D.fwd(fake)
            dx=D.bwd(-np.ones((bs,1))/bs)
            G.bwd(dx); G.step(lr,t_g)
        if (ep+1)%100==0:
            s1=G.sample(500); dr=D.fwd(data_norm[:200]).mean(); df=D.fwd(s1[:200]).mean()
            w=float(dr-df); hist.append((ep+1,w))
            print(f"  ep {ep+1:4d} | W={w:+.4f} | {time.time()-t0:.1f}s")
    return G, D, hist

print("\n[GAN] Normalising fire coords to [-1,1]")
lo_c=np.array([(lon_min+lon_max)/2,(lat_min+lat_max)/2])
lo_s=np.array([(lon_max-lon_min)/2,(lat_max-lat_min)/2])
fire_n=(fire-lo_c)/lo_s

print("[GAN] Training WGAN …")
G,D,gan_hist=train_wgan(fire_n)
np.save('../../outputs/gan_hist.npy',np.array(gan_hist))

print("[GAN] Generating PA candidates …")
n_gen=max(80000,N_PA*20)
raw=G.sample(n_gen)
pts_g=raw*lo_s+lo_c
# bbox clip
m=(pts_g[:,0]>=lon_min)&(pts_g[:,0]<=lon_max)&(pts_g[:,1]>=lat_min)&(pts_g[:,1]<=lat_max)
pts_g=pts_g[m]
# fire buffer
d,_=fire_tree.query(pts_g,k=1); pts_g=pts_g[d>=D_MIN]
# water
pts_g=pts_g[~in_water_arr(pts_g)]
print(f"[GAN] after filters: {len(pts_g)} candidates")
if len(pts_g)<N_PA:
    print(f"  pool short ({len(pts_g)}), topping up with BP_V6-style random+buffer")
    top=[]
    while len(top)<N_PA-len(pts_g):
        bt=np.column_stack([np.random.uniform(lon_min,lon_max,20000),
                            np.random.uniform(lat_min,lat_max,20000)])
        d2,_=fire_tree.query(bt,k=1); bt=bt[d2>=D_MIN]; bt=bt[~in_water_arr(bt)]
        top.extend(bt.tolist())
    pts_g=np.vstack([pts_g,np.array(top[:N_PA-len(pts_g)])])
np.random.shuffle(pts_g)
gan_pts=pts_g[:N_PA]
wn=in_water_arr(gan_pts).sum()
print(f"[GAN] {len(gan_pts)} pts | water={wn}")

# ── 5. Spatial evaluation ─────────────────────────────────────────────────────
def k_sse(p):
    rv=np.linspace(0.05,0.8,15)
    area=(fire[:,0].max()-fire[:,0].min())*(fire[:,1].max()-fire[:,1].min())
    def kc(q):
        t=cKDTree(q); lam=len(q)/area
        return np.array([(np.array(t.query_ball_point(q,r,return_length=True))-1).mean()/lam for r in rv])
    return float(np.sum((kc(p)-kc(fire))**2))

def cen(p): return float(np.sqrt(((p.mean(0)-fire.mean(0))**2).sum()))
def gvar(p,n=10):
    le=np.linspace(p[:,0].min(),p[:,0].max(),n+1); ae=np.linspace(p[:,1].min(),p[:,1].max(),n+1)
    return float(np.var([((p[:,0]>=le[i])&(p[:,0]<le[i+1])&(p[:,1]>=ae[j])&(p[:,1]<ae[j+1])).sum() for i in range(n) for j in range(n)]))
def nn(p): t=cKDTree(p); d,_=t.query(p,k=2); return float(d[:,1].mean())

print("\nSpatial evaluation …")
sh,sr,ss,sg=k_sse(heur_pts),k_sse(rand_pts),k_sse(sa_pts),k_sse(gan_pts)
ch,cr,cs,cg=cen(heur_pts),cen(rand_pts),cen(sa_pts),cen(gan_pts)
gh,gr,gs,gg=gvar(heur_pts),gvar(rand_pts),gvar(sa_pts),gvar(gan_pts)
nh,nr,ns,ng=nn(heur_pts),nn(rand_pts),nn(sa_pts),nn(gan_pts)
nf=nn(fire)

sp=pd.DataFrame({'Metric':['K SSE','Centroid','Grid Var','Mean NN'],
    'Heuristic':[round(sh,3),round(ch,4),round(gh,2),round(nh,4)],
    'SA':[round(ss,3),round(cs,4),round(gs,2),round(ns,4)],
    'GAN':[round(sg,3),round(cg,4),round(gg,2),round(ng,4)],
    'Random':[round(sr,3),round(cr,4),round(gr,2),round(nr,4)],
    'Fire_Ref':['0.00','0.0000','--',round(nf,4)]})
print(sp.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
np.save('../../outputs/heur_pts.npy',heur_pts)
np.save('../../outputs/rand_pts.npy',rand_pts)
np.save('../../outputs/sa_pts.npy',sa_pts)
np.save('../../outputs/gan_pts.npy',gan_pts)
np.save('../../outputs/spatial_stats.npy',
        np.array([sh,sr,ss,sg,ch,cr,cs,cg,gh,gr,gs,gg,nh,nr,ns,ng,nf]))
sp.to_csv('../../outputs/spatial_results.csv',index=False)
for pts,nm in[(heur_pts,'heuristic'),(rand_pts,'random'),(sa_pts,'sa'),(gan_pts,'gan')]:
    pd.DataFrame(pts,columns=['LONGITUDE','LATITUDE']).to_csv(
        f'../../outputs/{nm}_pseudo_absences.csv',index=False)
print("\nStage 1 complete.")
