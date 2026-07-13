"""Spatial evaluation for all 4 methods (5500 pts each)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts = df[['LONGITUDE','LATITUDE']].values
OUT = '../../outputs/'

heur_pts = np.load(OUT+'heur_pts.npy')
rand_pts  = np.load(OUT+'rand_pts.npy')
sa_pts    = np.load(OUT+'sa_pts.npy')
gan_pts   = np.load(OUT+'gan_pts.npy')

def k_sse(pts_test):
    r_vals = np.linspace(0.05,0.8,15)
    area = (fire_pts[:,0].max()-fire_pts[:,0].min())*(fire_pts[:,1].max()-fire_pts[:,1].min())
    def kc(pts):
        t=cKDTree(pts); lam=len(pts)/area
        return np.array([(np.array(t.query_ball_point(pts,r,return_length=True))-1).mean()/lam for r in r_vals])
    return float(np.sum((kc(pts_test)-kc(fire_pts))**2))

def centroid_d(p): return float(np.sqrt(((p.mean(0)-fire_pts.mean(0))**2).sum()))
def grid_var(pts,n=10):
    le=np.linspace(pts[:,0].min(),pts[:,0].max(),n+1)
    ae=np.linspace(pts[:,1].min(),pts[:,1].max(),n+1)
    return float(np.var([((pts[:,0]>=le[i])&(pts[:,0]<le[i+1])&(pts[:,1]>=ae[j])&(pts[:,1]<ae[j+1])).sum()
                         for i in range(n) for j in range(n)]))
def nn_d(pts):
    t=cKDTree(pts); d,_=t.query(pts,k=2); return float(d[:,1].mean())

print("Computing spatial metrics ...")
rows=[]
for name,pts in [('Heuristic',heur_pts),('Random',rand_pts),('SA',sa_pts),('GAN',gan_pts)]:
    rows.append({'Method':name,'K_SSE':round(k_sse(pts),3),'Centroid':round(centroid_d(pts),4),
                 'Grid_Var':round(grid_var(pts),2),'Mean_NN':round(nn_d(pts),4)})
    print(f"  {name} done")
nn_f=nn_d(fire_pts)
sp_df=pd.DataFrame(rows)
sp_df['Fire_NN_ref']=round(nn_f,4)
print(sp_df.to_string(index=False))

stats=np.array([sp_df.loc[sp_df.Method==m,col].values[0]
                for m in ['Heuristic','Random','SA','GAN']
                for col in ['K_SSE','Centroid','Grid_Var','Mean_NN']] + [nn_f])
np.save(OUT+'spatial_stats.npy', stats)
sp_df.to_csv(OUT+'spatial_results.csv',index=False)
print("Spatial eval saved.")
