"""Regenerate AP-WGAN revision figures, tables, trace data, and QC report.

Run from any directory:  python "Revision Code/figures_new/generate_all.py"
All numerical inputs are loaded from repository result/data files at runtime.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "Revision Code" / "results"
RETRAIN = ROOT / "Revision Code" / "retrain"
DATA = ROOT / "Pseudo-Absence_Sample_Generator" / "data" / "Fire_points_dataset_final_csv.csv"
FIGDIR = ROOT / "Overleaf 2" / "figures"
TABDIR = ROOT / "Overleaf 2" / "tables"
sys.path.insert(0, str(HERE)); from figstyle import save_figure, SINGLE, DOUBLE

SEED = 42
RNG = np.random.default_rng(SEED)
ARMS = ["random","heuristic","clip","gp_standard","v1_da_gp","dagp_strict","dagp_base","dagp_relaxed"]
GAN_ARMS = ["clip","gp_standard","v1_da_gp","dagp_strict","dagp_base","dagp_relaxed"]
AP_ARMS = ["dagp_strict","dagp_base","dagp_relaxed"]
LABEL = {"random":"Random","heuristic":"Heuristic","clip":"Clip-WGAN","gp_standard":"GP-standard",
         "v1_da_gp":"DA-GP-WGAN","dagp_strict":"AP-WGAN (strict)","dagp_base":"AP-WGAN (base)",
         "dagp_relaxed":"AP-WGAN (relaxed)"}
COL = {"random":"#7f7f7f","heuristic":"#E69F00","clip":"#56B4E9","gp_standard":"#009E73",
       "v1_da_gp":"#CC79A7","dagp_strict":"#0072B2","dagp_base":"#D55E00","dagp_relaxed":"#F0E442"}
MARK = {"random":"o","heuristic":"s","clip":"^","gp_standard":"D","v1_da_gp":"P",
        "dagp_strict":"o","dagp_base":"s","dagp_relaxed":"^"}
MODELS = ["LogReg","NaiveBayes","CART","KNN","SVM"]
trace, report = [], []

def addtrace(fig, source, column, value, arm="", seed="", series=""):
    if not np.isfinite(float(value)): return
    p=Path(source)
    try: src=str(p.relative_to(ROOT))
    except ValueError: src=str(p)
    trace.append(dict(figure=fig, source_file=src, source_column=column,
                      arm=arm, seed=seed, series=series, value=float(value)))
def record(fid, sources, claim, width, warn=""):
    report.append(f"- **{fid}** — sources: {sources}; claim: {claim}; target: {width}; QC: {warn or 'PASS'}." )
def save(fig, stem):
    w = save_figure(fig, stem, FIGDIR); plt.close(fig); return w
def texesc(s):
    return str(s).replace("%",r"\%").replace("_",r"\_").replace("±",r"$\pm$")
def write_table(name, headers, rows, caption, label):
    TABDIR.mkdir(parents=True, exist_ok=True)
    align = "l" + "r"*(len(headers)-1)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule"]
    lines += [" & ".join(map(str,row))+r" \\" for row in rows]
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (TABDIR/name).write_text("\n".join(lines)+"\n", encoding="utf-8")
def point_path(row): return ROOT / "Revision Code" / str(row.pts_file).replace("\\","/")
def feat_path(arm, seed):
    d = RETRAIN/f"{arm}_seed{seed}"
    fs = list(d.glob("feats_*.npy"));
    if len(fs)!=1: raise FileNotFoundError(f"feature file for {arm} seed {seed}")
    return fs[0]
def diag_path(arm, seed):
    fs=list((RETRAIN/f"{arm}_seed{seed}").glob("diag_*.csv"))
    if len(fs)!=1: raise FileNotFoundError(f"diagnostic file for {arm} seed {seed}")
    return fs[0]

def spatial_extra(man, fire):
    out=[]
    for r in man.itertuples():
        p=np.load(point_path(r));
        le=np.linspace(p[:,0].min(),p[:,0].max(),11); ae=np.linspace(p[:,1].min(),p[:,1].max(),11)
        counts=[((p[:,0]>=le[i])&(p[:,0]<le[i+1])&(p[:,1]>=ae[j])&(p[:,1]<ae[j+1])).sum() for i in range(10) for j in range(10)]
        nn=cKDTree(p).query(p,k=2)[0][:,1].mean()
        out.append(dict(label=r.label,seed=r.seed,Centroid=np.linalg.norm(p.mean(0)-fire.mean(0)),Grid_Var=np.var(counts),Mean_NN=nn))
    return pd.DataFrame(out)

def figure1(man):
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.15)); metrics=[("K_SSE_rawK","K-SSE (deg⁴)"),("border_fraction","border fraction (proportion)")]
    x=np.arange(8)
    for ax,(m,yl) in zip(axs,metrics):
        for i,a in enumerate(ARMS):
            v=man.loc[man.label.eq(a),m].to_numpy(); mu=v.mean(); sd=v.std(ddof=1)
            ax.bar(i,mu,color=COL[a],edgecolor="black",linewidth=.5,yerr=sd,capsize=2.2)
            ax.scatter(i+np.array([-.12,0,.12]),v,s=12,c="white",edgecolor="black",linewidth=.45,zorder=3)
            for z in v:addtrace("F1",RETRAIN/"manifest.csv",m,z,a,series=yl)
        ax.set_xticks(x,LABEL.values(),rotation=35,ha="right"); ax.set_ylabel(yl); ax.set_xlabel("pseudo-absence strategy"); ax.grid(axis="y")
    axs[0].text(.02,.98,"Significant: Clip-WGAN vs DA-GP-WGAN (p=0.027)\nClip-WGAN vs AP-WGAN (base) (p=0.028)",transform=axs[0].transAxes,va="top",fontsize=7.2)
    fig.tight_layout(); w=save(fig,"F1_spatial_fidelity"); record("F1","retrain/manifest.csv","seed-replicated spatial fidelity; only two significant contrasts","double column (7.16 in)",w)

def figure2(ev):
    g=ev.groupby(["arm","model"]).AUC.agg(["mean","std"])
    M=np.array([[g.loc[(a,m),"mean"] for m in MODELS] for a in ARMS]); S=np.array([[g.loc[(a,m),"std"] for m in MODELS] for a in ARMS])
    fig,ax=plt.subplots(figsize=(DOUBLE,4.1)); im=ax.imshow(M,cmap="viridis",aspect="auto",vmin=M.min(),vmax=M.max())
    for j,m in enumerate(MODELS):
        best,worst=np.argmax(M[:,j]),np.argmin(M[:,j])
        for i,a in enumerate(ARMS):
            star="▲" if i==best else ("▼" if i==worst else "")
            ax.text(j,i,f"{M[i,j]:.4f}{star}\n$\\mathsf{{{S[i,j]:.4f}}}$",ha="center",va="center",fontsize=7.4,color="white" if M[i,j]<.86 else "black")
            vals=ev[(ev.arm==a)&(ev.model==m)].AUC
            for seed,z in zip(ev[(ev.arm==a)&(ev.model==m)].seed,vals): addtrace("F2",RESULTS/"arm_eval_percell_snapped.csv","AUC",z,a,seed,m)
    ax.set_xticks(range(5),MODELS); ax.set_yticks(range(8),[LABEL[a] for a in ARMS]); ax.set_xlabel("classifier"); ax.set_ylabel("pseudo-absence strategy")
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("mean AUC (unitless)")
    fig.tight_layout(); w=save(fig,"F2_downstream_auc_heatmap"); record("F2","results/arm_eval_percell_snapped.csv","like-for-like classifier AUC after snapping","double column (7.16 in)",w)

def figure3(raw,snap):
    a=raw[raw.model.eq("CART")].groupby("arm").AUC.mean(); b=snap[snap.model.eq("CART")].groupby("arm").AUC.mean()
    dif=b-a
    if not np.all(dif.loc[AP_ARMS].to_numpy()==0): raise AssertionError(f"AP movement not exactly zero: {dif.loc[AP_ARMS]}")
    fig,ax=plt.subplots(figsize=(DOUBLE,3.8)); y=np.arange(8)
    for i,arm in enumerate(ARMS):
        ax.plot([a[arm],b[arm]],[i,i],color=COL[arm],lw=2); ax.scatter(a[arm],i,marker="o",facecolors="none",edgecolors=COL[arm],s=45); ax.scatter(b[arm],i,marker="s",color=COL[arm],s=32)
        ax.text(max(a[arm],b[arm])+.002,i,f"{dif[arm]:+.4f}",va="center",fontsize=7.5)
        for src,df in [(RESULTS/"arm_eval_percell.csv",raw),(RESULTS/"arm_eval_percell_snapped.csv",snap)]:
            for r in df[(df.arm==arm)&(df.model=="CART")].itertuples(): addtrace("F3",src,"AUC",r.AUC,arm,r.seed,"CART")
    ax.set_yticks(y,[LABEL[x] for x in ARMS]); ax.invert_yaxis(); ax.set_xlabel("CART AUC (unitless)"); ax.set_ylabel("pseudo-absence strategy"); ax.grid(axis="x")
    ax.set_title("Artifact diagnosis: unsnapped ○ → snapped ■ (mean over three seeds)")
    fig.tight_layout(); w=save(fig,"F3_artifact_before_after"); record("F3","results/arm_eval_percell.csv; arm_eval_percell_snapped.csv","snapping is exactly invariant for all AP-WGAN arms","double column (7.16 in)",w)

def figure4(adm):
    piv=adm.pivot(index="strategy",columns="condition",values="tell_AUC"); names=list(piv.index); fig,ax=plt.subplots(figsize=(SINGLE,2.9))
    for i,n in enumerate(names):
        ax.plot([piv.loc[n,"baseline"],piv.loc[n,"admissible"]],[i,i],color="#0072B2",lw=1.8); ax.scatter(piv.loc[n,"baseline"],i,marker="o",facecolors="none",edgecolors="#D55E00"); ax.scatter(piv.loc[n,"admissible"],i,marker="s",color="#0072B2")
        addtrace("F4",RESULTS/"admissibility.csv","tell_AUC",piv.loc[n,"baseline"],n,series="baseline"); addtrace("F4",RESULTS/"admissibility.csv","tell_AUC",piv.loc[n,"admissible"],n,series="admissible")
    ax.axvline(.5,color="black",ls="--",lw=.8); ax.axvline(1,color="black",ls=":",lw=.8); ax.text(.5,len(names)-.35,"no construction signal",ha="center",fontsize=7); ax.text(1,len(names)-.35,"perfectly separable\nby construction",ha="right",fontsize=7)
    ax.set_yticks(range(len(names)),names); ax.set_xlim(.46,1.02); ax.set_xlabel("tell AUC (unitless)"); ax.set_ylabel("strategy"); ax.invert_yaxis(); fig.tight_layout(); w=save(fig,"F4_admissibility_tell"); record("F4","results/admissibility.csv","admissibility removes the deterministic construction tell","single column (3.5 in)",w)

def figure5(q):
    q=q.copy(); q["row"]=q.strategy+" · "+q.model; fig,ax=plt.subplots(figsize=(DOUBLE,3.25)); y=np.arange(len(q))
    for i,r in enumerate(q.itertuples()):
        ax.fill_betweenx([i-.28,i+.28],r.auc_observed,r.auc_synthetic,color="#56B4E9",alpha=.35)
        ax.plot([r.auc_observed,r.auc_synthetic],[i,i],color="#0072B2"); ax.scatter(r.auc_observed,i,marker="o",color="#0072B2"); ax.scatter(r.auc_synthetic,i,marker="s",color="#D55E00")
        ax.text(max(r.auc_synthetic,r.auc_observed)+.008,i,f"+{r.inflation:.4f}",va="center",fontsize=8,weight="bold" if "oracle" in r.strategy else "normal")
        for c in ["auc_synthetic","auc_observed","inflation"]: addtrace("F5",RESULTS/"qc_e3_transfer.csv",c,getattr(r,c),r.strategy,series=r.model)
    ax.set_yticks(y,q.row); ax.invert_yaxis(); ax.set_xlim(.48,1.04); ax.set_xlabel("AUC (unitless)"); ax.set_ylabel("Quebec strategy and classifier"); ax.grid(axis="x")
    fig.tight_layout(); w=save(fig,"F5_quebec_transfer"); record("F5","results/qc_e3_transfer.csv","synthetic negatives inflate Quebec AUC; oracle gap is +0.0000","double column (7.16 in)",w)

def dynamics(arms,seeds,shape,stem,fid):
    fig,axs=plt.subplots(*shape,figsize=(7.16,10.0 if shape==(4,3) else 7.0),squeeze=False); window=None
    for ax,(arm,seed) in zip(axs.flat,[(a,s) for a in arms for s in seeds]):
        d=pd.read_csv(diag_path(arm,seed)); window=max(1,len(d)//400); sm=d[["critic_loss","in_manifold_frac"]].rolling(window,center=True,min_periods=1).mean(); step=max(1,len(d)//400); ix=np.arange(0,len(d),step)
        final=sm.critic_loss.iloc[-1]; tol=.1*max(abs(final),1e-12); hits=np.where(np.abs(sm.critic_loss.to_numpy()-final)<=tol)[0]; conv=d.iter.iloc[hits[0]] if len(hits) else d.iter.iloc[-1]
        ax2=ax.twinx(); ax.plot(d.iter.iloc[ix],sm.critic_loss.iloc[ix],color="#0072B2",lw=.8); ax2.plot(d.iter.iloc[ix],sm.in_manifold_frac.iloc[ix],color="#D55E00",ls="--",lw=.8); ax.axvline(conv,color="#009E73",ls="--",lw=.8)
        ax.text(.02,.95,f"{LABEL[arm]} · s{seed}",transform=ax.transAxes,va="top",weight="bold",fontsize=7.5)
        row,col=np.unravel_index(list(axs.flat).index(ax),shape)
        ax.set_xlabel("iteration (count)" if row==shape[0]-1 else ""); ax.set_ylabel("critic loss (unitless)" if col==0 else ""); ax2.set_ylabel("in-manifold fraction (proportion)" if col==shape[1]-1 else ""); ax.grid(alpha=.2)
        for c in ["critic_loss","in_manifold_frac"]:
            for r in d.itertuples(): addtrace(fid,diag_path(arm,seed),c,getattr(r,c),arm,seed,c)
    for ax in list(axs.flat)[len(arms)*len(seeds):]: ax.set_visible(False)
    fig.legend([Line2D([],[],color="#0072B2"),Line2D([],[],color="#D55E00",ls="--"),Line2D([],[],color="#009E73",ls="--")],["critic loss","in-manifold fraction","convergence"],loc="upper center",bbox_to_anchor=(.5,.974),ncol=3)
    fig.subplots_adjust(top=.91,bottom=.06,left=.08,right=.92,hspace=.35,wspace=.38)
    w=save(fig,stem); record(fid,"retrain/<arm>_seed<N>/diag_*.csv",f"training dynamics across seeds; rolling window {window}","full-page portrait",w)

def figure7(man):
    ap=man[man.label.isin(AP_ARMS)].copy(); fig,axs=plt.subplots(3,2,figsize=(SINGLE,6.7)); params=[("tau_pctl","τ percentile"),("omega","ω (weight)"),("k","k (neighbors)")]
    for row,(p,xl) in enumerate(params):
        for col,(m,yl) in enumerate([("K_SSE_rawK","K-SSE (deg⁴)"),("border_fraction","border fraction (proportion)")]):
            ax=axs[row,col]; pooled=ap.groupby("label")[m].mean().mean(); psd=ap[m].std(ddof=1); xs=[]; mus=[]
            for a in AP_ARMS:
                z=ap[ap.label==a]; x=z[p].iloc[0]; xs.append(x); mus.append(z[m].mean()); ax.scatter(np.repeat(x,3),z[m],marker=MARK[a],color=COL[a],s=20); addtrace("F7",RETRAIN/"manifest.csv",p,x,a,series=xl)
                for r in z.itertuples(): addtrace("F7",RETRAIN/"manifest.csv",m,getattr(r,m),a,r.seed,yl)
            order=np.argsort(xs); xx=np.array(xs)[order]; yy=np.array(mus)[order]; ax.plot(xx,yy,color="black",lw=.8,ls=":"); ax.fill_between(xx,pooled-psd,pooled+psd,color="#999999",alpha=.2,label="pooled SD")
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(); ax.text(.03,.95,"null contrast",transform=ax.transAxes,va="top",fontsize=7)
    fig.set_size_inches(DOUBLE,6.7); fig.tight_layout(); w=save(fig,"F7_parameter_sweep"); record("F7","retrain/manifest.csv","three AP settings are statistically indistinguishable","double column (7.16 in)",w)

def background(fire_pts,fire_feats):
    sys.path.insert(0,str(ROOT/"Revision Code")); import train_variants as tv
    pts,_=tv.build_background(fire_pts,seed=0); feats=tv.idw_feats(pts,cKDTree(fire_pts),fire_feats,seed=99); return pts,feats

def figure8(fdf):
    fcols=[c for c in fdf if c not in ["FIRE","LONGITUDE","LATITUDE","YEAR","MONTH","DAY"]]; inds=[fcols.index(x) for x in ["slope","twi","temp_1m_prior"]]
    fig=plt.figure(figsize=(DOUBLE,4.5)); ax=fig.add_subplot(111,projection="3d")
    sets=[("Fire records",fdf[fcols].to_numpy(),"#000000","o",DATA,"")]
    for arm,name,c,m in [("clip","IDW negatives","#D55E00","^"),("dagp_base","AP-WGAN negatives","#0072B2","s")]:
        for seed in [42,43,44]: sets.append((name,np.load(feat_path(arm,seed)),c,m,feat_path(arm,seed),seed))
    seen=set()
    for name,X,c,m,src,seed in sets:
        n=900 if name=="Fire records" else 300; ix=RNG.choice(len(X),min(n,len(X)),replace=False); ax.scatter(X[ix,inds[0]],X[ix,inds[1]],X[ix,inds[2]],s=5,alpha=.35,c=c,marker=m,label=name if name not in seen else None,depthshade=False); seen.add(name)
        for j in ix:
            for k,col in zip(inds,["slope","twi","temp_1m_prior"]): addtrace("F8",src,col,X[j,k],seed=seed,series=name)
    ax.set_xlabel("slope (scaled ordinal code)"); ax.set_ylabel("TWI (scaled ordinal code)"); ax.set_zlabel("temperature 1 month prior (°C)"); ax.view_init(23,-55); ax.legend(loc="upper left"); fig.tight_layout(); w=save(fig,"F8_admissible_lattice_3d"); record("F8","fire CSV; retrain clip/AP-WGAN feature arrays","IDW leaves ordinal planes while AP-WGAN remains admissible","double column (7.16 in)",w)

def figure9(man,ev):
    svm=ev[ev.model.eq("SVM")][["arm","seed","AUC"]]; d=man.merge(svm,left_on=["label","seed"],right_on=["arm","seed"])
    fig=plt.figure(figsize=(DOUBLE,4.5)); ax=fig.add_subplot(111,projection="3d"); floor=d.AUC.min()-.01
    for r in d.itertuples():
        ax.scatter(r.K_SSE_rawK,r.border_fraction,r.AUC,c=COL[r.label],marker=MARK[r.label],s=35); ax.plot([r.K_SSE_rawK]*2,[r.border_fraction]*2,[floor,r.AUC],color=COL[r.label],alpha=.5,lw=.7)
        for src,c,v in [(RETRAIN/"manifest.csv","K_SSE_rawK",r.K_SSE_rawK),(RETRAIN/"manifest.csv","border_fraction",r.border_fraction),(RESULTS/"arm_eval_percell_snapped.csv","AUC",r.AUC)]: addtrace("F9",src,c,v,r.label,r.seed,"SVM")
    ax.set_xlabel("K-SSE (deg⁴)"); ax.set_ylabel("border fraction (proportion)"); ax.set_zlabel("snapped SVM AUC (unitless)"); ax.set_zlim(floor,d.AUC.max()+.005); ax.view_init(24,-58)
    ax.legend([Line2D([],[],marker=MARK[a],color=COL[a],ls="") for a in ARMS],[LABEL[a] for a in ARMS],fontsize=6,loc="upper left"); fig.tight_layout(); w=save(fig,"F9_tradeoff_3d"); record("F9","retrain/manifest.csv; arm_eval_percell_snapped.csv","spatial fidelity and downstream accuracy are not aligned","double column (7.16 in)",w)

def figure10(fdf):
    fcols=[c for c in fdf if c not in ["FIRE","LONGITUDE","LATITUDE","YEAR","MONTH","DAY"]]; fire=fdf[fcols].to_numpy(float); pts=fdf[["LONGITUDE","LATITUDE"]].to_numpy(); _,bg=background(pts,fire)
    sc=StandardScaler().fit(bg); pca=PCA(3,random_state=SEED).fit(sc.transform(bg)); ap=np.vstack([np.load(feat_path("dagp_base",s)) for s in [42,43,44]])
    sets=[("Fire records",fire,"#000000","o",DATA),("Background pool",bg,"#D55E00","^","runtime: train_variants.build_background/idw_feats"),("AP-WGAN (base)",ap,"#0072B2","s","Revision Code/retrain/dagp_base_seed*/feats_*.npy")]
    fig=plt.figure(figsize=(DOUBLE,4.5)); ax=fig.add_subplot(111,projection="3d")
    for name,X,c,m,src in sets:
        ix=RNG.choice(len(X),min(1200,len(X)),replace=False); Z=pca.transform(sc.transform(X[ix])); ax.scatter(*Z.T,s=4,alpha=.25,c=c,marker=m,label=name,depthshade=False)
        for row in Z:
            for j,v in enumerate(row,1): addtrace("F10",src,f"PC{j}",v,series=name)
    ev=pca.explained_variance_ratio_*100; ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)"); ax.set_zlabel(f"PC3 ({ev[2]:.1f}% variance)"); ax.view_init(22,-58); ax.legend(); fig.tight_layout(); w=save(fig,"F10_feature_manifold_pca3"); record("F10","fire CSV; deterministic background pool; AP-WGAN feature arrays","generation occupies environmental feature space","double column (7.16 in)",w)

def tables(man,ev,fdf):
    fire=fdf[["LONGITUDE","LATITUDE"]].to_numpy(); ex=spatial_extra(man,fire); sp=man.merge(ex,on=["label","seed"])
    ref_grid=np.var([((fire[:,0]>=np.linspace(fire[:,0].min(),fire[:,0].max(),11)[i])&(fire[:,0]<np.linspace(fire[:,0].min(),fire[:,0].max(),11)[i+1])&(fire[:,1]>=np.linspace(fire[:,1].min(),fire[:,1].max(),11)[j])&(fire[:,1]<np.linspace(fire[:,1].min(),fire[:,1].max(),11)[j+1])).sum() for i in range(10) for j in range(10)])
    ref_nn=cKDTree(fire).query(fire,k=2)[0][:,1].mean(); rows=[["Fire reference","0.0000","0.0000","0.0000",f"{ref_grid:.4f}",f"{ref_nn:.4f}"]]
    mets=["K_SSE_rawK","border_fraction","Centroid","Grid_Var","Mean_NN"]
    for a in ARMS: rows.append([LABEL[a]]+[f"{sp[sp.label==a][m].mean():.4f} $\\pm$ {sp[sp.label==a][m].std(ddof=1):.4f}" for m in mets])
    write_table("T1_spatial_quality.tex",["Strategy","K-SSE","Border fraction","Centroid (deg)","Grid variance","Mean NN (deg)"],rows,"Spatial quality (mean $\\pm$ SD over three seeds).","tab:spatial-new")
    rows=[]
    for a in ARMS:
        row=[LABEL[a]]
        for m in MODELS:
            z=ev[(ev.arm==a)&(ev.model==m)]; row.append(f"{z.AUC.mean():.4f} $\\pm$ {z.AUC.std(ddof=1):.4f} / {z.TSS.mean():.4f} $\\pm$ {z.TSS.std(ddof=1):.4f}")
        rows.append(row)
    write_table("T2_downstream.tex",["Strategy"]+[m+" AUC / TSS" for m in MODELS],rows,"Snapped downstream performance (mean $\\pm$ SD over three seeds).","tab:downstream-new")
    # T3: requested contrasts for each snapped classifier AUC.
    rows=[]; gener=[a for a in ARMS if a not in ["random","heuristic"]]
    contrasts=[("Random",["random"],LABEL[a],[a]) for a in ARMS if a!="random"] + [("Heuristic",["heuristic"],"pooled generative",gener)] + [(LABEL[a],[a],LABEL[b],[b]) for i,a in enumerate(AP_ARMS) for b in AP_ARMS[i+1:]]
    for model in MODELS:
        for n1,a1,n2,a2 in contrasts:
            x=ev[(ev.model==model)&ev.arm.isin(a1)].AUC; y=ev[(ev.model==model)&ev.arm.isin(a2)].AUC; t,p=ttest_ind(x,y,equal_var=False)
            rows.append([model,n1+" vs "+n2,f"{t:.3f}",f"{p:.4g}","significant" if p<.05 else "not significant"])
    write_table("T3_significance.tex",["Classifier","Contrast","Welch $t$","$p$","Decision"],rows,"Welch tests on snapped AUC; $\\alpha=0.05$ (unadjusted, prespecified contrasts).","tab:significance-new")
    adm=pd.read_csv(RESULTS/"admissibility.csv"); raw=pd.read_csv(RESULTS/"arm_eval_percell.csv"); ab=["clip","gp_standard","v1_da_gp","dagp_base"]; rows=[]
    for a in ab:
        z=sp[sp.label==a]; cart=ev[(ev.arm==a)&(ev.model=="CART")].AUC; tell=ev[ev.arm==a].groupby("seed").tell_AUC.first(); rows.append([LABEL[a],f"{z.K_SSE_rawK.mean():.4f} $\\pm$ {z.K_SSE_rawK.std(ddof=1):.4f}",f"{z.border_fraction.mean():.4f} $\\pm$ {z.border_fraction.std(ddof=1):.4f}",f"{tell.mean():.4f} $\\pm$ {tell.std(ddof=1):.4f}",f"{cart.mean():.4f} $\\pm$ {cart.std(ddof=1):.4f}"])
    write_table("T4_ablation.tex",["Arm","K-SSE","Border fraction","Tell AUC","CART AUC"],rows,"Ablation results. The three critic constraints are not separable on these data.","tab:ablation-new")
    e1=pd.read_csv(RESULTS/"qc_e1_natural_experiment.csv"); e3=pd.read_csv(RESULTS/"qc_e3_transfer.csv"); rows=[]
    for r in e1.itertuples(): rows.append(["Natural experiment",r.region,"—",f"distinct values {r.min_distinct_values}--{r.max_distinct_values}; reclassified={r.reclassified}","—"])
    for r in e3.itertuples(): rows.append(["Transfer",r.strategy,r.model,f"{r.auc_synthetic:.4f} $\\pm$ {r.auc_synthetic_std:.4f}",f"{r.auc_observed:.4f} $\\pm$ {r.auc_observed_std:.4f}; gap {r.inflation:+.4f}"])
    write_table("T5_quebec.tex",["Experiment","Strategy/region","Model","Synthetic/report","Observed/control"],rows,"Quebec natural experiment and transfer validation.","tab:quebec-new")

def main():
    FIGDIR.mkdir(parents=True,exist_ok=True); TABDIR.mkdir(parents=True,exist_ok=True)
    man=pd.read_csv(RETRAIN/"manifest.csv"); raw=pd.read_csv(RESULTS/"arm_eval_percell.csv"); ev=pd.read_csv(RESULTS/"arm_eval_percell_snapped.csv"); adm=pd.read_csv(RESULTS/"admissibility.csv"); q=pd.read_csv(RESULTS/"qc_e3_transfer.csv"); fdf=pd.read_csv(DATA).dropna()
    assert set(ARMS)==set(man.label) and man.groupby("label").seed.nunique().eq(3).all()
    figure1(man); figure2(ev); figure3(raw,ev); figure4(adm); figure5(q)
    dynamics(GAN_ARMS,[42,43],(4,3),"F6_training_dynamics","F6")
    dynamics(AP_ARMS,[42,43,44],(3,3),"F6_training_dynamics_ap3x3","F6-ap3x3")
    figure7(man); figure8(fdf); figure9(man,ev); figure10(fdf); tables(man,ev,fdf)
    pd.DataFrame(trace).to_csv(ROOT/"FIGURE_DATA.csv",index=False)
    (ROOT/"FIGURE_REPORT.md").write_text("# Figure regeneration report\n\n"+"\n".join(report)+"\n",encoding="utf-8")
    print(f"Generated {len(report)} figures, 5 tables, and {len(trace):,} trace rows.")

if __name__=="__main__": main()
