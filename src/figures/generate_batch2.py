"""Generate AP-WGAN revision figures F11--F20 and append provenance/report."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; RETRAIN=ROOT/'Revision Code'/'retrain'; RESULTS=ROOT/'Revision Code'/'results'
DATA=ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv'; FIGDIR=ROOT/'Overleaf 2'/'figures'
sys.path[:0]=[str(HERE),str(ROOT/'Revision Code')]
from figstyle import save_figure, SINGLE, DOUBLE
from construction_audit import geometry_meta
import train_variants as tv

RNG=np.random.default_rng(42); ARMS=['random','heuristic','clip','gp_standard','v1_da_gp','dagp_strict','dagp_base','dagp_relaxed']
LABEL={'random':'Random','heuristic':'Heuristic','clip':'Clip-WGAN','gp_standard':'GP-standard','v1_da_gp':'DA-GP-WGAN','dagp_strict':'AP-WGAN (strict)','dagp_base':'AP-WGAN (base)','dagp_relaxed':'AP-WGAN (relaxed)'}
COL={'random':'#7f7f7f','heuristic':'#E69F00','clip':'#56B4E9','gp_standard':'#009E73','v1_da_gp':'#CC79A7','dagp_strict':'#0072B2','dagp_base':'#D55E00','dagp_relaxed':'#F0E442'}
LS={a:s for a,s in zip(ARMS,['-','--','-.',':','-','--','-.',':'])}; MODELS=['LogReg','NaiveBayes','CART','KNN','SVM']
trace=[]; report=[]
def rel(p):
    try:return str(Path(p).relative_to(ROOT))
    except:return str(p)
def tr(fid,src,col,v,arm='',seed='',series=''):
    if np.isfinite(float(v)): trace.append(dict(figure=fid,source_file=rel(src),source_column=col,arm=arm,seed=seed,series=series,value=float(v)))
def rec(fid,src,claim,width,axis='natural baselines used',qc='PASS'): report.append(f"- **{fid}** — sources: {src}; claim: {claim}; target: {width}; axes: {axis}; QC: {qc}.")
def save(fig,name): w=save_figure(fig,name,FIGDIR); plt.close(fig); return w
def pts(a,s=42):
    r=MAN[(MAN.label==a)&(MAN.seed==s)].iloc[0]; return np.load(ROOT/'Revision Code'/str(r.pts_file).replace('\\','/'))
def featfile(a,s=42):
    fs=list((RETRAIN/f'{a}_seed{s}').glob('feats_*.npy')); assert len(fs)==1; return fs[0]
def diagfile(a,s):
    fs=list((RETRAIN/f'{a}_seed{s}').glob('diag_*.csv')); assert len(fs)==1; return fs[0]
def gridvar(p):
    xe=np.linspace(FIREXY[:,0].min()-.2,FIREXY[:,0].max()+.2,11); ye=np.linspace(FIREXY[:,1].min()-.2,FIREXY[:,1].max()+.2,11)
    h=np.histogram2d(p[:,0],p[:,1],bins=[xe,ye])[0]; return h,float(h.var())

def f11():
    sets=[('Fire reference',FIREXY,DATA)]+[(LABEL[a],pts(a),RETRAIN/f'{a}_seed42') for a in ARMS]; hs=[gridvar(x[1])[0] for x in sets]; vmax=max(h.max() for h in hs)
    fig,axs=plt.subplots(3,3,figsize=(DOUBLE,6.7),sharex=True,sharey=True); im=None
    for ax,(name,p,src),h in zip(axs.flat,sets,hs):
        im=ax.imshow(h.T,origin='lower',cmap='viridis',vmin=0,vmax=vmax,aspect='auto',extent=[FIREXY[:,0].min()-.2,FIREXY[:,0].max()+.2,FIREXY[:,1].min()-.2,FIREXY[:,1].max()+.2]); gv=h.var(); ax.text(.03,.95,f'{name}\nvariance = {gv:.1f}',transform=ax.transAxes,va='top',color='white',fontsize=7,weight='bold')
        for v in h.ravel():tr('F11',src,'10x10_grid_count',v,series=name)
        tr('F11',src,'grid_variance',gv,series=name)
    for i,ax in enumerate(axs.flat): ax.set_xlabel('longitude (°E)' if i//3==2 else ''); ax.set_ylabel('latitude (°N)' if i%3==0 else '')
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8); cb.set_label('points per grid cell (count)'); fig.subplots_adjust(left=.08,right=.88,bottom=.07,top=.98,wspace=.08,hspace=.12)
    w=save(fig,'F11_point_density_grid'); rec('F11','fire CSV; retrain seed-42 point arrays','spatial density and AP-WGAN strict border advantage','double column (7.16 in)',qc=w or 'PASS')

def f12():
    mu=FIREF.mean(0); sd=np.where(FIREF.std(0)==0,1,FIREF.std(0)); X={a:np.load(featfile(a)) for a in ARMS}; score=np.sum([np.abs((z.mean(0)-mu)/sd) for z in X.values()],axis=0); top=np.argsort(score)[-12:][::-1]
    fig,axs=plt.subplots(4,3,figsize=(7.16,9.5))
    for ax,j in zip(axs.flat,top):
        lo=min([FIREF[:,j].min()]+[z[:,j].min() for z in X.values()]); hi=max([FIREF[:,j].max()]+[z[:,j].max() for z in X.values()]); bins=np.linspace(lo,hi,31)
        ax.hist(FIREF[:,j],bins,density=True,color='.7',alpha=.55,label='Fire')
        for a,z in X.items():
            h,_=np.histogram(z[:,j],bins,density=True); ax.hist(z[:,j],bins,density=True,histtype='step',lw=.8,color=COL[a],ls=LS[a],label=LABEL[a])
            for k,v in enumerate(h): tr('F12',featfile(a),f'{FC[j]}_density_bin{k}',v,a,42,FC[j])
        ax.set_xlabel(f'{FC[j]} (source units; D={score[j]:.2f})'); ax.set_ylabel('probability density (1/source unit)'); ax.grid(axis='y')
    handles=[Line2D([],[],color='.5',lw=5,alpha=.5,label='Fire')]+[Line2D([],[],color=COL[a],ls=LS[a],label=LABEL[a]) for a in ARMS]
    fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,.995)); fig.tight_layout(rect=[0,0,1,.94]); w=save(fig,'F12_marginal_features_top12'); rec('F12','fire CSV; retrain seed-42 feature arrays','honest marginal feature separation ranked at runtime','full page',qc=w or 'PASS')

def f13():
    fig,ax=plt.subplots(figsize=(SINGLE,3.0)); tree=cKDTree(FIREXY); bins=np.geomspace(.02,120,65)
    for a in ARMS:
        d=tree.query(pts(a))[0]*111; h,_=np.histogram(d,bins,density=True); ax.hist(d,bins,density=True,histtype='step',color=COL[a],ls=LS[a],lw=1.1,label=LABEL[a]); [tr('F13',RETRAIN/f'{a}_seed42',f'nearest_fire_density_bin{k}',v,a,42) for k,v in enumerate(h)]
    ax.axvline(10,color='black',ls='--',lw=.9,label='10 km buffer'); ax.set_xscale('log'); ax.set_xlabel('distance to nearest fire record (km, log scale)'); ax.set_ylabel('probability density (km⁻¹)'); ax.legend(fontsize=6,ncol=2); ax.grid(); fig.tight_layout(); w=save(fig,'F13_nearest_fire_distance'); rec('F13','fire CSV; retrain seed-42 point arrays','buffer compliance and Random near-fire mass','single column (3.5 in)','log-x spans full positive support',w or 'PASS')

def f14():
    fig,axs=plt.subplots(2,1,figsize=(DOUBLE,5.0));
    for ax,j,n,u in [(axs[0],0,'longitude','°E'),(axs[1],1,'latitude','°N')]:
        vals=[FIREXY[:,j]]+[pts(a)[:,j] for a in ARMS]; bins=np.linspace(min(v.min() for v in vals),max(v.max() for v in vals),45); ax.hist(FIREXY[:,j],bins,density=True,color='.7',alpha=.55,label='Fire')
        for a in ARMS:
            z=pts(a)[:,j]; h,_=np.histogram(z,bins,density=True); ax.hist(z,bins,density=True,histtype='step',color=COL[a],ls=LS[a],label=LABEL[a]); [tr('F14',RETRAIN/f'{a}_seed42',f'{n}_density_bin{k}',v,a,42) for k,v in enumerate(h)]
        ax.set_xlabel(f'{n} ({u})'); ax.set_ylabel(f'probability density (1/{u})'); ax.grid(axis='y')
    axs[0].legend(ncol=3,fontsize=7); fig.tight_layout(); w=save(fig,'F14_coordinate_marginals'); rec('F14','fire CSV; retrain seed-42 point arrays','Heuristic edge concentration and Random profile','double column (7.16 in)',qc=w or 'PASS')

def heat(ax,metric,title):
    g=EV.groupby(['arm','model'])[metric].agg(['mean','std']); M=np.array([[g.loc[(a,m),'mean'] for m in MODELS] for a in ARMS]); S=np.array([[g.loc[(a,m),'std'] for m in MODELS] for a in ARMS]); norm=TwoSlopeNorm(vmin=M.min(),vcenter=M.mean(),vmax=M.max()); im=ax.imshow(M,cmap='coolwarm',norm=norm,aspect='auto')
    for j,m in enumerate(MODELS):
        bi,wi=np.argmax(M[:,j]),np.argmin(M[:,j])
        for i,a in enumerate(ARMS): ax.text(j,i,f'{M[i,j]:.3f}{"▲" if i==bi else ("▼" if i==wi else "")}\n{S[i,j]:.3f}',ha='center',va='center',fontsize=5.8); [tr('F15',RESULTS/'arm_eval_percell_snapped.csv',metric,r[metric],a,r.seed,m) for _,r in EV[(EV.arm==a)&(EV.model==m)].iterrows()]
    ax.set_xticks(range(5),MODELS,rotation=30,ha='right'); ax.set_yticks(range(8),[LABEL[a] for a in ARMS]); ax.set_xlabel(f'classifier — {title}'); ax.set_ylabel('strategy'); return im
def f15():
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE,4.0)); ims=[heat(ax,m,m) for ax,m in zip(axs,['AUC','TSS'])]
    for ax,im,m in zip(axs,ims,['AUC','TSS']): fig.colorbar(im,ax=ax,shrink=.75,label=f'{m} (unitless)')
    fig.tight_layout(); w=save(fig,'F15_auc_tss_heatmap'); rec('F15','results/arm_eval_percell_snapped.csv','like-for-like AUC and TSS without unsupported ranking','double column (7.16 in)',qc=w or 'PASS')

def f16():
    P=pd.read_csv(RESULTS/'arm_eval_probs_calibration.csv'); fig,axs=plt.subplots(4,2,figsize=(DOUBLE,8.5),sharex=True,sharey=True)
    for ax,a in zip(axs.flat,ARMS):
        ax.plot([0,1],[0,1],color='.4',ls=':',lw=.8)
        txt=[]
        for model,c,mk in [('LogReg','#0072B2','o'),('SVM','#D55E00','s')]:
            z=P[(P.arm==a)&(P.model==model)]; frac,pred=calibration_curve(z.y_true,z.y_prob,n_bins=10,strategy='uniform'); ax.plot(pred,frac,color=c,marker=mk,ms=3,label=model)
            bs=brier_score_loss(z.y_true,z.y_prob); bins=np.minimum((z.y_prob*10).astype(int),9); ece=sum((bins==b).mean()*abs(z.y_true[bins==b].mean()-z.y_prob[bins==b].mean()) for b in np.unique(bins)); txt.append(f'{model}: Brier {bs:.3f}, ECE {ece:.3f}')
            for r in z.itertuples(): tr('F16',RESULTS/'arm_eval_probs_calibration.csv','y_prob',r.y_prob,a,r.seed,model)
        ax.text(.03,.97,LABEL[a]+'\n'+'\n'.join(txt),transform=ax.transAxes,va='top',fontsize=6.5); ax.set_xlabel('predicted fire probability'); ax.set_ylabel('observed fire frequency'); ax.grid()
    axs[0,0].legend(loc='lower right'); fig.tight_layout(); w=save(fig,'F16_calibration_curves'); rec('F16','results/arm_eval_probs_calibration.csv','calibration of the permitted main-suite probabilistic models','double column (7.16 in)',qc=w or 'PASS')

CON=[('dagp_base','random'),('dagp_base','heuristic'),('dagp_base','clip'),('dagp_base','gp_standard'),('dagp_base','v1_da_gp'),('dagp_strict','dagp_relaxed')]
def boots(model,a,b,n=2000):
    x=EV[(EV.model==model)&(EV.arm==a)].AUC.to_numpy(); y=EV[(EV.model==model)&(EV.arm==b)].AUC.to_numpy(); d=np.array([RNG.choice(x,len(x)).mean()-RNG.choice(y,len(y)).mean() for _ in range(n)]); return d
def f17():
    fig,axs=plt.subplots(2,3,figsize=(DOUBLE,5.0))
    for ax,(a,b) in zip(axs.flat,CON):
        d=boots('CART',a,b); lo,hi=np.percentile(d,[2.5,97.5]); ax.hist(d,bins=35,color='#56B4E9',edgecolor='white'); ax.axvspan(lo,hi,color='#E69F00',alpha=.2); ax.axvline(0,color='black',ls='--'); ns=lo<=0<=hi; ax.text(.04,.94,'n.s.' if ns else 'significant',transform=ax.transAxes,va='top',weight='bold'); ax.set_xlabel(f'{LABEL[a]} − {LABEL[b]}\nbootstrap ΔAUC (unitless)'); ax.set_ylabel('replicates (count)'); [tr('F17','derived from arm_eval_percell_snapped.csv','bootstrap_delta_AUC',v,series=f'{a}-{b}') for v in d]
    fig.tight_layout(); w=save(fig,'F17_bootstrap_delta_auc'); rec('F17','results/arm_eval_percell_snapped.csv','CART contrasts including null uncertainty','double column (7.16 in)','x ranges show complete bootstrap support',w or 'PASS')
def f18():
    fig,ax=plt.subplots(figsize=(DOUBLE,6.0)); offsets=np.linspace(-.28,.28,5); y=np.arange(len(CON));
    for j,m in enumerate(MODELS):
        for i,(a,b) in enumerate(CON):
            d=boots(m,a,b); est=EV[(EV.model==m)&(EV.arm==a)].AUC.mean()-EV[(EV.model==m)&(EV.arm==b)].AUC.mean(); lo,hi=np.percentile(d,[2.5,97.5]); sig=not(lo<=0<=hi); ax.errorbar(est,i+offsets[j],xerr=[[est-lo],[hi-est]],fmt='o',mfc=plt.cm.tab10(j) if sig else 'white',mec=plt.cm.tab10(j),ms=4,c=plt.cm.tab10(j),label=m if i==0 else None); tr('F18',RESULTS/'arm_eval_percell_snapped.csv','delta_AUC',est,series=f'{m}:{a}-{b}')
    ax.axvline(0,color='black',ls='--'); ax.set_yticks(y,[f'{LABEL[a]} − {LABEL[b]}' for a,b in CON],fontsize=9.5); ax.invert_yaxis(); ax.set_xlabel('ΔAUC with 95% bootstrap CI (unitless)'); ax.set_ylabel('contrast'); handles,labels=ax.get_legend_handles_labels(); handles += [Line2D([],[],marker='o',mfc='black',mec='black',ls='',label='CI excludes zero'),Line2D([],[],marker='o',mfc='white',mec='black',ls='',label='CI includes zero')]; labels += ['CI excludes zero','CI includes zero']; ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=len(handles),frameon=False,fontsize=7); ax.grid(axis='x'); fig.subplots_adjust(left=.28,right=.98,bottom=.1,top=.84); w=save(fig,'F18_bootstrap_forest'); rec('F18','results/arm_eval_percell_snapped.csv','all requested contrasts retained across five classifiers; filled marker denotes CI excluding zero','double column (7.16 in)','x ranges show all CIs; labels unclipped',w or 'PASS')

def f19():
    bgpts,_=tv.build_background(FIREXY,seed=0); bg=tv.idw_feats(bgpts,cKDTree(FIREXY),FIREF,seed=99); sc=StandardScaler().fit(bg); bgm=sc.transform(bg).mean(0)
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.5)); ax=axs[0]
    for a in [x for x in ARMS if x not in ['random','heuristic']]:
        lines=[]
        for s in [42,43,44]:
            d=pd.read_csv(diagfile(a,s)); w=max(1,len(d)//400); W=-d.critic_loss if a=='clip' else 10*d.gp-d.critic_loss; W=W.rolling(w,center=True,min_periods=1).mean(); step=max(1,len(W)//400); ax.plot(d.iter.iloc[::step],W.iloc[::step],color=COL[a],alpha=.25,lw=.5); lines.append(np.interp(np.arange(1,4501),d.iter,W)); [tr('F19',diagfile(a,s),'W_estimate',v,a,s,'clip: -critic_loss' if a=='clip' else 'GP: 10*gp-critic_loss') for v in W.dropna()]
        ax.plot(np.arange(1,4501),np.mean(lines,axis=0),color=COL[a],lw=1.4,label=LABEL[a])
    ax.set_xlabel('iteration (count)'); ax.set_ylabel('Wasserstein estimate (unitless; rolling window = 11)'); ax.grid(); ax.legend(fontsize=6,ncol=2); ax.text(.02,.02,'Clip: W = −critic loss\nGP arms: W = 10·gp − critic loss',transform=ax.transAxes,fontsize=7)
    ax=axs[1]
    for i,a in enumerate(ARMS):
        vals=[]
        for s in [42,43,44]: z=sc.transform(np.load(featfile(a,s))); v=np.mean(np.abs(z.mean(0)-bgm)); vals.append(v); tr('F19',featfile(a,s),'mean_abs_standardized_feature_bias',v,a,s)
        ax.scatter(i+np.array([-.1,0,.1]),vals,c=COL[a],marker='o',s=18); ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='_',c='black',capsize=2)
    ax.set_xticks(range(8),[LABEL[a] for a in ARMS],rotation=40,ha='right'); ax.set_xlabel('strategy (points = seeds; bar = mean ± SD)'); ax.set_ylabel('mean absolute standardized feature bias (SD units)'); ax.grid(axis='y')
    fig.tight_layout(); w=save(fig,'F19_training_wasserstein_bias'); rec('F19','diagnostic CSVs; deterministic background; feature arrays','explicit Wasserstein branches and seed-spread feature bias','double column (7.16 in)',qc=w or 'PASS')

def f20():
    fig,axs=plt.subplots(4,2,figsize=(DOUBLE,8.5)); bins=np.geomspace(.01,30,55)
    for idx,(ax,a) in enumerate(zip(axs.flat,ARMS)):
        ratios=[]
        for s,alpha in zip([42,43,44],[.35,.55,.8]):
            pa=np.load(featfile(a,s)); M=geometry_meta(np.vstack([FIREF,pa])); f,p=M[:len(FIREF),0],M[len(FIREF):,0]; ratio=np.median(f)/np.median(p); ratios.append(ratio); ax.hist(f,bins,density=True,histtype='step',color='black',alpha=alpha); ax.hist(p,bins,density=True,histtype='step',color=COL[a],alpha=alpha); hp,_=np.histogram(p,bins,density=True); [tr('F20',featfile(a,s),f'geometry_m1_density_bin{k}',v,a,s,'PA') for k,v in enumerate(hp)]; tr('F20',featfile(a,s),'fire_to_pa_median_ratio',ratio,a,s)
        ax.set_xscale('log'); ax.text(.03,.95,f'{LABEL[a]}\nfire:PA median = {np.mean(ratios):.2f}×',transform=ax.transAxes,va='top',fontsize=7,weight='bold'); ax.set_xlabel('7-NN centroid distance\n(standardized units, log)' if idx//2==3 else ''); ax.set_ylabel('density (1/unit)' if idx%2==0 else ''); ax.grid()
    fig.legend([Line2D([],[],c='black'),Line2D([],[],c='#0072B2')],['Fire','PA (strategy colour)'],loc='upper center',bbox_to_anchor=(.5,.995),ncol=2); fig.tight_layout(rect=[0,0,1,.95]); w=save(fig,'F20_local_geometry_fingerprint'); rec('F20','fire CSV; all retrain feature arrays','local IDW geometry diagnostic using authoritative implementation','double column (7.16 in)','log-x spans full positive support',w or 'PASS')

def main():
    global MAN,EV,FDF,FIREXY,FC,FIREF
    MAN=pd.read_csv(RETRAIN/'manifest.csv'); EV=pd.read_csv(RESULTS/'arm_eval_percell_snapped.csv'); FDF=pd.read_csv(DATA).dropna(); FIREXY=FDF[['LONGITUDE','LATITUDE']].to_numpy(); FC=[c for c in FDF if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]; FIREF=FDF[FC].to_numpy(float)
    f11(); f12(); f13(); f14(); f15(); f16(); f17(); f18(); f19(); f20()
    old=pd.read_csv(ROOT/'FIGURE_DATA.csv'); pd.concat([old,pd.DataFrame(trace)],ignore_index=True).to_csv(ROOT/'FIGURE_DATA.csv',index=False)
    with open(ROOT/'FIGURE_REPORT.md','a',encoding='utf-8') as f: f.write('\n## Second batch\n\n'+'\n'.join(report)+'\n')
    print(f'Generated F11--F20 and appended {len(trace):,} trace rows.')
if __name__=='__main__': main()
