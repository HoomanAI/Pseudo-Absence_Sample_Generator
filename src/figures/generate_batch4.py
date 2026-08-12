"""Round-4 regional and structural figures; idempotent provenance output."""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent; CODE=HERE.parent; ROOT=CODE.parent
RESULTS=CODE/'results'; RETRAIN=CODE/'retrain'; OUT=ROOT/'Overleaf 2'/'figures'; PROV=ROOT/'FIGURE_DATA.csv'
sys.path.insert(0,str(HERE)); from figstyle import save_figure, DOUBLE
ARMS=['random','heuristic','clip','gp_standard','v1_da_gp','dagp_strict','dagp_base','dagp_relaxed']
LABEL={'random':'Random','heuristic':'Heuristic','clip':'GAN (clip)','gp_standard':'WGAN-GP','v1_da_gp':'DA-GP-WGAN','dagp_strict':'AP-WGAN (strict)','dagp_base':'AP-WGAN (base)','dagp_relaxed':'AP-WGAN (relaxed)'}
COLOR=dict(zip(ARMS,['#7F7F7F','#8C564B','#D55E00','#CC79A7','#009E73','#56B4E9','#0072B2','#7BAFD4']))
MARK=dict(zip(ARMS,['o','s','^','v','D','P','X','*'])); RECORDS=[]; WARN={}
FIRECSV=ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv'; QPTS=RESULTS/'quebec_points'
def rec(fig,src,col,arm='',seed='',series='',value=np.nan): RECORDS.append([fig,str(src).replace('\\','/'),col,arm,seed,series,float(value) if pd.notna(value) else ''])
def panel(ax,s): ax.text(-.14,1.06,s,transform=ax.transAxes,weight='bold',ha='right',va='bottom',clip_on=False)
def finish(fig,stem): WARN[stem]=save_figure(fig,stem,OUT); plt.close(fig)
def border(ax,b,deg=.15):
    x0,x1,y0,y1=b; kw=dict(fc='#E8A33D',alpha=.10,ec='none',zorder=0)
    ax.add_patch(Rectangle((x0,y0),x1-x0,deg,**kw)); ax.add_patch(Rectangle((x0,y1-deg),x1-x0,deg,**kw)); ax.add_patch(Rectangle((x0,y0+deg),deg,y1-y0-2*deg,**kw)); ax.add_patch(Rectangle((x1-deg,y0+deg),deg,y1-y0-2*deg,**kw))
def f26():
    sys.path.insert(0,str(CODE)); import train_variants as tv
    fire=pd.read_csv(FIRECSV).dropna()[['LONGITUDE','LATITUDE']].to_numpy(); man=pd.read_csv(RETRAIN/'manifest.csv').query('seed == 42').set_index('label'); b=(fire[:,0].min()-.2,fire[:,0].max()+.2,fire[:,1].min()-.2,fire[:,1].max()+.2)
    gx=np.linspace(b[0],b[1],220); gy=np.linspace(b[2],b[3],180); X,Y=np.meshgrid(gx,gy); D=cKDTree(fire).query(np.c_[X.ravel(),Y.ravel()])[0].reshape(X.shape)
    fig,axs=plt.subplots(4,2,figsize=(DOUBLE,1.35*DOUBLE),sharex=True,sharey=True)
    for k,(arm,ax) in enumerate(zip(ARMS,axs.flat)):
        r=man.loc[arm]; p=ROOT/'Revision Code'/Path(str(r.pts_file).replace('retrain\\','retrain/')); pts=np.load(p)
        border(ax,b); ax.contour(X,Y,D,levels=[10/111],colors=['.5'],linewidths=.75,alpha=.65)
        for cx,cy,sx,sy in tv.WATER_ELLIPSES: ax.add_patch(Ellipse((cx,cy),2*sx,2*sy,fc='none',ec='.4',hatch='////',lw=.75,alpha=.65))
        ax.scatter(fire[:,0],fire[:,1],s=3,c='.65',alpha=.25,rasterized=True); ax.scatter(pts[:,0],pts[:,1],s=5,c=COLOR[arm],alpha=.55,rasterized=True)
        ax.text(.02,.98,LABEL[arm],transform=ax.transAxes,ha='left',va='top',fontsize=7,weight='bold',bbox=dict(fc='white',alpha=.72,ec='none')); ax.text(.98,.98,f"K-SSE {r.K_SSE_rawK:.3f}\nBorder {r.border_fraction:.3f}",transform=ax.transAxes,ha='right',va='top',fontsize=7,bbox=dict(fc='white',alpha=.65,ec='none')); ax.set_aspect('equal'); ax.set_xlim(b[:2]); ax.set_ylim(b[2:]); panel(ax,chr(97+k))
        rec('F26','Revision Code/retrain/manifest.csv','K_SSE_rawK',arm,42,'annotation',r.K_SSE_rawK); rec('F26','Revision Code/retrain/manifest.csv','border_fraction',arm,42,'annotation',r.border_fraction)
        for i,(xx,yy) in enumerate(pts): rec('F26',r.pts_file,'LONGITUDE',arm,42,f'point={i}',xx); rec('F26',r.pts_file,'LATITUDE',arm,42,f'point={i}',yy)
    fig.text(.5,.015,'Longitude (°)',ha='center'); fig.text(.01,.5,'Latitude (°)',rotation=90,va='center'); fig.text(.99,.015,'Alberta: n = 3,370 presences / 5,500 pseudo-absences',ha='right',fontsize=7); fig.subplots_adjust(left=.1,right=.99,bottom=.06,top=.99,wspace=.12,hspace=.16); finish(fig,'F26_map_alberta')
def f27():
    fire=np.load(QPTS/'qc_pts_fire.npy'); sets=[('Observed (oracle)',np.load(QPTS/'qc_pts_observed_absence.npy'),'#0072B2'),('Random',np.load(QPTS/'qc_pts_random.npy'),COLOR['random']),('Heuristic',np.load(QPTS/'qc_pts_heuristic.npy'),COLOR['heuristic'])]; meta=json.loads((QPTS/'qc_bounds.json').read_text()); b=(meta['lo_x'],meta['hi_x'],meta['lo_y'],meta['hi_y']); e2=pd.read_csv(RESULTS/'qc_e2_fidelity.csv').set_index('strategy')
    fig,axs=plt.subplots(3,1,figsize=(DOUBLE,1.4*DOUBLE),sharex=True,sharey=False)
    for k,(name,pts,c) in enumerate(sets):
        ax=axs[k]; border(ax,b); ax.scatter(fire[:,0],fire[:,1],s=7,c='.65',alpha=.28); ax.scatter(pts[:,0],pts[:,1],s=10,c=c,alpha=.62); r=e2.loc[name]; ax.text(.02,.98,name,transform=ax.transAxes,ha='left',va='top',fontsize=8,weight='bold',bbox=dict(fc='white',alpha=.72,ec='none')); ax.text(.98,.98,f"K-SSE {r.K_SSE_vs_observed_absence:.3f}\nBorder {r.border_frac:.3f}",transform=ax.transAxes,ha='right',va='top',fontsize=7,bbox=dict(fc='white',alpha=.7,ec='none')); ax.set_aspect('equal'); panel(ax,chr(97+k));
        if k==0:
            for sp in ax.spines.values(): sp.set_linewidth(1.8)
        rec('F27','Revision Code/results/qc_e2_fidelity.csv','K_SSE_vs_observed_absence',name,'','annotation',r.K_SSE_vs_observed_absence); rec('F27','Revision Code/results/qc_e2_fidelity.csv','border_frac',name,'','annotation',r.border_frac)
        src={'Observed (oracle)':'qc_pts_observed_absence.npy','Random':'qc_pts_random.npy','Heuristic':'qc_pts_heuristic.npy'}[name]
        for i,(xx,yy) in enumerate(pts): rec('F27',f'Revision Code/results/quebec_points/{src}','LONGITUDE',name,'',f'point={i}',xx); rec('F27',f'Revision Code/results/quebec_points/{src}','LATITUDE',name,'',f'point={i}',yy)
        if k<2: ax.tick_params(labelbottom=False)
    fig.text(.5,.035,'Longitude (°)',ha='center'); fig.text(.025,.5,'Latitude (°)',rotation=90,va='center'); fig.text(.99,.008,'Quebec: n = 521 presences / 521 observed absences / 1,042 records',ha='right',fontsize=7); fig.subplots_adjust(left=.13,right=.98,bottom=.08,top=.99,hspace=.12); finish(fig,'F27_map_quebec')
def f28():
    df=pd.read_csv(FIRECSV).dropna(); meta=['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']; fc=[c for c in df if c not in meta]; disc=[c for c in fc if df[c].nunique()<=12]; assert len(disc)==13
    man=pd.read_csv(RETRAIN/'manifest.csv').query('seed==42').set_index('label'); clip=np.load(RETRAIN/'clip_seed42'/'feats_clip.npy'); ap=np.load(RETRAIN/'dagp_base_seed42'/'feats_v2_admissible.npy')
    tex=(ROOT/'Overleaf 2'/'sections'/'03_data.tex').read_text(); table_weights=[float(x) for x in re.findall(r'&\s*[59]\s*&\s*([0-9.]+)\s*&',tex)][:13]
    fig,axs=plt.subplots(13,1,figsize=(DOUBLE,8.8)); rng=np.random.default_rng(42)
    pretty=['Slope','Elevation','Aspect','Plan curvature','Profile curvature','Valley depth','TWI','NDVI','Mean temperature','Mean precipitation','Mean wind speed','River distance','Road distance']
    for i,(c,ax) in enumerate(zip(disc,axs)):
        j=fc.index(c); levels=np.unique(df[c]); w=float(np.median(np.diff(levels))); assert abs(w-table_weights[i])<1e-6,(c,w,table_weights[i])
        bins=np.linspace(levels.min(),levels.max(),100); hist,edges=np.histogram(clip[:,j],bins=bins,density=True); cen=(edges[:-1]+edges[1:])/2; hist=hist/max(hist.max(),1e-12); ax.fill_between(cen,0,hist*.6,color=COLOR['clip'],alpha=.22)
        for vals,color,y in [(df[c].to_numpy(),'.25',.78),(ap[:,j],COLOR['dagp_base'],.66)]:
            take=rng.choice(len(vals),min(450,len(vals)),replace=False); ax.vlines(vals[take],y,y+.16,color=color,alpha=.18,lw=.35)
        for lv in levels: ax.axvline(lv,color='.75',lw=.45)
        ax.set_ylim(0,1); ax.set_yticks([]); ax.set_ylabel(pretty[i],rotation=0,ha='right',va='center',fontsize=7); ax.text(1.01,.5,f'w={w:.6f}',transform=ax.transAxes,va='center',family='monospace',fontsize=7); 
        if c=='river': ax.text(.82,.12,r'$m_j=9$',transform=ax.transAxes,fontsize=7)
        if i<12: ax.set_xticklabels([])
        for obs,v in enumerate(clip[:,j]): rec('F28','Revision Code/retrain/clip_seed42/feats_clip.npy',c,'clip',42,f'observation={obs}',v)
        for obs,v in enumerate(ap[:,j]): rec('F28','Revision Code/retrain/dagp_base_seed42/feats_v2_admissible.npy',c,'dagp_base',42,f'observation={obs}',v)
        for obs,v in enumerate(df[c]): rec('F28',str(FIRECSV.relative_to(ROOT)),c,'presence','',f'observation={obs}',v)
    axs[0].text(.01,.92,'Presence rug',transform=axs[0].transAxes,fontsize=6,color='.25'); axs[0].text(.20,.92,'GAN (clip) density',transform=axs[0].transAxes,fontsize=6,color=COLOR['clip']); axs[0].text(.47,.92,'AP-WGAN rug',transform=axs[0].transAxes,fontsize=6,color=COLOR['dagp_base'])
    axs[-1].set_xlabel('Reclassified covariate value'); fig.subplots_adjust(left=.22,right=.86,bottom=.05,top=.99,hspace=.06); finish(fig,'F28_level_set_lattice')

def f29():
    sys.path.insert(0,str(CODE)); import recompute_spatial as rs
    fire=pd.read_csv(FIRECSV).dropna()[['LONGITUDE','LATITUDE']].to_numpy(); area=np.ptp(fire[:,0])*np.ptp(fire[:,1]); ref=rs.k_curve(fire,area); man=pd.read_csv(RETRAIN/'manifest.csv'); fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.0)); r=rs.R_VALS
    axs[0].plot(r,ref,color='black',lw=1.5,label='Presence reference')
    for arm in ARMS:
        curves=[]
        for rr in man[man.label==arm].itertuples():
            p=ROOT/'Revision Code'/Path(str(rr.pts_file).replace('retrain\\','retrain/')); c=rs.k_curve(np.load(p),area); curves.append(c)
            for rad,v in zip(r,c): rec('F29',rr.pts_file,'K(r)',arm,rr.seed,f'r={rad:.6f}',v)
        A=np.array(curves); m=A.mean(0); sd=A.std(0); dev=((A-ref)**2); dm=dev.mean(0); ds=dev.std(0)
        axs[0].plot(r,m,color=COLOR[arm],marker=MARK[arm],mfc='white',ms=3,lw=.9,label=LABEL[arm]); axs[0].fill_between(r,m-sd,m+sd,color=COLOR[arm],alpha=.15)
        axs[1].plot(r,dm,color=COLOR[arm],marker=MARK[arm],mfc='white',ms=3,lw=.9); axs[1].fill_between(r,np.maximum(0,dm-ds),dm+ds,color=COLOR[arm],alpha=.15)
    axs[0].set(xlabel='Radius (degrees)',ylabel='Raw K(r)'); axs[1].set(xlabel='Radius (degrees)',ylabel='Squared deviation from presence K(r)');
    for k,ax in enumerate(axs): ax.grid(alpha=.25); panel(ax,chr(97+k))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='center left',bbox_to_anchor=(.99,.5)); fig.text(.01,.01,'Alberta: n = 3,370 presences / 5,500 pseudo-absences',fontsize=7); fig.subplots_adjust(left=.1,right=.76,bottom=.2,wspace=.35); finish(fig,'F29_ripley_k_alberta')
def f30():
    fire=np.load(QPTS/'qc_pts_fire.npy'); obs=np.load(QPTS/'qc_pts_observed_absence.npy'); sets=[('Observed',obs,'#0072B2','o'),('Random',np.load(QPTS/'qc_pts_random.npy'),COLOR['random'],'o'),('Heuristic',np.load(QPTS/'qc_pts_heuristic.npy'),COLOR['heuristic'],'s')]; meta=json.loads((QPTS/'qc_bounds.json').read_text()); area=(meta['hi_x']-meta['lo_x'])*(meta['hi_y']-meta['lo_y']); r=qv.R_VALS if 'qv' in globals() else np.linspace(.05,.8,15)
    def kc(p): t=cKDTree(p); n=len(p); return (t.count_neighbors(t,r)-n)/n/(n/area)
    refs=[('Presence reference',kc(fire)),('Observed-absence reference',kc(obs))]; fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.0))
    for k,(title,ref) in enumerate(refs):
        axs[k].plot(r,ref,color='black',lw=1.5,label=title)
        for name,p,c,m in sets:
            v=kc(p); axs[k].plot(r,v,color=c,marker=m,mfc='white',ms=3,lw=1,label=name)
            for rad,x in zip(r,v): rec('F30',f'Revision Code/results/quebec_points/qc_pts_{name.lower().replace(" ","_")}.npy','K(r)',name,'',f'{title}:r={rad:.6f}',x)
        axs[k].set(xlabel=f'Radius (degrees) — {title}',ylabel='Raw K(r)'); axs[k].grid(alpha=.25); panel(axs[k],chr(97+k)); axs[k].legend(fontsize=6.5)
    fig.text(.99,.01,'Quebec: n = 521 presences / 521 observed absences / 1,042 records',ha='right',fontsize=7); fig.subplots_adjust(left=.1,right=.99,bottom=.18,wspace=.32); finish(fig,'F30_ripley_k_quebec')
def f31():
    a=pd.read_csv(FIRECSV).dropna(); meta=['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']; ac=[c for c in a if c not in meta]; an=np.array([a[c].nunique() for c in ac]); q=qv.load_quebec(str(ROOT/'Data Quebec'/'WildFire_TrainTest.xlsx')) if 'qv' in globals() else pd.concat([pd.read_excel(ROOT/'Data Quebec'/'WildFire_TrainTest.xlsx',sheet_name=s) for s in ['Train','Test']]); qn=np.array([q[c].nunique() for c in qv.FEATS]) if 'qv' in globals() else np.array([q[c].nunique() for c in ['Elevation','Slope','Aspect','Profile Curvature','Plan Curvature','Valley Depth','TWI','Distance From Rivers','Distance From Roads','NDVI','Mean Annual Precipitation','Mean Annual Temperature','Mean Annual Wind Speed']])
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.2)); rng=np.random.default_rng(4); disc=an<=12
    axs[0].scatter(an[~disc],rng.normal(.15,.018,(~disc).sum()),c='#0072B2',marker='o',label='Alberta continuous',alpha=.7); axs[0].scatter(an[disc],rng.normal(.15,.018,disc.sum()),fc='#E8A33D',ec='black',marker='o',label='Alberta reclassified'); axs[0].scatter(qn,rng.normal(.65,.018,len(qn)),fc='none',ec='#8C564B',marker='s',label='Quebec continuous'); axs[0].set_xscale('log'); axs[0].set_yticks([.15,.65],['Alberta','Quebec']); axs[0].set_xlabel('Distinct values per covariate (log scale)'); axs[0].legend(fontsize=6.5); axs[0].grid(axis='x',alpha=.25); panel(axs[0],'a')
    axs[1].plot([0,1],[1,.5],color='#555',marker='o',mfc='white',lw=1.4); axs[1].axhline(.5,color='#E8A33D',ls='--',lw=1); axs[1].text(1.86,.74,'N/A\n(no lattice)',ha='center',va='center',fontsize=8,bbox=dict(fc='.95',ec='.6')); axs[1].set(xticks=[0,1,2],xticklabels=['Alberta raw','Alberta projected','Quebec'],ylabel='Admissibility-tell AUC',ylim=(.4,1.08)); panel(axs[1],'b'); axs[1].grid(axis='y',alpha=.25)
    fig.text(.01,.01,'Alberta n = 3,370 / 5,500; Quebec n = 521 / 521 / 1,042 records',fontsize=7); fig.subplots_adjust(left=.1,right=.99,bottom=.22,wspace=.4); finish(fig,'F31_natural_experiment')
    for c,v in zip(ac,an): rec('F31',str(FIRECSV.relative_to(ROOT)),'nunique', 'Alberta','',c,v)
    for c,v in zip(range(len(qn)),qn): rec('F31','Data Quebec/WildFire_TrainTest.xlsx','nunique','Quebec','',f'covariate={c}',v)
def f33():
    df=pd.read_csv(FIRECSV).dropna(); meta=['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']; fc=[c for c in df if c not in meta]; fire=df[fc].to_numpy(float); cref=np.corrcoef(fire,rowvar=False); man=pd.read_csv(RETRAIN/'manifest.csv'); diffs={}; scores={}
    for arm in ARMS:
        mats=[]
        for rr in man[man.label==arm].itertuples():
            run=RETRAIN/f'{rr.label}_seed{rr.seed}'; A=np.load(run/f'feats_{rr.arm}.npy'); mats.append(cref-np.corrcoef(A,rowvar=False))
        diffs[arm]=np.mean(mats,axis=0); scores[arm]=float(np.linalg.norm(diffs[arm],'fro'))
    lim=max(np.percentile(np.abs(v),99) for v in diffs.values()); fig,axs=plt.subplots(4,2,figsize=(DOUBLE,1.35*DOUBLE),sharex=True,sharey=True)
    for k,(arm,ax) in enumerate(zip(ARMS,axs.flat)):
        im=ax.imshow(diffs[arm],cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True); ax.text(.02,.98,LABEL[arm],transform=ax.transAxes,ha='left',va='top',fontsize=6.5,weight='bold',bbox=dict(fc='white',alpha=.72,ec='none')); ax.text(.98,.98,f'Frobenius {scores[arm]:.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,bbox=dict(fc='white',alpha=.7,ec='none')); ax.set_xticks([0,12,29,45,57]); ax.set_yticks([0,12,29,45,57]); panel(ax,chr(97+k))
        for i in range(58):
            for j in range(58): rec('F33','presence correlation minus seed-mean arm correlation',f'{fc[i]}__{fc[j]}',arm,'','correlation_difference',diffs[arm][i,j])
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.02,pad=.02); cb.set_label('Presence − arm correlation'); fig.subplots_adjust(left=.12,right=.86,bottom=.06,top=.99,wspace=.16,hspace=.20); finish(fig,'F33_joint_structure')
    (RESULTS/'f33_joint_structure_scores.csv').write_text(pd.Series(scores,name='frobenius_norm').rename_axis('arm').to_csv(),encoding='utf-8')
def f32():
    raw=pd.read_csv(RESULTS/'arm_eval_summary.csv'); snap=pd.read_csv(RESULTS/'arm_eval_summary_snapped.csv'); q=pd.read_csv(RESULTS/'qc_e3_transfer.csv'); model='SVM'
    fig,ax=plt.subplots(figsize=(DOUBLE,3.8)); x=np.arange(4); names=['Alberta raw','After projection','Quebec observed','Observed ceiling']
    for arm in ARMS:
        a=float(raw[(raw.arm==arm)&(raw.model==model)].AUC_mean.iloc[0]); b=float(snap[(snap.arm==arm)&(snap.model==model)].AUC_mean.iloc[0]);
        ax.plot(x[:2],[a,b],color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1.1,label=LABEL[arm]); ax.plot(x[1:3],[b,np.nan],color=COLOR[arm],ls=':',lw=.7)
        rec('F32','Revision Code/results/arm_eval_summary.csv','AUC_mean',arm,'',model,a); rec('F32','Revision Code/results/arm_eval_summary_snapped.csv','AUC_mean',arm,'',model,b)
    for arm,strategy,off in [('random','Random',-.03),('heuristic','Heuristic',.03)]:
        row=q[(q.strategy==strategy)&(q.model=='RF')].iloc[0]; b=float(snap[(snap.arm==arm)&(snap.model==model)].AUC_mean.iloc[0]); ax.plot([1,2],[b,row.auc_observed],color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1.4); rec('F32','Revision Code/results/qc_e3_transfer.csv','auc_observed',arm,'','RF',row.auc_observed)
    oracle=q[(q.strategy=='Observed (oracle)')&(q.model=='RF')].iloc[0]; ax.errorbar(3,oracle.auc_observed,yerr=oracle.auc_observed_std,color='#0072B2',marker='o',mfc='white',capsize=3,lw=1.2); ax.axhspan(oracle.auc_observed-oracle.auc_observed_std,oracle.auc_observed+oracle.auc_observed_std,color='#E8A33D',alpha=.12)
    rec('F32','Revision Code/results/qc_e3_transfer.csv','auc_observed','oracle','','RF',oracle.auc_observed); rec('F32','Revision Code/results/qc_e3_transfer.csv','auc_observed_std','oracle','','RF',oracle.auc_observed_std)
    ax.text(1.5,.69,'Generative arms: Alberta only',ha='center',fontsize=7); ax.set(xticks=x,xticklabels=names,ylabel='AUC',ylim=(.5,1.02)); ax.grid(axis='y',alpha=.25); ax.legend(ncol=2,bbox_to_anchor=(1.02,1),loc='upper left'); fig.text(.01,.01,'Alberta n = 3,370 / 5,500; Quebec n = 521 / 521 / 1,042 records',fontsize=7); fig.subplots_adjust(left=.1,right=.7,bottom=.2); finish(fig,'F32_honest_ceiling')

FUN={'F26':f26,'F27':f27,'F28':f28,'F29':f29,'F30':f30,'F31':f31,'F32':f32,'F33':f33}
def provenance():
    cols=['figure','source_file','source_column','arm','seed','series','value']; figs={r[0] for r in RECORDS}; tmp=PROV.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as fo:
        w=csv.writer(fo); w.writerow(cols)
        if PROV.exists() and PROV.stat().st_size:
            with PROV.open(newline='',encoding='utf-8') as fi:
                for row in csv.DictReader(fi):
                    if row.get('figure') not in figs: w.writerow([row.get(c,'') for c in cols])
        w.writerows(RECORDS)
    tmp.replace(PROV)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only',choices=list(FUN)); a=ap.parse_args(); keys=[a.only] if a.only else list(FUN)
    for k in keys:FUN[k]()
    provenance(); print('generated',','.join(keys)); [print(k,':',v or 'no clipping warning') for k,v in WARN.items()]
if __name__=='__main__':main()
