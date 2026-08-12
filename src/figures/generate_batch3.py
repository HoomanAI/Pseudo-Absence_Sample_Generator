"""Round-3 publication figures. Every mark is sourced from repository data."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree, ConvexHull

HERE=Path(__file__).resolve().parent; CODE=HERE.parent; ROOT=CODE.parent
RESULTS=CODE/'results'; RETRAIN=CODE/'retrain'; OUT=ROOT/'Overleaf 2'/'figures'; PROV=ROOT/'FIGURE_DATA.csv'
sys.path.insert(0,str(HERE)); from figstyle import save_figure, DOUBLE

ARMS=['random','heuristic','clip','gp_standard','v1_da_gp','dagp_strict','dagp_base','dagp_relaxed']
LABEL={'random':'Random','heuristic':'Heuristic','clip':'GAN (clip)','gp_standard':'WGAN-GP','v1_da_gp':'DA-GP-WGAN','dagp_strict':'AP-WGAN (strict)','dagp_base':'AP-WGAN (base)','dagp_relaxed':'AP-WGAN (relaxed)'}
COLOR=dict(zip(ARMS,['#7F7F7F','#8C564B','#D55E00','#CC79A7','#009E73','#56B4E9','#0072B2','#7BAFD4']))
MARK=dict(zip(ARMS,['o','s','^','v','D','P','X','*']))
MODELS=['LogReg','NaiveBayes','CART','KNN','SVM']; MLABEL=['LogReg','Naive Bayes','CART','KNN (k=9)','SVM-RBF']
REF='#E8A33D'; WARN={}; RECORDS=[]

def rec(fig,source,col,arm='',seed='',series='',value=np.nan):
    RECORDS.append([fig,str(source).replace('\\','/'),col,arm,seed,series,float(value) if pd.notna(value) else ''])
def panel(ax,s):
    fn=getattr(ax,'text2D',ax.text); fn(-.14,1.06,s,transform=ax.transAxes,weight='bold',ha='right',va='bottom',clip_on=False)
def finish(fig,stem): WARN[stem]=save_figure(fig,stem,OUT); plt.close(fig)
def mean_sd(df, keys, val):
    return df.groupby(keys,as_index=False)[val].agg(['mean','std']).reset_index()
def text_color(im,v):
    rgba=im.cmap(im.norm(v)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]; return 'black' if lum>.55 else 'white'
def heat(ax,A,rows,cols,title,cmap,fmt='.3f',vmin=None,vmax=None,missing=False,norm=None):
    m=np.ma.masked_invalid(A); im=ax.imshow(m,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,norm=norm)
    ax.set_xticks(range(len(cols)),cols,rotation=35,ha='right'); ax.set_yticks(range(len(rows)),rows)
    ax.set_xlabel(title)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v=A[i,j]; ax.text(j,i,'—' if np.isnan(v) else format(v,fmt),ha='center',va='center',fontsize=6.5,color='black' if np.isnan(v) else text_color(im,v))
    return im
def ellipse(ax,x,y,c):
    if len(x)<2:return
    cov=np.cov(x,y); vals,vec=np.linalg.eigh(cov); order=vals.argsort()[::-1]; vals=vals[order]; vec=vec[:,order]
    angle=np.degrees(np.arctan2(*vec[:,0][::-1])); e=Ellipse((np.mean(x),np.mean(y)),2*np.sqrt(max(vals[0],0)),2*np.sqrt(max(vals[1],0)),angle=angle,fc=c,ec=c,alpha=.12,lw=.8); ax.add_patch(e)

def f1():
    man=pd.read_csv(RETRAIN/'manifest.csv'); fire=pd.read_csv(ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv').dropna()
    vals=[]
    for r in man.itertuples():
        pts=np.load(RETRAIN/f'{r.label}_seed{r.seed}'/f'pts_{r.arm}.npy'); nn=cKDTree(pts).query(pts,k=2)[0][:,1].mean()
        vals.append([r.label,r.seed,r.K_SSE_rawK,r.border_fraction,nn])
    d=pd.DataFrame(vals,columns=['arm','seed','K-SSE','Border fraction','Mean NN distance'])
    fig,axs=plt.subplots(3,1,figsize=(DOUBLE,1.1*DOUBLE),sharex=True); x=np.arange(8)
    refnn=cKDTree(fire[['LONGITUDE','LATITUDE']]).query(fire[['LONGITUDE','LATITUDE']],k=2)[0][:,1].mean()
    for k,(ax,col) in enumerate(zip(axs,['K-SSE','Border fraction','Mean NN distance'])):
        s=d.groupby('arm')[col].agg(['mean','std']).reindex(ARMS); y=s['mean'].to_numpy(); sd=s['std'].to_numpy()
        ax.plot(x,y,color='#365E7D',marker='o',mfc='white',lw=1.4); ax.fill_between(x,y-sd,y+sd,color='#365E7D',alpha=.18)
        ref=0 if col!='Mean NN distance' else refnn; ax.axhline(ref,color=REF,ls='--',lw=1.4); ax.text(.98,ref,' fire reference',transform=ax.get_yaxis_transform(),rotation=0,va='bottom',ha='right',fontsize=7)
        ax.set_xticks(x,[LABEL[a] for a in ARMS] if k==2 else ['']*8,rotation=40,ha='right'); ax.set_ylabel(col); ax.grid(axis='y'); panel(ax,chr(97+k))
        for _,r in d.iterrows(): rec('F1','Revision Code/retrain/manifest.csv' if col!='Mean NN distance' else f"Revision Code/retrain/{r.arm}_seed{r.seed}/pts_{r.arm}.npy",col,r.arm,r.seed,col,r[col])
    fig.subplots_adjust(left=.16,right=.98,bottom=.22,top=.98,hspace=.16); finish(fig,'F1_spatial_fidelity')

def f15():
    raw=pd.read_csv(RESULTS/'arm_eval_summary.csv'); snap=pd.read_csv(RESULTS/'arm_eval_summary_snapped.csv'); man=pd.read_csv(RETRAIN/'manifest.csv'); q=pd.read_csv(RESULTS/'qc_e3_transfer.csv')
    def mat(df,col): return df.pivot(index='arm',columns='model',values=col).reindex(index=ARMS,columns=MODELS).to_numpy()
    auc=mat(snap,'AUC_mean'); tss=mat(snap,'TSS_mean'); delta=auc-mat(raw,'AUC_mean')
    sm=man.groupby('label').agg(K=('K_SSE_rawK','mean'),B=('border_fraction','mean')).reindex(ARMS)
    qm=np.full((8,2),np.nan); qmap={'Random':'random','Heuristic':'heuristic'}
    for r in q.itertuples():
        if r.strategy in qmap: qm[ARMS.index(qmap[r.strategy]),['RF','XGB'].index(r.model)]=r.inflation
    arrays=[auc,tss,delta,sm[['K']].to_numpy(),sm[['B']].to_numpy(),qm]; titles=['Snapped AUC','Snapped TSS','ΔAUC (snapped − raw)','K-SSE','Border fraction','Quebec inflation']; cm=['YlGn','YlGn','RdBu','YlOrRd','YlOrRd','YlOrRd']; cols=[MLABEL,MLABEL,MLABEL,['K-SSE'],['Border'],['RF','XGB']]; fmts=['.3f','.3f','+.3f','.4f','.3f','.3f']
    fig,axs=plt.subplots(3,2,figsize=(DOUBLE,8.0));
    for k,ax in enumerate(axs.flat):
        lim=max(abs(np.nanmin(delta)),abs(np.nanmax(delta))) if k==2 else None
        heat(ax,arrays[k],[LABEL[a] for a in ARMS],cols[k],titles[k],cm[k],fmts[k],norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim) if k==2 else None); panel(ax,chr(97+k))
        if k % 2: ax.set_yticklabels([])
        if k==2:
            for sp in ax.spines.values(): sp.set_linewidth(2); sp.set_color('#333333')
    fig.subplots_adjust(left=.18,right=.99,wspace=.28,hspace=.72); finish(fig,'F15_auc_tss_heatmap')
    for name,A in zip(titles,arrays):
        for i,a in enumerate(ARMS):
            for j,v in enumerate(A[i]):
                if np.isfinite(v): rec('F15','derived from named result tables',name,a,'',cols[titles.index(name)][j],v)

def f9():
    man=pd.read_csv(RETRAIN/'manifest.csv'); pc=pd.read_csv(RESULTS/'arm_eval_percell_snapped.csv'); svm=pc[pc.model=='SVM'][['arm','seed','AUC']]
    d=man.merge(svm,left_on=['label','seed'],right_on=['arm','seed']); x=d.K_SSE_rawK.to_numpy(); y=d.border_fraction.to_numpy(); z=d.AUC.to_numpy()
    fig=plt.figure(figsize=(DOUBLE,DOUBLE)); gs=fig.add_gridspec(2,2); a3=fig.add_subplot(gs[0,0],projection='3d'); axes=[a3,fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    for arm in ARMS:
        u=d[d.label==arm]; a3.scatter(u.K_SSE_rawK,u.border_fraction,u.AUC,c=COLOR[arm],marker=MARK[arm],s=46)
        for r in u.itertuples(): a3.plot([r.K_SSE_rawK]*2,[r.border_fraction]*2,[z.min(),r.AUC],color=COLOR[arm],alpha=.3,lw=1.0)
        for ax,xx,yy in [(axes[1],u.K_SSE_rawK,u.border_fraction),(axes[2],u.K_SSE_rawK,u.AUC),(axes[3],u.border_fraction,u.AUC)]: ax.scatter(xx,yy,c=COLOR[arm],marker=MARK[arm],s=40); ellipse(ax,np.asarray(xx),np.asarray(yy),COLOR[arm])
    a3.set(xlabel='K-SSE',ylabel='Border fraction',zlabel='Snapped SVM AUC'); a3.zaxis.labelpad=-3; axes[1].set(xlabel='K-SSE',ylabel='Border fraction'); axes[2].set(xlabel='K-SSE',ylabel='SVM AUC'); axes[3].set(xlabel='Border fraction',ylabel='SVM AUC')
    for k,ax in enumerate(axes):
        panel(ax,chr(97+k)); ax.xaxis.label.set_size(11); ax.yaxis.label.set_size(11); ax.tick_params(labelsize=9.5)
        if hasattr(ax,'zaxis'): ax.zaxis.label.set_size(11); ax.zaxis.set_tick_params(labelsize=9.5)
    handles=[Line2D([],[],color=COLOR[a],marker=MARK[a],ls='',label=LABEL[a]) for a in ARMS]; fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=4,frameon=False); fig.subplots_adjust(left=.1,right=.98,bottom=.14,top=.98,wspace=.38,hspace=.32); finish(fig,'F9_tradeoff_3d')
    for r in d.itertuples():
        for col in ['K_SSE_rawK','border_fraction','AUC']: rec('F9','Revision Code/retrain/manifest.csv' if col!='AUC' else 'Revision Code/results/arm_eval_percell_snapped.csv',col,r.label,r.seed,'SVM',getattr(r,col))

def f8():
    data=pd.read_csv(ROOT/'Pseudo-Absence_Sample_Generator'/'data'/'Fire_points_dataset_final_csv.csv').dropna(); fcols=[c for c in data if c not in ['FIRE','LONGITUDE','LATITUDE','YEAR','MONTH','DAY']]; idx=[fcols.index(c) for c in ['slope','twi','temp_1m_prior']]
    manifest=pd.read_csv(RETRAIN/'manifest.csv').query('seed == 42').set_index('label')
    fig=plt.figure(figsize=(DOUBLE,5.5)); gs=fig.add_gridspec(2,2); axes=[fig.add_subplot(gs[0,0],projection='3d'),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    for arm in ARMS:
        file_arm=manifest.loc[arm,'arm']; A=np.load(RETRAIN/f'{arm}_seed42'/f'feats_{file_arm}.npy')[:,idx]; axes[0].scatter(*A.T,c=COLOR[arm],marker=MARK[arm],s=2,alpha=.18,rasterized=True)
        source=f'Revision Code/retrain/{arm}_seed42/feats_{file_arm}.npy'
        for obs,row in enumerate(A):
            for name,value in zip(['slope','twi','temp_1m_prior'],row): rec('F8',source,name,arm,42,f'observation={obs}',value)
        for ax,i,j in [(axes[1],0,1),(axes[2],0,2),(axes[3],1,2)]: ax.scatter(A[:,i],A[:,j],c=COLOR[arm],marker=MARK[arm],s=2,alpha=.15,rasterized=True)
    names=['Slope','TWI','Temp. (1 mo prior)']; axes[0].set(xlabel=names[0],ylabel=names[1],zlabel=names[2]); axes[1].set(xlabel=names[0],ylabel=names[1]); axes[2].set(xlabel=names[0],ylabel=names[2]); axes[3].set(xlabel=names[1],ylabel=names[2])
    for ax,i,j in [(axes[1],0,1),(axes[2],0,2),(axes[3],1,2)]:
        for v in np.unique(data[fcols[idx[i]]]): ax.axvline(v,color='.75',lw=.35,zorder=0)
        for v in np.unique(data[fcols[idx[j]]]): ax.axhline(v,color='.75',lw=.35,zorder=0)
    for k,ax in enumerate(axes): panel(ax,chr(97+k))
    fig.legend(handles=[Line2D([],[],color=COLOR[a],marker=MARK[a],ls='',label=LABEL[a]) for a in ARMS],loc='center left',bbox_to_anchor=(.99,.5)); fig.subplots_adjust(left=.1,right=.77,wspace=.42,hspace=.35); finish(fig,'F8_admissible_lattice_3d')

def f4_f5():
    raw=pd.read_csv(RESULTS/'arm_eval_tell.csv'); snap=pd.read_csv(RESULTS/'arm_eval_tell_snapped.csv')
    fig,ax=plt.subplots(figsize=(DOUBLE,3.1));
    for arm in ARMS:
        r=raw[raw.arm==arm]; s=snap[snap.arm==arm]; y=[r.tell_AUC.mean(),s.tell_AUC.mean()]; ap=arm.startswith('dagp_')
        ax.plot([0,1],y,color=COLOR[arm],marker=MARK[arm],mfc='white',label=LABEL[arm],lw=2 if ap else 1.2,alpha=1 if ap else .75,zorder=5 if ap else 2)
        for row in r.itertuples(): rec('F4','Revision Code/results/arm_eval_tell.csv','tell_AUC',arm,row.seed,'raw',row.tell_AUC)
        for row in s.itertuples(): rec('F4','Revision Code/results/arm_eval_tell_snapped.csv','tell_AUC',arm,row.seed,'projected',row.tell_AUC)
    ax.axhline(.5,color=REF,ls='--',lw=1.4); ax.text(.98,.495,'chance — no construction signal',transform=ax.get_yaxis_transform(),ha='right',va='top',fontsize=7)
    ax.text(.48,.24,'AP-WGAN arms require no projection',transform=ax.transAxes,fontsize=8,weight='bold')
    ax.set(xticks=[0,1],xticklabels=['Raw',r'Projected onto $\mathcal{L}_j$'],ylabel='Tell AUC',ylim=(.45,1.05)); ax.grid(axis='y'); ax.legend(ncol=2,bbox_to_anchor=(1.02,1),loc='upper left'); fig.subplots_adjust(right=.7); finish(fig,'F4_admissibility_tell')
    q=pd.read_csv(RESULTS/'qc_e3_transfer.csv'); fig,ax=plt.subplots(figsize=(DOUBLE,3.2)); xpos={'RF':0,'XGB':1}
    for strategy,g in q.groupby('strategy'):
        arm={'Random':'random','Heuristic':'heuristic'}.get(strategy); col='#0072B2' if strategy=='Observed (oracle)' else COLOR[arm]
        for r in g.itertuples():
            x=xpos[r.model]+(-.08 if strategy=='Random' else .08 if strategy=='Heuristic' else 0); ax.plot([x-.13,x+.13],[r.auc_synthetic,r.auc_observed],color=col,marker='o',mfc='white',lw=1.2); ax.fill_between([x-.13,x+.13],[r.auc_synthetic-r.auc_synthetic_std,r.auc_observed-r.auc_observed_std],[r.auc_synthetic+r.auc_synthetic_std,r.auc_observed+r.auc_observed_std],color=col,alpha=.16)
            rec('F5','Revision Code/results/qc_e3_transfer.csv','auc_synthetic',arm or 'oracle','',r.model,r.auc_synthetic); rec('F5','Revision Code/results/qc_e3_transfer.csv','auc_observed',arm or 'oracle','',r.model,r.auc_observed)
    oracle=q[(q.strategy=='Observed (oracle)')&(q.model=='RF')].iloc[0]; ax.axhspan(oracle.auc_observed-oracle.auc_observed_std,oracle.auc_observed+oracle.auc_observed_std,color=REF,alpha=.13); ax.axhline(oracle.auc_observed,color=REF,ls='--',lw=1.4,label='Observed ceiling (RF)')
    ax.set(xticks=[0,1],xticklabels=['RF','XGB'],ylabel='AUC',ylim=(.48,1.02)); ax.grid(axis='y'); ax.legend(); finish(fig,'F5_quebec_transfer')

def f23():
    raw=pd.read_csv(RESULTS/'arm_eval_summary.csv'); snap=pd.read_csv(RESULTS/'arm_eval_summary_snapped.csv'); fig,axs=plt.subplots(1,2,figsize=(DOUBLE,3.2),sharex=True); x=np.arange(5)
    for arm in ARMS:
        s=snap[snap.arm==arm].set_index('model').reindex(MODELS); r=raw[raw.arm==arm].set_index('model').reindex(MODELS); y=s.AUC_mean.to_numpy(); sd=s.AUC_sd.to_numpy(); delta=y-r.AUC_mean.to_numpy(); dsd=np.sqrt(sd**2+r.AUC_sd.to_numpy()**2)
        for ax,v,e in [(axs[0],y,sd),(axs[1],delta,dsd)]: ax.plot(x,v,color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1,label=LABEL[arm]); ax.fill_between(x,v-e,v+e,color=COLOR[arm],alpha=.15)
        for model,v in zip(MODELS,y): rec('F23','Revision Code/results/arm_eval_summary_snapped.csv','AUC_mean',arm,'',model,v)
        for model,v in zip(MODELS,delta): rec('F23','Revision Code/results/arm_eval_summary.csv + arm_eval_summary_snapped.csv','delta_AUC',arm,'',model,v)
    axs[0].set_ylabel('Snapped AUC'); axs[1].set_ylabel('Snapped − raw AUC'); axs[1].axhline(0,color='.3',lw=.8)
    for k,ax in enumerate(axs): ax.set_xticks(x,MLABEL,rotation=35,ha='right'); ax.grid(axis='y'); panel(ax,chr(97+k))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='center left',bbox_to_anchor=(.99,.5)); fig.subplots_adjust(right=.78,bottom=.25,wspace=.32); finish(fig,'F23_arm_profile')

def f24():
    d=pd.read_csv(RESULTS/'e4_dual_artifact_wide.csv'); cond=['baseline','lattice','bw','lattice_bw']; metrics=[('lattice_tell_AUC','Lattice tell AUC'),('geometry_tell_AUC','Geometry tell AUC'),('m1_fire_over_pa','Fire/PA geometry ratio'),('AUC','Downstream CART AUC')]
    fig,axs=plt.subplots(4,1,figsize=(DOUBLE,1.25*DOUBLE),sharex=True);
    for k,(col,title) in enumerate(metrics):
        u=d[d.model=='CART'].groupby(['arm','condition'])[col].mean().unstack().reindex(index=ARMS,columns=cond); A=u.to_numpy(); heat(axs.flat[k],A,[LABEL[a] for a in ARMS],cond if k==3 else ['']*4,title,'YlOrRd' if k==2 else 'YlGn','.3f',.5,1 if k<2 else None); panel(axs.flat[k],chr(97+k))
        for i,a in enumerate(ARMS):
            for j,c in enumerate(cond): rec('F24','Revision Code/results/e4_dual_artifact_wide.csv',col,a,'',c,A[i,j])
    fig.subplots_adjust(left=.22,right=.98,bottom=.11,top=.98,hspace=.26); finish(fig,'F24_e4_condition_grid')

def f21():
    src=RESULTS/'projection_response.csv'
    if not src.exists(): print('skip F21: projection_response.csv does not exist'); return
    d=pd.read_csv(src); s=d.groupby(['arm','p']).AUC.agg(['mean','std']).reset_index(); fig,axs=plt.subplots(3,1,figsize=(DOUBLE,6.5),sharex=True)
    for arm in ARMS:
        u=s[s.arm==arm].sort_values('p'); p=u.p.to_numpy(); y=u['mean'].to_numpy(); sd=u['std'].fillna(0).to_numpy(); d1=np.diff(y); d2=np.diff(y,n=2)
        axs[0].plot(p,y,color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1,label=LABEL[arm]); axs[0].fill_between(p,y-sd,y+sd,color=COLOR[arm],alpha=.15)
        axs[1].plot(p[1:],d1,color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1); axs[2].plot(p[2:],d2,color=COLOR[arm],marker=MARK[arm],mfc='white',lw=1)
    for k,ax in enumerate(axs): ax.axvline(13,color=REF,ls='--',lw=1.4); ax.grid(axis='y'); panel(ax,chr(97+k))
    axs[0].set_ylabel('AUC'); axs[1].set_ylabel('ΔAUC / Δp'); axs[2].set_ylabel('Δ²AUC'); axs[2].set_xlabel('Projected reclassified dimensions, p'); axs[2].axhline(0,color='.3',lw=.7); axs[0].text(.985,.05,'fully projected',rotation=90,transform=axs[0].transAxes,ha='right',fontsize=7)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='center left',bbox_to_anchor=(.99,.5)); fig.subplots_adjust(right=.78,hspace=.12); finish(fig,'F21_projection_response')
    for r in d.itertuples():
        for col in ['AUC','TSS','tell_AUC']: rec('F21','Revision Code/results/projection_response.csv',col,r.arm,r.seed,f'{r.model}:p={r.p}:subset={r.subset_id}',getattr(r,col))

def f22():
    src=RESULTS/'sweeps'/'sweep_results.csv'
    if not src.exists(): print('skip F22: stress sweep has not been authorized/run'); return
    d=pd.read_csv(src)
    if 'K_SSE_rawK' not in d: print('skip F22: sweep_results.csv has no scored K_SSE_rawK column; run spatial scoring first'); return

FUN={'F1':f1,'F15':f15,'F9':f9,'F8':f8,'F4':lambda:f4_f5(),'F5':lambda:None,'F21':f21,'F22':f22,'F23':f23,'F24':f24}
def write_prov():
    old=pd.read_csv(PROV); figs={r[0] for r in RECORDS}; old=old[~old.figure.astype(str).isin(figs)]; new=pd.DataFrame(RECORDS,columns=['figure','source_file','source_column','arm','seed','series','value']); pd.concat([old,new],ignore_index=True).to_csv(PROV,index=False)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only',choices=list(FUN)); args=ap.parse_args(); keys=[args.only] if args.only else list(FUN)
    for key in keys: FUN[key]()
    write_prov(); print('generated',', '.join(keys));
    for k,v in WARN.items(): print(k,':',v or 'no clipping warning')
if __name__=='__main__': main()
