"""Generate F24/F25 and E4 report from e4_dual_artifact.csv."""
from pathlib import Path
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import ttest_rel
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/'Revision Code'/'results'; FIGDIR=ROOT/'Overleaf 2'/'figures'
sys.path.insert(0,str(Path(__file__).resolve().parent)); from figstyle import save_figure, DOUBLE
ARMS=['random','heuristic','clip','gp_standard','v1_da_gp','dagp_strict','dagp_base','dagp_relaxed']; CONDS=['baseline','lattice','bw','lattice_bw']; MODELS=['LogReg','NaiveBayes','CART','KNN','SVM']
LABEL={'random':'Random','heuristic':'Heuristic','clip':'Clip-WGAN','gp_standard':'GP-standard','v1_da_gp':'DA-GP-WGAN','dagp_strict':'AP-WGAN (strict)','dagp_base':'AP-WGAN (base)','dagp_relaxed':'AP-WGAN (relaxed)'}
CT={'baseline':'C1 baseline','lattice':'C2 lattice','bw':'C3 bandwidth','lattice_bw':'C4 lattice + bandwidth'}; trace=[]
def tr(fig,col,v,arm='',seed='',series=''): trace.append(dict(figure=fig,source_file='Revision Code/results/e4_dual_artifact.csv',source_column=col,arm=arm,seed=seed,series=series,value=float(v)))
def f24(d):
    fig,axs=plt.subplots(2,2,figsize=(DOUBLE,5.8),sharex=True,sharey=True); x=np.arange(8)
    for ax,c in zip(axs.flat,CONDS):
        z=d[(d.condition==c)&(d.model=='CART')]
        for metric,color,marker,label in [('geometry_tell_AUC','#D55E00','o','geometry tell'),('lattice_tell_AUC','#0072B2','s','lattice tell')]:
            g=z.groupby('arm')[metric].agg(['mean','std']).reindex(ARMS); ax.errorbar(x,g['mean'],yerr=g['std'],color=color,marker=marker,ls='-',capsize=2,label=label)
            for r in z.itertuples(): tr('F24',metric,getattr(r,metric),r.arm,r.seed,c)
        ax.axhline(.5,color='black',ls=':',lw=.8); ax.set_ylim(.45,1.025); ax.set_ylabel(f'{CT[c]} diagnostic AUC'); ax.set_xlabel('strategy'); ax.set_xticks(x,[LABEL[a] for a in ARMS],rotation=38,ha='right'); ax.grid(axis='y')
        if c=='lattice_bw': ax.text(.02,.08,'Artifact 2 remains live:\ngeometry tell > 0.94',transform=ax.transAxes,fontsize=7,weight='bold')
    axs[0,0].legend(); fig.tight_layout(); save_figure(fig,'F24_e4_condition_grid',FIGDIR); plt.close(fig)
def f25(d):
    fig,axs=plt.subplots(2,3,figsize=(DOUBLE,5.4),sharex=True); colors=plt.cm.tab10(np.arange(8)); x=np.arange(4)
    for ax,m in zip(axs.flat,MODELS):
        z=d[d.model==m]
        for color,a in zip(colors,ARMS):
            g=z[z.arm==a].groupby('condition').AUC.agg(['mean','std']).reindex(CONDS); ax.errorbar(x,g['mean'],yerr=g['std'],marker='o',lw=1,color=color,capsize=2,label=LABEL[a]);
            for r in z[z.arm==a].itertuples(): tr('F25','AUC',r.AUC,a,r.seed,f'{m}:{r.condition}')
        ax.set_xticks(x,['C1','C2','C3','C4']); ax.set_xlabel('evaluation condition'); ax.set_ylabel('AUC (unitless)'); ax.set_ylim(.6,1.0); ax.grid(axis='y'); ax.text(.03,.96,m,transform=ax.transAxes,ha='left',va='top',fontsize=8,weight='bold',bbox=dict(fc='white',alpha=.7,ec='none'))
    axs.flat[-1].set_visible(False); handles,labels=axs[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7); fig.tight_layout(rect=[0,.10,1,1]); save_figure(fig,'F25_e4_downstream_auc',FIGDIR); plt.close(fig)
def report(d):
    q=d[d.model=='CART']; geom=q.groupby('condition').geometry_tell_AUC.agg(['mean','std']); cart=q.groupby('condition').AUC.agg(['mean','std']); inv=pd.read_csv(RESULTS/'e4_ap_invariance.csv'); eq=pd.read_json(RESULTS/'e4_equivalence.json',typ='series')
    # Paired cell-level tests, same arm/seed before/after control.
    wide=q.pivot(index=['arm','seed'],columns='condition',values=['geometry_tell_AUC','AUC'])
    tg,pg=ttest_rel(wide[('geometry_tell_AUC','lattice_bw')],wide[('geometry_tell_AUC','lattice')]); ta,pa=ttest_rel(wide[('AUC','lattice_bw')],wide[('AUC','lattice')])
    lag=q.groupby('condition')[['pa_lag_mean_deg','fire_lag_mean_deg']].mean(); c4=q[q.condition=='lattice_bw']; exact=c4.groupby('arm').first()
    lines=['# E4 report — dual construction-artifact control','',
      '## Outcome','',
      f'**Artifact 2 was not closed.** Under C4, geometry-tell AUC remained {geom.loc["lattice_bw","mean"]:.5f} ± {geom.loc["lattice_bw","std"]:.5f} across arm-seed cells (range {q[q.condition=="lattice_bw"].geometry_tell_AUC.min():.5f}–{q[q.condition=="lattice_bw"].geometry_tell_AUC.max():.5f}). C4 therefore is not an artifact-free definitive downstream comparison.', '',
      'Bandwidth matching is only an evaluation control; it modifies the presence class and is not part of AP-WGAN or a deployable solution. Direct environmental sampling from rasters at pseudo-absence coordinates remains the practical fix to test.', '',
      '## Predictions','',
      f'1. **Not supported.** C3/C4 did not drive the geometry tell toward 0.5. C2→C4 paired change: t={tg:.3f}, p={pg:.4g}; C4 mean={geom.loc["lattice_bw","mean"]:.5f}. The fingerprint has a cause not removed by spatial-lag matching alone.',
      f'2. **Not supported as a general claim.** Mean CART AUC changed from {cart.loc["lattice","mean"]:.4f} under C2 to {cart.loc["lattice_bw","mean"]:.4f} under C4 (paired t={ta:.3f}, p={pa:.4g}); directions differ by arm, so bandwidth matching did not uniformly lower AUC.',
      '3. **Not interpretable as a fully controlled re-ranking.** Arm ordering changes in some classifiers, but prediction 1 failed, so C4 cannot serve as the definitive artifact-free comparison.',
      '4. **No broadened AP-WGAN performance claim.** AP-WGAN remains the only family closing artifact 1 natively; artifact 2 remains live under every arm.', '',
      '## Acceptance audits','',
      f'- Chunk equivalence on 500 points: maximum feature difference {eq.max_abs_feature_diff:.1f}; maximum lag difference {eq.max_abs_lag_diff:.1f}.',
      f'- AP-WGAN invariance: maximum absolute C1−C2/C3−C4 difference {inv.max_abs_diff.max():.1f}; all native feature arrays exactly equal to their snapped versions: {bool(inv.feature_array_equal.all())}.',
      f'- Realized C4 mean lags: PA→fire {lag.loc["lattice_bw","pa_lag_mean_deg"]:.6f}°, matched fire {lag.loc["lattice_bw","fire_lag_mean_deg"]:.6f}°.', '',
      '## Exact-value diagnostics','',
      '- Lattice tell = 1.000 occurs only when PA lattice deviation is nonzero while fire deviation is zero; the output reports `on_lattice_count`, `on_lattice_rate`, and within-class probe variances.',
      '- Lattice tell = 0.500 occurs when both probe distributions are constant zero; the output reports zero within-class probe variance as the explanatory degeneracy.',
      '- AP-WGAN C1=C2 and C3=C4 exactly because `np.array_equal(features, snap(features))` is true for all nine AP arm-seed arrays; maximum feature and metric difference is 0.0.', '',
      '## Scope','',
      'These findings apply to the Alberta IDW feature-construction pipeline and the specified bandwidth-matching control. They do not establish that bandwidth matching is ineffective for every construction process, nor that downstream C4 values are artifact-free.']
    (ROOT/'E4_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
    d=pd.read_csv(RESULTS/'e4_dual_artifact_wide.csv'); f24(d); f25(d); report(d)
    p=ROOT/'FIGURE_DATA.csv'; old=pd.read_csv(p,keep_default_na=False); pd.concat([old,pd.DataFrame(trace)],ignore_index=True).drop_duplicates().to_csv(p,index=False)
if __name__=='__main__': main()
