"""Pseudo-absence infographic: updated with actual study results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings; warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(20, 13), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20); ax.set_ylim(0, 13)
ax.axis('off'); ax.set_facecolor('white')

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.25, zorder=2):
    p = FancyBboxPatch((x,y), w, h, boxstyle=f'round,pad={radius}',
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p); return p

def txt(ax, x, y, s, fs=9, fw='normal', col='#222', ha='center', va='center',
        zorder=5, wrap=False):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col,
            ha=ha, va=va, zorder=zorder, fontfamily='DejaVu Sans',
            wrap=wrap, linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, col='#555', lw=1.5, hw=0.15, hl=0.2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=f'->,head_width={hw},head_length={hl}',
                                color=col, lw=lw), zorder=6)

def fire_dot(ax, x, y, r=0.09, col='#E74C3C', zorder=8):
    c = plt.Circle((x,y), r, color=col, zorder=zorder)
    ax.add_patch(c)

def pa_dot(ax, x, y, r=0.09, col='#3498DB', zorder=8):
    c = plt.Circle((x,y), r, color=col, zorder=zorder)
    ax.add_patch(c)

# ══════════════════════════════════════════════════════════════════════════════
# TOP ROW
# ══════════════════════════════════════════════════════════════════════════════

# Box 1 — "What we have"
rbox(ax, 0.3, 8.6, 4.4, 4.0, '#FFFFFF', '#888', lw=1.5)
rbox(ax, 0.55, 8.75, 3.9, 3.2, '#F0FAF0', '#6CC070', lw=1.2, radius=0.15)
txt(ax, 2.5, 12.35, 'What we have', fs=13, fw='bold', col='#222')
txt(ax, 2.5, 9.0, 'No "non-fire" records exist', fs=8.5, col='#666')
# fire dots cluster
for (dx,dy) in [(1.1,1.0),(1.5,1.4),(1.9,1.0),(1.3,1.7),(1.8,1.7),(2.2,1.4),(1.6,1.1)]:
    fire_dot(ax, 0.55+dx, 8.75+dy)
txt(ax, 1.4, 10.35, '● Fire location', fs=8, col='#E74C3C', ha='left')
txt(ax, 1.4, 10.0,  'Study area', fs=8, col='#6CC070', ha='left')

# Arrow 1→2
arrow(ax, 4.7, 10.6, 5.15, 10.6, col='#555', lw=1.8)
txt(ax, 4.93, 10.85, 'needs', fs=8.5, col='#555')

# Box 2 — "What the model needs"
rbox(ax, 5.2, 8.6, 4.8, 4.0, '#FFFFFF', '#888', lw=1.5)
txt(ax, 7.6, 12.35, 'What the model needs', fs=13, fw='bold', col='#222')
txt(ax, 7.6, 8.82, 'Both classes needed to train', fs=8.5, col='#666')
txt(ax, 6.4, 11.9, 'Fire (1)', fs=11, fw='bold', col='#E74C3C')
for (dx,dy) in [(0,0),(0.4,0.35),(0.2,0.7),(-0.25,0.4),(0.5,0.7)]:
    fire_dot(ax, 6.15+dx, 10.6+dy)
# dashed divider
ax.plot([7.6,7.6],[9.1,12.1],'--',color='#BBB',lw=1.2,zorder=4)
txt(ax, 8.8, 11.9, 'No-fire (0)', fs=11, fw='bold', col='#5B9BD5')
ax.text(8.75, 10.85, '?', fontsize=36, color='#BBB', ha='center', va='center',
        fontweight='bold', zorder=5)
txt(ax, 8.75, 9.9, 'unknown', fs=8.5, col='#AAA')

# Arrow 2→3
arrow(ax, 10.0, 10.6, 10.45, 10.6, col='#555', lw=1.8)
txt(ax, 10.23, 10.85, 'leads to', fs=8.5, col='#555')

# Box 3 — Core Problem (yellow)
rbox(ax, 10.5, 8.6, 5.2, 4.0, '#FFFDE7', '#F4C842', lw=2.0)
txt(ax, 13.1, 12.35, '⚠  Core Problem', fs=13, fw='bold', col='#7D5A00')
txt(ax, 13.1, 11.7,  'True absences don\'t exist.', fs=10.5, col='#5A4000')
txt(ax, 13.1, 11.2,  'We must generate', fs=10.5, col='#5A4000')
txt(ax, 13.1, 10.65, 'pseudo-absences', fs=14, fw='bold', col='#7D5A00')
txt(ax, 13.1, 10.15, 'as training controls.', fs=10.5, col='#5A4000')
txt(ax, 13.1, 9.55,  'Placement quality determines', fs=9.5, col='#7A6000')
txt(ax, 13.1, 9.1,   'whether the model is reliable.', fs=9.5, col='#7A6000')

# Arrow down from core problem → "Two strategies"
arrow(ax, 13.1, 8.6, 13.1, 8.1, col='#555', lw=2.0, hw=0.18, hl=0.22)
# Split arrows to left and right boxes
ax.plot([3.9,13.1],[8.1,8.1],color='#555',lw=1.8,zorder=4)
ax.plot([13.1,16.1],[8.1,8.1],color='#555',lw=1.8,zorder=4)
arrow(ax, 3.9,8.1, 3.9,7.75, col='#555', lw=1.8, hw=0.18, hl=0.22)
arrow(ax, 16.1,8.1, 16.1,7.75, col='#555', lw=1.8, hw=0.18, hl=0.22)
txt(ax, 10.0, 8.28, 'Two strategies', fs=10, fw='bold', col='#555')

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM LEFT — Existing: Random Placement
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 0.3, 0.3, 7.6, 7.3, '#FDECEA', '#C0392B', lw=2.5, radius=0.3)
txt(ax, 4.1, 7.35, 'Existing: Random Placement  ✗', fs=13, fw='bold', col='#C0392B')

# Step labels
for xi,lab in zip([1.05,2.35,3.65,5.3],['① Sampling','② Training Data','③ Quality Test','④ Outcome']):
    txt(ax, xi, 6.9, lab, fs=8, col='#888')

# ── Sampling mini-map ─────────────────────────────────────────────────────────
rbox(ax, 0.45, 3.5, 1.5, 3.3, '#F0FAF0', '#6CC070', lw=1.0, radius=0.12)
fire_positions=[(0.75,6.2),(1.05,6.5),(0.9,5.95),(1.2,6.4),(0.8,5.7),(1.15,6.0)]
for fx,fy in fire_positions: fire_dot(ax,fx,fy)
rand_positions=[(0.6,5.3),(1.25,5.5),(0.7,4.8),(1.3,5.1),(0.55,5.8),(1.1,4.6),(0.9,4.3)]
for rx,ry in rand_positions: pa_dot(ax,rx,ry)
ax.annotate('too close!',xy=(0.82,5.85),xytext=(0.45,5.3),fontsize=7,color='#E74C3C',
            arrowprops=dict(arrowstyle='-',color='#E74C3C',lw=0.8),zorder=9,
            fontfamily='DejaVu Sans')
txt(ax,0.72,3.75,'● Fire', fs=7.5, col='#E74C3C', ha='left')
txt(ax,0.72,3.58,'● Random', fs=7.5, col='#3498DB', ha='left')
rbox(ax,0.5,3.52,1.4,0.32,'#FDECEA','#E74C3C',lw=0.8,radius=0.06)
txt(ax,1.2,3.68,'Biased sample',fs=7.5,col='#C0392B')

# ── Training Data ─────────────────────────────────────────────────────────────
rbox(ax, 2.1, 3.5, 1.5, 3.3, 'white', '#CCC', lw=1.0, radius=0.12)
txt(ax,2.85,6.55,'Training Data',fs=8.5,fw='bold',col='#333')
txt(ax,2.85,6.2,'Fire (1)',fs=8,col='#E74C3C')
for i in range(4): fire_dot(ax,2.52+i*0.22,5.95,r=0.075)
ax.plot([2.2,3.5],[5.75,5.75],'--',color='#DDD',lw=1,zorder=4)
txt(ax,2.85,5.5,'Random (0)',fs=8,col='#3498DB')
for i in range(4): pa_dot(ax,2.52+i*0.22,5.25,r=0.075)
for i in range(4): pa_dot(ax,2.52+i*0.22,5.0,r=0.075)

# ── Quality Test ──────────────────────────────────────────────────────────────
rbox(ax, 3.7, 3.5, 1.6, 3.3, 'white', '#CCC', lw=1.0, radius=0.12)
txt(ax,4.5,6.55,'Quality Test',fs=8.5,fw='bold',col='#333')
txt(ax,4.5,6.2,'Can classifier tell them apart?',fs=7,col='#555')
# Scatter showing separation
fire_q=[(4.0,5.8),(4.2,6.0),(3.95,5.6),(4.25,5.8)]
rand_q=[(4.9,5.4),(5.1,5.6),(4.85,5.2),(5.15,5.4),(5.0,5.65)]
for fx,fy in fire_q: fire_dot(ax,fx,fy,r=0.07)
for rx,ry in rand_q: pa_dot(ax,rx,ry,r=0.07)
ax.plot([4.55,4.55],[4.9,6.15],'--',color='#C0392B',lw=1.5,zorder=6)
rbox(ax,3.78,4.55,1.44,0.65,'#FDECEA','#E74C3C',lw=0.8,radius=0.06)
txt(ax,4.5,5.08,'Easily separated!',fs=7.5,fw='bold',col='#C0392B')
txt(ax,4.5,4.82,'Controls ≠ Fire conditions',fs=7,col='#C0392B')
txt(ax,4.5,4.62,'→ Unrealistic controls ✗',fs=7,fw='bold',col='#C0392B')

# ── Outcome ───────────────────────────────────────────────────────────────────
rbox(ax, 5.45, 3.5, 2.3, 3.3, '#FDECEA', '#C0392B', lw=1.5, radius=0.15)
txt(ax,6.6,6.55,'Outcome',fs=10,fw='bold',col='#C0392B')
txt(ax,6.6,6.2,'Quality Score',fs=8.5,col='#555')
ax.text(6.6,5.55,'0.90',fontsize=36,color='#C0392B',ha='center',va='center',
        fontweight='bold',zorder=5)
txt(ax,6.6,4.9,'AUC  ✗  Too high',fs=9,fw='bold',col='#C0392B')
ax.plot([5.6,7.55],[4.72,4.72],color='#E8B0AA',lw=0.8,zorder=4)
txt(ax,6.6,4.52,'Classifier separates',fs=7.5,col='#666')
txt(ax,6.6,4.28,'fire vs controls easily',fs=7.5,col='#666')
txt(ax,6.6,4.04,'→ controls don\'t look',fs=7.5,col='#888')
txt(ax,6.6,3.80,'like real fire conditions',fs=7.5,col='#888')

# Arrows between sub-panels (left box)
for xa,xb in [(1.95,2.1),(3.6,3.7),(5.3,5.45)]:
    arrow(ax,(xa+xb)/2-0.15+0.15,5.15,(xa+xb)/2+0.08+0.15,5.15,col='#C0392B',lw=1.5,hw=0.1,hl=0.12)

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM RIGHT — Proposed: Heuristic / GAN
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 8.1, 0.3, 7.6, 7.3, '#E8F8F5', '#1A7A5E', lw=2.5, radius=0.3)
txt(ax, 11.9, 7.35, 'Proposed: Heuristic / GAN  ✓', fs=13, fw='bold', col='#1A7A5E')

for xi,lab in zip([8.85,10.15,11.45,13.1],['① Sampling','② Training Data','③ Quality Test','④ Outcome']):
    txt(ax, xi, 6.9, lab, fs=8, col='#888')

# ── Sampling mini-map ─────────────────────────────────────────────────────────
TEAL='#1ABC9C'
rbox(ax, 8.25, 3.5, 1.5, 3.3, '#F0FAF0', '#6CC070', lw=1.0, radius=0.12)
# dashed circle around fire
c_fire = plt.Circle((9.0,5.85),0.55,color='none',ec='#E74C3C',lw=1.2,ls='--',zorder=4)
ax.add_patch(c_fire)
fire_gan=[(8.8,5.9),(9.1,6.15),(8.95,5.65),(9.25,5.95),(8.75,5.6)]
for fx,fy in fire_gan: fire_dot(ax,fx,fy)
gan_pos=[(8.35,5.1),(9.55,5.3),(8.4,4.6),(9.6,4.8),(9.0,4.35),(8.6,6.5),(9.45,6.45)]
for gx,gy in gan_pos:
    c=plt.Circle((gx,gy),0.085,color=TEAL,zorder=8); ax.add_patch(c)
txt(ax,8.55,3.75,'● Fire', fs=7.5, col='#E74C3C', ha='left')
txt(ax,8.55,3.58,'● GAN', fs=7.5, col=TEAL, ha='left')
rbox(ax,8.3,3.52,1.4,0.32,'#E8F8F5','#1A7A5E',lw=0.8,radius=0.06)
txt(ax,9.0,3.68,'Spatially valid',fs=7.5,col='#1A7A5E')

# ── Training Data ─────────────────────────────────────────────────────────────
rbox(ax, 9.9, 3.5, 1.5, 3.3, 'white', '#CCC', lw=1.0, radius=0.12)
txt(ax,10.65,6.55,'Training Data',fs=8.5,fw='bold',col='#333')
txt(ax,10.65,6.2,'Fire (1)',fs=8,col='#E74C3C')
for i in range(4): fire_dot(ax,10.32+i*0.22,5.95,r=0.075)
ax.plot([10.0,11.3],[5.75,5.75],'--',color='#DDD',lw=1,zorder=4)
txt(ax,10.65,5.5,'GAN (0)',fs=8,col=TEAL)
for i in range(4):
    c=plt.Circle((10.32+i*0.22,5.25),0.075,color=TEAL,zorder=8); ax.add_patch(c)
for i in range(4):
    c=plt.Circle((10.32+i*0.22,5.0),0.075,color=TEAL,zorder=8); ax.add_patch(c)
rbox(ax,9.98,4.55,1.35,0.32,'#E8F8F5','#1A7A5E',lw=0.8,radius=0.06)
txt(ax,10.65,4.71,'Balanced sample',fs=7.5,col='#1A7A5E')

# ── Quality Test ──────────────────────────────────────────────────────────────
rbox(ax, 11.5, 3.5, 1.6, 3.3, 'white', '#CCC', lw=1.0, radius=0.12)
txt(ax,12.3,6.55,'Quality Test',fs=8.5,fw='bold',col='#333')
txt(ax,12.3,6.2,'Can classifier tell them apart?',fs=7,col='#555')
fire_q2=[(11.75,5.7),(12.0,5.95),(11.8,5.45),(12.1,5.75)]
gan_q2 =[(12.5,5.6),(12.7,5.85),(12.55,5.35),(12.75,5.55),(12.6,6.0)]
for fx,fy in fire_q2: fire_dot(ax,fx,fy,r=0.07)
for gx,gy in gan_q2:
    c=plt.Circle((gx,gy),0.07,color=TEAL,zorder=8); ax.add_patch(c)
ax.plot([12.25,12.25],[4.9,6.15],'--',color='#1A7A5E',lw=1.5,ls=(0,(4,2)),zorder=6)
rbox(ax,11.58,4.55,1.44,0.65,'#E8F8F5','#1A7A5E',lw=0.8,radius=0.06)
txt(ax,12.3,5.08,'Cannot separate!',fs=7.5,fw='bold',col='#1A7A5E')
txt(ax,12.3,4.82,'Controls = Fire conditions',fs=7,col='#1A7A5E')
txt(ax,12.3,4.62,'→ Realistic controls  ✓',fs=7,fw='bold',col='#1A7A5E')

# ── Outcome ───────────────────────────────────────────────────────────────────
rbox(ax, 13.25, 3.5, 2.3, 3.3, '#E8F8F5', '#1A7A5E', lw=1.5, radius=0.15)
txt(ax,14.4,6.55,'Outcome',fs=10,fw='bold',col='#1A7A5E')
txt(ax,14.4,6.2,'Quality Score',fs=8.5,col='#555')
ax.text(14.4,5.55,'0.50',fontsize=36,color='#1A7A5E',ha='center',va='center',
        fontweight='bold',zorder=5)
txt(ax,14.4,4.9,'AUC  ✓  Near random',fs=9,fw='bold',col='#1A7A5E')
ax.plot([13.4,15.4],[4.72,4.72],color='#A8D5C8',lw=0.8,zorder=4)
txt(ax,14.4,4.52,'Classifier cannot',fs=7.5,col='#666')
txt(ax,14.4,4.28,'tell them apart',fs=7.5,col='#666')
txt(ax,14.4,4.04,'→ controls match',fs=7.5,col='#888')
txt(ax,14.4,3.80,'real fire conditions',fs=7.5,col='#888')

# Arrows between sub-panels (right box)
for xa,xb in [(9.75,9.9),(11.4,11.5),(13.05,13.25)]:
    arrow(ax,(xa+xb)/2,5.15,(xa+xb)/2+0.12,5.15,col='#1A7A5E',lw=1.5,hw=0.1,hl=0.12)

# "vs" label
ax.text(7.97,5.3,'vs',fontsize=14,color='#888',ha='center',va='center',
        fontweight='bold',fontstyle='italic',fontfamily='DejaVu Sans',zorder=10)

# ── Bottom caption ─────────────────────────────────────────────────────────────
txt(ax, 9.2, 0.5,
    'Feature-Space WGAN (GAN) learns the 58-dim environmental signature of non-fire locations, '
    'producing pseudo-absences\nthat are spatially valid and environmentally realistic — '
    'making them indistinguishable from fire locations for classifiers.',
    fs=8.5, col='#555', ha='center')

plt.savefig('../../outputs/fig_infographic_updated.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Done")
