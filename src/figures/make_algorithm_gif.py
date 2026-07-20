"""
Animated "dots on a map" walkthrough of the pseudo-absence generation pipeline
(src/generation/gen_gan.py): fire presence points -> background candidate pool
-> WGAN-driven, border-aware selection -> final pseudo-absence set.

Reads data/Fire_points_dataset_final_csv.csv and outputs/gan_pts_<mode>.npy
(produced by gen_gan.py) and writes docs/algorithm_animation.gif.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
import warnings; warnings.filterwarnings('ignore')

GAN_MODE = 'clip'  # which outputs/gan_pts_<mode>.npy to illustrate
rng = np.random.RandomState(7)

# -- Recreate the same fire points / background pool as gen_gan.py (same seeds) --
df = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts = df[['LONGITUDE', 'LATITUDE']].values
fire_tree = cKDTree(fire_pts)
DEG = 1 / 111.0; D_MIN = 10 * DEG
lon_min, lon_max = fire_pts[:, 0].min() - 0.2, fire_pts[:, 0].max() + 0.2
lat_min, lat_max = fire_pts[:, 1].min() - 0.2, fire_pts[:, 1].max() + 0.2

def _ep(cx, cy, sx, sy, n=60):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(zip(cx + sx * np.cos(t), cy + sy * np.sin(t)))
_WP = [_ep(-115.36, 55.43, 0.69, 0.115), _ep(-115.49, 55.78, 0.25, 0.090),
       _ep(-113.19, 55.27, 0.11, 0.095), _ep(-113.27, 54.73, 0.10, 0.075),
       _ep(-114.70, 55.33, 0.07, 0.055), _ep(-113.10, 55.45, 0.07, 0.055)]

rng0 = np.random.RandomState(0)
lons = np.linspace(lon_min + 0.04, lon_max - 0.04, 150)
lats = np.linspace(lat_min + 0.04, lat_max - 0.04, 110)
gx, gy = np.meshgrid(lons, lats)
grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
grid_pts += rng0.uniform(-0.03, 0.03, grid_pts.shape)
rand_pts = np.column_stack([rng0.uniform(lon_min, lon_max, 15000),
                             rng0.uniform(lat_min, lat_max, 15000)])
all_cands = np.vstack([grid_pts, rand_pts])
dd, _ = fire_tree.query(all_cands, k=1)
cands = all_cands[dd >= D_MIN]
bg_pts = cands[:15000]

# thin the background pool for a legible animation frame
bg_show = bg_pts[rng.choice(len(bg_pts), size=3500, replace=False)]

gan_pts = np.load(f'../../outputs/gan_pts_{GAN_MODE}.npy')
reveal_order = rng.permutation(len(gan_pts))

# -- Figure / map chrome --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=100)
ax.set_facecolor('#eaf2e6')  # pale land
for poly in _WP:
    xs, ys = poly.exterior.xy
    ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True,
                             facecolor='#a9cbe8', edgecolor='none', zorder=1))
ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
title = ax.set_title('', fontsize=12, fontweight='bold')
caption = ax.text(0.5, -0.13, '', transform=ax.transAxes, ha='center',
                   fontsize=9.5, color='#444444')

sc_bg = ax.scatter([], [], s=4, c='#9a9a9a', alpha=0.0, zorder=2, label='Background pool')
sc_fire = ax.scatter(fire_pts[:, 0], fire_pts[:, 1], s=5, c='#d62728', alpha=0.0,
                      zorder=4, label='Fire presence points')
sc_gan = ax.scatter([], [], s=6, c='#1f6fd6', alpha=0.0, zorder=3, label='Pseudo-absence points')
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

# -- Stage schedule (frame ranges) ----------------------------------------------
F_FIRE_IN, F_FIRE_HOLD = 12, 10
F_BG_IN, F_BG_HOLD = 12, 10
F_REVEAL = 40
F_FINAL_HOLD = 20
frames = F_FIRE_IN + F_FIRE_HOLD + F_BG_IN + F_BG_HOLD + F_REVEAL + F_FINAL_HOLD

def ease(t): return t * t * (3 - 2 * t)  # smoothstep

def update(f):
    i = f
    if i < F_FIRE_IN:
        a = ease(i / F_FIRE_IN)
        sc_fire.set_alpha(a)
        title.set_text('Step 1 — Historical fire presence points')
        caption.set_text(f'{len(fire_pts):,} known fire locations (input)')
    elif i < F_FIRE_IN + F_FIRE_HOLD:
        sc_fire.set_alpha(1.0)
        title.set_text('Step 1 — Historical fire presence points')
        caption.set_text(f'{len(fire_pts):,} known fire locations (input)')
    elif i < F_FIRE_IN + F_FIRE_HOLD + F_BG_IN:
        j = i - (F_FIRE_IN + F_FIRE_HOLD)
        a = ease(j / F_BG_IN)
        sc_bg.set_offsets(bg_show); sc_bg.set_alpha(0.35 * a)
        title.set_text('Step 2 — Candidate background pool')
        caption.set_text('Non-fire locations kept ≥10 km from any fire point')
    elif i < F_FIRE_IN + F_FIRE_HOLD + F_BG_IN + F_BG_HOLD:
        sc_bg.set_offsets(bg_show); sc_bg.set_alpha(0.35)
        title.set_text('Step 2 — Candidate background pool')
        caption.set_text('Non-fire locations kept ≥10 km from any fire point')
    elif i < F_FIRE_IN + F_FIRE_HOLD + F_BG_IN + F_BG_HOLD + F_REVEAL:
        j = i - (F_FIRE_IN + F_FIRE_HOLD + F_BG_IN + F_BG_HOLD)
        a = ease(j / F_REVEAL)
        n_show = int(a * len(gan_pts))
        idx = reveal_order[:n_show]
        sc_gan.set_offsets(gan_pts[idx] if n_show else np.empty((0, 2)))
        sc_gan.set_alpha(0.75)
        sc_bg.set_alpha(0.35 * (1 - 0.6 * a))
        title.set_text('Step 3 — Feature-space WGAN selects pseudo-absences')
        caption.set_text('Generated feature vectors mapped to background locations\n'
                          'via kNN + border-aware weighting (no coordinate generation)')
    else:
        sc_gan.set_offsets(gan_pts); sc_gan.set_alpha(0.8)
        sc_bg.set_alpha(0.12)
        title.set_text('Step 4 — Final pseudo-absence set')
        caption.set_text(f'{len(gan_pts):,} points, spatially spread across the study area')
    return sc_bg, sc_fire, sc_gan, title, caption

anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
anim.save('../../docs/algorithm_animation.gif', writer=animation.PillowWriter(fps=10))
print('Saved docs/algorithm_animation.gif')
