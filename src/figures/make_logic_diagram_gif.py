"""
Simple 4-node schematic (circles + arrows) of the feature-space WGAN pipeline
logic used by src/generation/gen_gan.py, as opposed to the geographic
"dots on a map" animation in make_algorithm_gif.py.

Noise z -> Generator G -> Feature vector -> kNN match -> Geographic location

Writes docs/algorithm_logic.gif.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch

NODES = [
    (1.0, 0, 'Noise\nz ∈ R³²', '#6b7280'),
    (3.3, 0, 'Generator\nG(z)', '#1f6fd6'),
    (5.6, 0, 'Feature vector\n(58-dim)', '#2e9e5b'),
    (7.9, 0, 'kNN match →\ngeographic point', '#d62728'),
]
R = 0.62
ARROWS = [(0, 1), (1, 2), (2, 3)]

fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=110)
ax.set_xlim(-0.2, 9.3); ax.set_ylim(-1.5, 1.5)
ax.axis('off')

circles, labels = [], []
for x, y, text, color in NODES:
    c = Circle((x, y), R, facecolor='white', edgecolor=color, linewidth=2.5, zorder=3)
    ax.add_patch(c)
    t = ax.text(x, y, text, ha='center', va='center', fontsize=8.2, fontweight='bold',
                color=color, zorder=4)
    circles.append(c); labels.append(t)

arrow_patches = []
for i, j in ARROWS:
    x0, y0, *_ = NODES[i]; x1, y1, *_ = NODES[j]
    ap = FancyArrowPatch((x0 + R, y0), (x1 - R, y1), arrowstyle='-|>', mutation_scale=16,
                          linewidth=2, color='#c8c8c8', zorder=2)
    ax.add_patch(ap)
    arrow_patches.append(ap)

pulse = Circle((NODES[0][0], NODES[0][1]), 0.14, facecolor='#f5a623', edgecolor='none', zorder=5)
ax.add_patch(pulse)
pulse.set_alpha(0)

caption = ax.text(4.55, -1.25, '', ha='center', fontsize=9.5, color='#444444')

STAGE_TEXT = [
    'Sample random noise z',
    'Generator maps noise to a synthetic environmental feature vector',
    'Critic-trained (WGAN-GP) generator output is compared to real background features',
    'Feature vector is matched via kNN to a real background location → pseudo-absence point',
]

HOLD = 10
TRAVEL = 14
per_stage = HOLD + TRAVEL
n_arrows = len(ARROWS)
frames = HOLD + n_arrows * per_stage + HOLD  # initial hold, 3 travel+hold, final hold

def ease(t): return t * t * (3 - 2 * t)

def reset_colors():
    for k, (x, y, text, color) in enumerate(NODES):
        circles[k].set_edgecolor(color)
        circles[k].set_linewidth(2.5)
    for ap in arrow_patches:
        ap.set_color('#c8c8c8')

def update(f):
    reset_colors()
    if f < HOLD:
        pulse.set_alpha(0)
        circles[0].set_linewidth(4)
        caption.set_text(STAGE_TEXT[0])
        return circles + labels + arrow_patches + [pulse, caption]

    g = f - HOLD
    stage = min(g // per_stage, n_arrows - 1)
    within = g - stage * per_stage
    i, j = ARROWS[stage]
    x0, y0, *_ = NODES[i]; x1, y1, *_ = NODES[j]

    if within < TRAVEL:
        t = ease(within / TRAVEL)
        px = x0 + R + t * ((x1 - R) - (x0 + R))
        pulse.set_center((px, y0))
        pulse.set_alpha(1.0)
        arrow_patches[stage].set_color('#f5a623')
        circles[i].set_edgecolor('#f5a623'); circles[i].set_linewidth(4)
        caption.set_text(STAGE_TEXT[stage + 1])
    else:
        pulse.set_alpha(0)
        circles[j].set_edgecolor('#f5a623'); circles[j].set_linewidth(4)
        for s in range(stage + 1):
            arrow_patches[s].set_color('#f5a623')
        caption.set_text(STAGE_TEXT[stage + 1])

    return circles + labels + arrow_patches + [pulse, caption]

anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
anim.save('../../docs/algorithm_logic.gif', writer=animation.PillowWriter(fps=14))
print('Saved docs/algorithm_logic.gif')
