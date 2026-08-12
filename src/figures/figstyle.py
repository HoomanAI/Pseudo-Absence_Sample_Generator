"""Shared Elsevier figure style and save/QC helpers."""
from pathlib import Path
import matplotlib as mpl

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.linewidth": 0.9, "grid.alpha": 0.3,
    "savefig.dpi": 300, "figure.dpi": 120,
})

SINGLE = 3.5
DOUBLE = 7.16

def save_figure(fig, stem, outdir):
    """Save vector PDF and review PNG; return a conservative clipping warning."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_boxes = [a.get_tightbbox(renderer) for a in fig.axes if a.get_visible()]
    fig_box = fig.get_window_extent(renderer)
    warning = ""
    if any(b.x0 < fig_box.x0 or b.y0 < fig_box.y0 or b.x1 > fig_box.x1 or b.y1 > fig_box.y1
           for b in axes_boxes):
        warning = "tight-layout expansion required (verified after bbox_inches='tight')"
    fig.savefig(outdir / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    return warning
