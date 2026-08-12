"""Round-8 presence-fitted PCA density-contour figure."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
RETRAIN = CODE / "retrain"
OUT = ROOT / "Overleaf 2" / "figures"
PROV = ROOT / "FIGURE_DATA.csv"
FIRECSV = ROOT / "Pseudo-Absence_Sample_Generator" / "data" / "Fire_points_dataset_final_csv.csv"
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, save_figure

STEM = "F34_pca_density_contours"
FIGURE_ID = "F34"
ARMS = ["random", "heuristic", "clip", "gp_standard", "v1_da_gp",
        "dagp_strict", "dagp_base", "dagp_relaxed"]
LABEL = {
    "random": "Random", "heuristic": "Heuristic", "clip": "GAN (clip)",
    "gp_standard": "WGAN-GP", "v1_da_gp": "DA-GP-WGAN",
    "dagp_strict": "AP-WGAN (strict)", "dagp_base": "AP-WGAN (base)",
    "dagp_relaxed": "AP-WGAN (relaxed)",
}
COLOR = dict(zip(ARMS, ["#7F7F7F", "#8C564B", "#D55E00", "#CC79A7",
                        "#009E73", "#56B4E9", "#0072B2", "#7BAFD4"]))
META = {"FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"}


def resolve_feature_arrays():
    """Resolve feature filenames from each manifest pts_file; never guess run paths."""
    manifest = pd.read_csv(RETRAIN / "manifest.csv")
    rows = manifest[(manifest.seed == 42) & manifest.label.isin(ARMS)].set_index("label")
    if set(rows.index) != set(ARMS):
        raise RuntimeError("manifest.csv does not contain exactly the eight required seed-42 arms")
    resolved = {}
    for arm in ARMS:
        row = rows.loc[arm]
        pts = CODE / Path(str(row.pts_file).replace("\\", "/"))
        feats = pts.with_name(f"feats_{row.arm}.npy")
        if not pts.exists() or not feats.exists():
            raise FileNotFoundError(f"Manifest-resolved files missing for {arm}: {pts}, {feats}")
        resolved[arm] = (feats, np.load(feats).astype(float, copy=False))
    return resolved


def append_provenance(records):
    """Replace prior F34 records atomically while preserving the large shared ledger."""
    header = ["figure", "source_file", "source_column", "arm", "seed", "series", "value"]
    PROV.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="figure_data_", suffix=".csv", dir=PROV.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst)
            writer.writerow(header)
            if PROV.exists():
                with PROV.open("r", newline="", encoding="utf-8-sig") as src:
                    reader = csv.DictReader(src)
                    for row in reader:
                        if row.get("figure") != FIGURE_ID:
                            writer.writerow([row.get(k, "") for k in header])
            writer.writerows(records)
        os.replace(temp, PROV)
    finally:
        if temp.exists():
            temp.unlink()


def build():
    fire_df = pd.read_csv(FIRECSV).dropna()
    feature_cols = [c for c in fire_df.columns if c not in META]
    presence = fire_df[feature_cols].to_numpy(float)
    arrays = resolve_feature_arrays()
    for arm, (_, values) in arrays.items():
        if values.shape[1] != presence.shape[1]:
            raise ValueError(f"{arm} has {values.shape[1]} features; presence has {presence.shape[1]}")

    scaler = StandardScaler().fit(presence)
    pca = PCA(n_components=2, random_state=42).fit(scaler.transform(presence))
    projected_presence = pca.transform(scaler.transform(presence))
    projected = {arm: pca.transform(scaler.transform(values)) for arm, (_, values) in arrays.items()}

    all_points = np.vstack([projected_presence, *projected.values()])
    lo = all_points.min(axis=0); hi = all_points.max(axis=0)
    margin = np.maximum((hi - lo) * 0.05, 1e-6)
    gx = np.linspace(lo[0] - margin[0], hi[0] + margin[0], 120)
    gy = np.linspace(lo[1] - margin[1], hi[1] + margin[1], 120)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    presence_density = gaussian_kde(projected_presence.T)(grid).reshape(xx.shape)
    positive = presence_density[presence_density > 0]
    levels = np.unique(np.quantile(positive, [0.60, 0.75, 0.875, 0.95]))
    if len(levels) < 3:
        raise RuntimeError("Presence KDE did not yield enough distinct shared contour levels")

    fig, axs = plt.subplots(4, 2, figsize=(DOUBLE, 1.35 * DOUBLE), sharex=True, sharey=True)
    arm_density = {}
    for index, (arm, ax) in enumerate(zip(ARMS, axs.flat)):
        density = gaussian_kde(projected[arm].T)(grid).reshape(xx.shape)
        arm_density[arm] = density
        ax.scatter(projected_presence[:, 0], projected_presence[:, 1], s=4, c="0.55",
                   alpha=0.25, edgecolors="none", rasterized=True, zorder=1)
        ax.contour(xx, yy, presence_density, levels=levels, colors="black",
                   linewidths=1.25, linestyles="solid", zorder=2)
        available = levels[levels < density.max()]
        if len(available):
            ax.contour(xx, yy, density, levels=available, colors=COLOR[arm],
                       linewidths=0.95, linestyles="solid", zorder=3)
        ax.text(0.02, 0.98, LABEL[arm], transform=ax.transAxes, ha="left", va="top",
                fontsize=7, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.76), zorder=4)
        ax.text(0.0, 1.02, chr(97 + index), transform=ax.transAxes, fontweight="bold",
                ha="left", va="bottom", clip_on=False)
        ax.tick_params(labelbottom=index >= 6, labelleft=index % 2 == 0)

    variance = pca.explained_variance_ratio_ * 100
    fig.supxlabel(f"PC1 ({variance[0]:.1f}% variance)", y=0.055)
    fig.supylabel(f"PC2 ({variance[1]:.1f}% variance)", x=0.025)
    handles = [Line2D([0], [0], color="black", lw=1.25, label="Presence reference"),
               Line2D([0], [0], color="0.35", lw=0.95, label="Pseudo-absence arm")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.11, top=0.975,
                        wspace=0.18, hspace=0.14)
    warning = save_figure(fig, STEM, OUT)
    plt.close(fig)

    records = []
    def rec(src, col, arm, series, value, seed="42"):
        records.append([FIGURE_ID, str(src).replace("\\", "/"), col, arm, seed, series, float(value)])
    for i, point in enumerate(projected_presence):
        rec(FIRECSV.relative_to(ROOT), "PC1", "presence", f"point={i}", point[0], "")
        rec(FIRECSV.relative_to(ROOT), "PC2", "presence", f"point={i}", point[1], "")
    for arm in ARMS:
        source = arrays[arm][0].relative_to(ROOT)
        for i, point in enumerate(projected[arm]):
            rec(source, "PC1", arm, f"point={i}", point[0])
            rec(source, "PC2", arm, f"point={i}", point[1])
    for i, value in enumerate(levels):
        rec(FIRECSV.relative_to(ROOT), "presence_KDE_density", "presence", f"shared_contour_level={i}", value, "")
    for i, value in enumerate(pca.explained_variance_ratio_):
        rec(FIRECSV.relative_to(ROOT), "explained_variance_ratio", "presence", f"PC{i+1}", value, "")
    append_provenance(records)
    return arrays, levels, variance, (gx[0], gx[-1], gy[0], gy[-1]), warning, len(records)


def write_report(arrays, levels, variance, bounds, warning, record_count):
    paths = "\n".join(f"- {LABEL[a]}: `{arrays[a][0].relative_to(ROOT).as_posix()}`" for a in ARMS)
    report = f"""# Figure report 8

## F34_pca_density_contours

- Presence source: `{FIRECSV.relative_to(ROOT).as_posix()}` ({len(pd.read_csv(FIRECSV).dropna()):,} complete presence records; all covariates except FIRE, coordinates, and date fields).
- Seed: 42 only.
- Caption handoff: state explicitly that the plotted pseudo-absence arms use seed 42 only (the generator does not edit manuscript `.tex` files).
- Arm feature arrays, resolved from each seed-42 `pts_file` entry in `Revision Code/retrain/manifest.csv`:
{paths}
- PCA method: `StandardScaler` and the two-component PCA were fitted once on the presence covariate matrix only. Every arm was standardized and projected with that same fitted transformation; PCA was not refitted per arm.
- Explained variance: PC1 = {variance[0]:.4f}%; PC2 = {variance[1]:.4f}% (outer labels display one decimal place).
- KDE method: `scipy.stats.gaussian_kde`, evaluated on one shared 120 x 120 grid spanning the union of presence and all eight projected arms plus a 5% margin. Grid bounds: PC1 [{bounds[0]:.8g}, {bounds[1]:.8g}], PC2 [{bounds[2]:.8g}, {bounds[3]:.8g}].
- Shared contour levels: presence-density grid quantiles 60%, 75%, 87.5%, and 95%, computed once; exact density levels = {', '.join(f'{x:.12g}' for x in levels)}. These identical absolute levels were used for every panel, never selected per arm.
- Figure size: `figsize=({DOUBLE}, {1.35*DOUBLE:.3f})` inches; 4 rows x 2 columns, all eight arms retained.
- Artwork contains contour lines only: faint presence scatter, black presence reference, and the panel arm in its fixed colour. No fill and no in-figure title.
- Provenance: {record_count:,} idempotent F34 rows written to `FIGURE_DATA.csv`, including all projected points, shared levels, and explained-variance ratios.
- Verification: plotted values were loaded from the listed arrays; none were fabricated or altered. Save/QC warning: {warning or 'none'}.
"""
    (ROOT / "FIGURE_REPORT_8.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    write_report(*build())
    print(f"Generated {OUT / (STEM + '.pdf')} and .png")


if __name__ == "__main__":
    main()
