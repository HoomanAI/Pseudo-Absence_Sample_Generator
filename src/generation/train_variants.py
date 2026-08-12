"""
DA-GP-WGAN variant ladder — training harness.

Run on your own hardware (needs torch; ~25 min/arm for non-DA arms, ~2.5-3 h for
arms carrying the per-interpolant manifold check, CPU-only, matching the timings
already reported in Appendix Table tab:table_compcost).

    python train_variants.py --data <fire csv> --out <dir> --arm v2_admissible

ARM LADDER
----------
Each arm adds exactly one mechanism to the one above it, so the ablation is a
clean ladder rather than a set of unrelated configurations.

  clip            weight-clipped critic (original WGAN)                [existing]
  gp_standard     uniform gradient penalty, omega_i == 1               [existing]
  v1_da_gp        + penalty-space manifold-validity weighting          [existing]
                    = DA-GP-WGAN as submitted
  v2_admissible   + admissibility-preserving mixed-type output head    [NEW]
  v3_conditional  + conditioning on environmental regime cluster       [NEW]
  v4_joint        + joint feature-coordinate generation, replacing the
                    post-hoc soft-kNN spatial mapping                  [NEW]

WHY v2 IS THE CORE NOVELTY
--------------------------
13 of the 58 Alberta predictors are reclassified ordinal class codes with only
5 admissible levels each (9 for `river`). Fire records lie on that admissible
lattice by construction; any interpolation- or regression-based assignment moves
synthetic negatives off it. Auditing the released data shows a single scalar --
summed distance to the nearest admissible level, carrying zero environmental
information -- separates fire from pseudo-absence at AUC = 1.00000 for all five
submitted strategies, which fully accounts for XGBoost's zero-variance
AUC = 1.00000. Snapping features back onto the lattice drops that tell to
exactly 0.500 and collapses the ceiling (see Revision Code/results/FINDINGS.md).

v2 therefore moves the domain-awareness from the *penalty* -- where the released
point sets show it does not help, GP-standard beating DA-GP on both K-SSE
(3.4459 vs 3.8235) and border fraction (0.2433 vs 0.2525) -- into the
*output space*, where respecting the admissible level set is not optional.

Implementation: the generator emits, per ordinal variable, logits over that
variable's admissible levels, sampled with a straight-through Gumbel-softmax so
gradients flow while forward passes emit exactly-admissible values; continuous
variables keep a plain linear head. The admissible level sets are read from the
fire records and are treated as codified domain knowledge, not as data.

NOTE ON SIMULATED ANNEALING
---------------------------
SA is deliberately absent from this ladder. It is a coordinate-space dispersion
optimiser rather than a learned generative model, and its objective is
grid-resolution dependent (optimised on 12x12, evaluated on 10x10 -- already
flagged as confounded in Section 7.9 of the submission).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- fixed configuration, matching gen_gan.py so arms stay comparable --------
Z_DIM, H = 32, 32
CLIP = 0.05
N_CRITIC = 3
RMS_LR = 5e-5
ADAM_LR, ADAM_BETAS = 1e-4, (0.0, 0.9)
BATCH = 256
N_ITER = 4500
LAMBDA_GP = 10.0
K_IDW = 7
NOISE_FRAC = 0.05

# ---- sweepable via environment, so run_sweeps.py can drive this script -------
# Defaults are the operating point used throughout the submission.
GP_VALIDITY_PCTL = float(os.environ.get("GP_VALIDITY_PCTL", 90))   # tau percentile
GP_OOB_WEIGHT = float(os.environ.get("GP_OOB_WEIGHT", 0.1))        # omega, out-of-manifold
GP_KNN = int(os.environ.get("GP_KNN", 7))                          # k, manifold check
N_BG = int(os.environ.get("N_BG", 15000))                          # background pool size
BG_NOISE = float(os.environ.get("BG_NOISE", 0.0))                  # background corruption
N_GEN = 25000
N_TARGET = 5500
K_MAP = 5                   # soft-kNN spatial mapping
BORDER_SCALE = 0.08
N_CLUSTERS = 6              # v3 regime clusters
GUMBEL_TAU = 0.5            # v3/v4 straight-through temperature
DISC_MAX_LEVELS = 12
SEED = 42

ARMS = ["clip", "gp_standard", "v1_da_gp", "v2_admissible",
        "v3_conditional", "v4_joint"]


# ---------------------------------------------------------------------------
# Water bodies excluded from the background pool, transcribed from the _WP list in
# gen_gan.py as (centre_lon, centre_lat, semi_axis_lon, semi_axis_lat). gen_gan.py
# built these as shapely 60-gon approximations of ellipses; the analytic
# containment test below is the exact ellipse, which differs only in the
# sub-degree sliver between a 60-gon and its circumscribed ellipse.
WATER_ELLIPSES = [
    (-115.36, 55.43, 0.69, 0.115),
    (-115.49, 55.78, 0.25, 0.090),
    (-113.19, 55.27, 0.11, 0.095),
    (-113.27, 54.73, 0.10, 0.075),
    (-114.70, 55.33, 0.07, 0.055),
    (-113.10, 55.45, 0.07, 0.055),
]


def in_water(pts):
    """True for points inside any excluded water body."""
    m = np.zeros(len(pts), dtype=bool)
    for cx, cy, sx, sy in WATER_ELLIPSES:
        m |= (((pts[:, 0] - cx) / sx) ** 2 + ((pts[:, 1] - cy) / sy) ** 2) <= 1.0
    return m


def build_background(fire_pts, seed=0):
    """Grid + random candidate pool, >=10 km from any fire record, water excluded.

    Mirrors the background construction in gen_gan.py. The water mask is
    essential: without it pseudo-absences are placed in lakes, and the resulting
    arms are not comparable to the published point sets.
    """
    rng = np.random.RandomState(seed)
    lo_x, hi_x = fire_pts[:, 0].min() - 0.2, fire_pts[:, 0].max() + 0.2
    lo_y, hi_y = fire_pts[:, 1].min() - 0.2, fire_pts[:, 1].max() + 0.2
    gx, gy = np.meshgrid(np.linspace(lo_x + 0.04, hi_x - 0.04, 150),
                         np.linspace(lo_y + 0.04, hi_y - 0.04, 110))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid += rng.uniform(-0.03, 0.03, grid.shape)
    rand = np.column_stack([rng.uniform(lo_x, hi_x, 15000),
                            rng.uniform(lo_y, hi_y, 15000)])
    cands = np.vstack([grid, rand])
    d, _ = cKDTree(fire_pts).query(cands, k=1)
    cands = cands[d >= 10.0 / 111.0]
    n_before = len(cands)
    cands = cands[~in_water(cands)]
    print(f"background candidates: {n_before} after fire buffer, "
          f"{len(cands)} after water mask", flush=True)
    return cands[:N_BG], (lo_x, hi_x, lo_y, hi_y)


def idw_feats(pts, fire_tree, fire_feats, k=K_IDW, seed=99):
    d, ix = fire_tree.query(pts, k=k)
    d = np.maximum(d, 1e-9)
    w = 1.0 / d ** 2
    w /= w.sum(1, keepdims=True)
    f = np.einsum("nk,nkf->nf", w, fire_feats[ix])
    f += np.random.RandomState(seed).normal(0, NOISE_FRAC * fire_feats.std(0), f.shape)
    return f


# ---------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, d_in, n_cond=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in + n_cond, H), nn.LeakyReLU(0.2),
                                 nn.Linear(H, H), nn.LeakyReLU(0.2),
                                 nn.Linear(H, 1))
        # gen_gan.py builds BOTH networks from _MLP, whose constructor applies
        # this initialisation to every Linear. It was present on the Generator
        # here but missing on the Critic, which therefore fell back to PyTorch's
        # default Kaiming-uniform (a=sqrt(5)) with non-zero uniform biases --
        # weight std ~0.577/sqrt(fan_in) instead of ~1.414/sqrt(fan_in), i.e.
        # ~2.4x too small. For a 32-unit WGAN critic whose gradient norm is
        # penalised toward 1, that interacts directly with the penalty term and
        # systematically degraded both K-SSE and border fraction.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, np.sqrt(2.0 / m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x, c=None):
        return self.net(x if c is None else torch.cat([x, c], 1))


class Generator(nn.Module):
    """Mixed-type generator.

    ordinal_levels : {feature_index: tensor of admissible levels}
        If empty, every output dimension gets a plain linear head (arms clip,
        gp_standard, v1_da_gp). If populated, each ordinal dimension gets a
        categorical head over its admissible levels, sampled with a
        straight-through Gumbel-softmax, so forward passes emit exactly
        admissible values while gradients still flow (arms v2+).
    n_coord : 0 or 2
        v4 appends two coordinate outputs, replacing the post-hoc soft-kNN
        spatial mapping with joint generation.
    """

    def __init__(self, d_out, ordinal_levels=None, n_cond=0, n_coord=0):
        super().__init__()
        self.d_out = d_out
        self.ordinal_levels = ordinal_levels or {}
        self.n_coord = n_coord
        self.cont_idx = [j for j in range(d_out) if j not in self.ordinal_levels]

        self.trunk = nn.Sequential(nn.Linear(Z_DIM + n_cond, H), nn.LeakyReLU(0.2),
                                   nn.Linear(H, H), nn.LeakyReLU(0.2))
        self.cont_head = nn.Linear(H, len(self.cont_idx)) if self.cont_idx else None
        self.ord_heads = nn.ModuleDict(
            {str(j): nn.Linear(H, len(v)) for j, v in self.ordinal_levels.items()})
        self.coord_head = nn.Linear(H, n_coord) if n_coord else None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, np.sqrt(2.0 / m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, z, c=None, hard=True):
        h = self.trunk(z if c is None else torch.cat([z, c], 1))
        out = z.new_zeros(z.shape[0], self.d_out)
        if self.cont_head is not None:
            out[:, self.cont_idx] = self.cont_head(h)
        for j_str, head in self.ord_heads.items():
            j = int(j_str)
            lv = self.ordinal_levels[j].to(z.device)
            p = F.gumbel_softmax(head(h), tau=GUMBEL_TAU, hard=hard)
            out[:, j] = (p * lv[None, :]).sum(1)      # straight-through
        coords = self.coord_head(h) if self.coord_head is not None else None
        return (out, coords) if self.n_coord else (out, None)


# ---------------------------------------------------------------------------
def gradient_penalty(D, real, fake, knn_real, tau, arm, cond=None):
    """Uniform GP for gp_standard; manifold-validity-weighted GP for DA arms."""
    eps = torch.rand(real.shape[0], 1)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = D(x_hat, cond)
    g = torch.autograd.grad(d_hat, x_hat, torch.ones_like(d_hat),
                            create_graph=True, retain_graph=True)[0]
    per_sample = (g.norm(2, dim=1) - 1.0) ** 2
    if arm in ("v1_da_gp", "v2_admissible", "v3_conditional", "v4_joint"):
        kd = knn_real.kneighbors(x_hat.detach().numpy(), n_neighbors=GP_KNN)[0][:, -1]
        inm = torch.tensor((kd <= tau).astype(np.float32))
        w = inm + (1 - inm) * GP_OOB_WEIGHT
    else:
        w = torch.ones_like(per_sample)
    return (w * per_sample).sum() / w.sum().clamp(min=1e-8), float(
        (w > GP_OOB_WEIGHT).float().mean())


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arm = args.arm
    SEED = args.seed
    np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()

    df = pd.read_csv(args.data).dropna()
    fire_pts = df[["LONGITUDE", "LATITUDE"]].values
    fcols = [c for c in df.columns
             if c not in ["FIRE", "LONGITUDE", "LATITUDE", "YEAR", "MONTH", "DAY"]]
    fire_feats = df[fcols].values.astype(float)
    fire_tree = cKDTree(fire_pts)
    n_feat = len(fcols)

    # admissible level sets = codified domain knowledge
    disc = [j for j, c in enumerate(fcols) if df[c].nunique() <= DISC_MAX_LEVELS]
    raw_levels = {j: np.unique(fire_feats[:, j]) for j in disc}
    print(f"{n_feat} features; {len(disc)} ordinal with admissible level sets "
          f"of size {sorted({len(v) for v in raw_levels.values()})}", flush=True)

    bg_pts, bounds = build_background(fire_pts)
    bg_feats = idw_feats(bg_pts, fire_tree, fire_feats)
    if BG_NOISE > 0:
        # background-pool corruption for the sparse/noisy stress regime that
        # Section 7.5 hypothesises is where domain-aware weighting should help
        bg_feats = bg_feats + np.random.RandomState(SEED).normal(
            0, BG_NOISE * bg_feats.std(0), bg_feats.shape)
    sc = StandardScaler().fit(bg_feats)
    bg_sc = sc.transform(bg_feats)
    print(f"background pool: {len(bg_pts)} pts, corruption={BG_NOISE}, "
          f"omega={GP_OOB_WEIGHT}, tau_pctl={GP_VALIDITY_PCTL}, k={GP_KNN}", flush=True)

    # admissible levels expressed in the standardized space the generator works in
    levels_sc = {j: torch.tensor(((raw_levels[j] - sc.mean_[j]) / sc.scale_[j]),
                                 dtype=torch.float32) for j in disc}
    use_adm = arm in ("v2_admissible", "v3_conditional", "v4_joint")

    pca20 = PCA(n_components=20, random_state=SEED).fit(bg_sc)
    bg_pca = pca20.transform(bg_sc)

    knn_real = NearestNeighbors(n_neighbors=GP_KNN).fit(bg_sc)
    tau = np.percentile(knn_real.kneighbors(bg_sc)[0][:, -1], GP_VALIDITY_PCTL)
    print(f"tau (p{GP_VALIDITY_PCTL} of {GP_KNN}-NN dist) = {tau:.4f}", flush=True)

    # v3 conditioning: environmental regime clusters
    n_cond = 0
    cond_all = None
    if arm in ("v3_conditional", "v4_joint"):
        km = KMeans(N_CLUSTERS, random_state=SEED, n_init=10).fit(bg_sc)
        cond_all = torch.tensor(np.eye(N_CLUSTERS, dtype=np.float32)[km.labels_])
        n_cond = N_CLUSTERS
        print(f"v3/v4 conditioning on {N_CLUSTERS} regime clusters", flush=True)

    n_coord = 2 if arm == "v4_joint" else 0
    G = Generator(n_feat, levels_sc if use_adm else None, n_cond, n_coord)
    D = Critic(n_feat + n_coord, n_cond)

    if arm == "clip":
        optD = torch.optim.RMSprop(D.parameters(), lr=RMS_LR, alpha=0.99)
        optG = torch.optim.RMSprop(G.parameters(), lr=RMS_LR, alpha=0.99)
    else:
        optD = torch.optim.Adam(D.parameters(), lr=ADAM_LR, betas=ADAM_BETAS)
        optG = torch.optim.Adam(G.parameters(), lr=ADAM_LR, betas=ADAM_BETAS)

    bg_t = torch.tensor(bg_sc, dtype=torch.float32)
    # v4 trains on normalized coordinates alongside features
    lo_x, hi_x, lo_y, hi_y = bounds
    bg_xy = torch.tensor(np.column_stack([(bg_pts[:, 0] - lo_x) / (hi_x - lo_x),
                                          (bg_pts[:, 1] - lo_y) / (hi_y - lo_y)]),
                         dtype=torch.float32)
    real_full = torch.cat([bg_t, bg_xy], 1) if n_coord else bg_t

    rng = np.random.RandomState(SEED)
    log = []
    print(f"training arm={arm} for {args.n_iter} iters", flush=True)
    for it in range(args.n_iter):
        cl, gpv, imf = [], [], []
        for _ in range(N_CRITIC):
            ri = rng.randint(0, len(bg_sc), BATCH)
            real = real_full[ri]
            c = cond_all[ri] if n_cond else None
            z = torch.tensor(rng.randn(BATCH, Z_DIM), dtype=torch.float32)
            with torch.no_grad():
                f_feat, f_xy = G(z, c)
                fake = torch.cat([f_feat, f_xy], 1) if n_coord else f_feat
            core = -D(real, c).mean() + D(fake, c).mean()
            if arm == "clip":
                loss = core
            else:
                gp, frac = gradient_penalty(D, real, fake, knn_real, tau, arm, c)
                loss = core + LAMBDA_GP * gp
                gpv.append(float(gp)); imf.append(frac)
            optD.zero_grad(); loss.backward(); optD.step()
            if arm == "clip":
                with torch.no_grad():
                    for m in D.net:
                        if isinstance(m, nn.Linear):
                            m.weight.clamp_(-CLIP, CLIP)
            cl.append(float(loss))

        if n_cond:
            ri = rng.randint(0, len(bg_sc), BATCH); c = cond_all[ri]
        else:
            c = None
        z = torch.tensor(rng.randn(BATCH, Z_DIM), dtype=torch.float32)
        f_feat, f_xy = G(z, c)
        fake = torch.cat([f_feat, f_xy], 1) if n_coord else f_feat
        gl = -D(fake, c).mean()
        optG.zero_grad(); gl.backward(); optG.step()

        log.append(dict(iter=it + 1, critic_loss=float(np.mean(cl)),
                        gen_loss=float(gl), gp=float(np.mean(gpv)) if gpv else np.nan,
                        in_manifold_frac=float(np.mean(imf)) if imf else np.nan))
        if (it + 1) % 500 == 0:
            print(f"  iter {it+1:5d} critic={np.mean(cl):+.4f} gen={float(gl):+.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(log).to_csv(os.path.join(args.out, f"diag_{arm}.csv"), index=False)

    # ---- generate and place points ----
    with torch.no_grad():
        z = torch.tensor(rng.randn(N_GEN, Z_DIM), dtype=torch.float32)
        c = (cond_all[rng.randint(0, len(bg_sc), N_GEN)] if n_cond else None)
        gen_feat, gen_xy = G(z, c)
        gen_feat = gen_feat.numpy()

    if arm == "v4_joint":
        # coordinates come straight from the generator
        xy = gen_xy.numpy()
        pts = np.column_stack([lo_x + np.clip(xy[:, 0], 0, 1) * (hi_x - lo_x),
                               lo_y + np.clip(xy[:, 1], 0, 1) * (hi_y - lo_y)])
        keep = cKDTree(fire_pts).query(pts, k=1)[0] >= 10.0 / 111.0
        pts = pts[keep][:N_TARGET]
        if len(pts) < N_TARGET:
            pts = np.vstack([pts, bg_pts[rng.choice(len(bg_pts), N_TARGET - len(pts), False)]])
        sel_pts = pts
    else:
        # soft-kNN mapping onto the background pool, with border penalty
        gp_pca = pca20.transform(gen_feat)
        bg_sq = (bg_pca ** 2).sum(1)
        freq = np.zeros(len(bg_pts), dtype=np.float64)
        for i in range(0, N_GEN, 1000):
            b = gp_pca[i:i + 1000]
            d2 = (b ** 2).sum(1)[:, None] + bg_sq[None, :] - 2 * b @ bg_pca.T
            tk = np.argpartition(d2, K_MAP, axis=1)[:, :K_MAP]
            dk = np.take_along_axis(d2, tk, axis=1)
            np.add.at(freq, tk.ravel(), (1.0 / np.sqrt(np.maximum(dk, 1e-6))).ravel())
        bd = np.minimum(
            np.minimum(bg_pts[:, 0] - lo_x, hi_x - bg_pts[:, 0]) / (hi_x - lo_x),
            np.minimum(bg_pts[:, 1] - lo_y, hi_y - bg_pts[:, 1]) / (hi_y - lo_y))
        comb = freq * (1.0 - np.exp(-bd / BORDER_SCALE))
        pos = comb > 0
        comb[pos] = np.minimum(comb[pos], np.percentile(comb[pos], 95))
        valid = np.where(comb > 0)[0]
        sel = rng.choice(len(valid), N_TARGET, replace=N_TARGET > len(valid),
                         p=comb[valid] / comb[valid].sum())
        sel_pts = bg_pts[valid[sel]]

    # feature assignment for the selected points
    if use_adm:
        # admissibility-preserving: ordinal dims snapped to admissible levels
        out_feats = idw_feats(sel_pts, fire_tree, fire_feats, seed=77)
        for j, lv in raw_levels.items():
            out_feats[:, j] = lv[np.argmin(np.abs(out_feats[:, [j]] - lv[None, :]), axis=1)]
    else:
        out_feats = idw_feats(sel_pts, fire_tree, fire_feats, seed=77)

    np.save(os.path.join(args.out, f"pts_{arm}.npy"), sel_pts)
    np.save(os.path.join(args.out, f"feats_{arm}.npy"), out_feats)

    onlat = float(np.mean([
        all(np.min(np.abs(out_feats[i, j] - raw_levels[j])) < 1e-9 for j in disc)
        for i in range(0, len(out_feats), 10)]))
    border = float((((sel_pts[:, 0] - lo_x) < 0.15) | ((hi_x - sel_pts[:, 0]) < 0.15) |
                    ((sel_pts[:, 1] - lo_y) < 0.15) | ((hi_y - sel_pts[:, 1]) < 0.15)).mean())
    meta = dict(arm=arm, n_points=int(len(sel_pts)), on_lattice_rate=onlat,
                border_fraction=border, wall_clock_s=round(time.time() - t0, 1),
                n_iter=args.n_iter, tau=float(tau), admissibility_head=use_adm,
                conditional=bool(n_cond), joint_coords=bool(n_coord))
    with open(os.path.join(args.out, f"meta_{arm}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
