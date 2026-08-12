r"""
Retrain every GAN arm under a single implementation, with seed replication,
inline scoring, and crash-safe resume.

WHY THIS EXISTS
---------------
The 4-seed study of v1_da_gp gave K-SSE 5.0405 +- 0.2245 (range 4.8546-5.3120)
and border fraction 0.2980 +- 0.0130 (range 0.2858-0.3111). The published
gen_gan.py values (3.8235 / 0.2525) sit ~3.6 sd below that mean and outside the
range on both metrics, while seed variance is small (CV ~4.5%). So the gap is
systematic, not noise: train_variants.py does not reproduce gen_gan.py, and the
cause has not been located.

Rather than debug that, this driver retrains EVERY GAN arm under
train_variants.py so the comparison is internally consistent. The published
single-run numbers are then superseded wholesale instead of partially matched.
Random and Heuristic are not retrained -- they are not GAN-based.

A second benefit: every arm gets a seed-variance estimate. The submitted
manuscript reports single runs with no variance at all, which is a reviewer
target on its own. Note what the 4-seed sd already implies -- GP-standard
(3.4459) vs DA-GP (3.8235) is a 0.38 gap against a difference-sd of ~0.32, i.e.
~1.2 sd. That contrast is NOT significant from single runs in either direction.

SEED ISOLATION
--------------
build_background() is seeded at 0, idw_feats at 99, and final feature assignment
at 77 -- all independent of --seed. So the background pool and feature
assignment are identical across seeds and only the training trajectory varies.
That is the correct isolation: it measures training stochasticity, not sampling
stochasticity.

REQUIREMENT
-----------
train_variants.py must accept --seed and apply it to BOTH np.random.seed /
np.random.RandomState AND torch.manual_seed. Verify before a long run:
    python train_variants.py --help | grep -- --seed

SURVIVING A CLOSED SESSION
--------------------------
The previous seed study died because its background job was killed with the
session. Launch detached:

  Linux / macOS:
    setsid nohup python retrain_all.py --data <csv> --out retrain/ \
        --seeds 42 43 44 --threads 8 --resume > retrain/driver.log 2>&1 < /dev/null &
    disown

  Windows PowerShell:
    Start-Process -WindowStyle Hidden python `
      -ArgumentList 'retrain_all.py','--data','<csv>','--out','retrain','--seeds','42','43','44','--threads','8','--resume' `
      -RedirectStandardOutput retrain\driver.log -RedirectStandardError retrain\driver.err

Progress is committed to manifest.csv after every completed cell, so a kill
costs at most one cell (~17 min). Re-run the identical command with --resume to
pick up.

ORDERING
--------
Seed-major: all six arms at seed 42, then all six at seed 43, and so on. If the
run is interrupted you hold complete, internally comparable seed sets rather
than a partial column for every arm.

USAGE
    python retrain_all.py --data <fire csv> --out retrain/ --seeds 42 43 44 --resume
    python retrain_all.py --data <fire csv> --out retrain/ --dry-run
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_variants.py")

# scoring constants -- identical to recompute_spatial.py
R_VALS = np.linspace(0.05, 0.80, 15)
BORDER_DEG = 0.15
BBOX_PAD = 0.20

# label, --arm, env overrides
CELLS = [
    ("clip",         "clip",          {}),
    ("gp_standard",  "gp_standard",   {}),
    ("v1_da_gp",     "v1_da_gp",      {"GP_VALIDITY_PCTL": "90",   "GP_OOB_WEIGHT": "0.1", "GP_KNN": "7"}),
    ("dagp_strict",  "v2_admissible", {"GP_VALIDITY_PCTL": "75",   "GP_OOB_WEIGHT": "0.0", "GP_KNN": "5"}),
    ("dagp_base",    "v2_admissible", {"GP_VALIDITY_PCTL": "90",   "GP_OOB_WEIGHT": "0.1", "GP_KNN": "7"}),
    ("dagp_relaxed", "v2_admissible", {"GP_VALIDITY_PCTL": "97.5", "GP_OOB_WEIGHT": "0.3", "GP_KNN": "15"}),
]

MANIFEST_COLS = ["label", "arm", "seed", "tau_pctl", "omega", "k",
                 "K_SSE_rawK", "K_SSE_L", "border_fraction", "on_lattice_rate",
                 "wall_clock_s", "pts_file", "finished_utc"]


# ---------------------------------------------------------------------------
def k_curve(pts, area):
    t = cKDTree(pts)
    n = len(pts)
    return (t.count_neighbors(t, R_VALS) - n) / n / (n / area)


def score(pts, fire, area, bounds):
    kp, kf = k_curve(pts, area), k_curve(fire, area)
    lo_x, hi_x, lo_y, hi_y = bounds
    b = (((pts[:, 0] - lo_x) < BORDER_DEG) | ((hi_x - pts[:, 0]) < BORDER_DEG) |
         ((pts[:, 1] - lo_y) < BORDER_DEG) | ((hi_y - pts[:, 1]) < BORDER_DEG))
    L = lambda k: np.sqrt(np.maximum(k, 0.0) / np.pi)
    return (float(np.sum((kp - kf) ** 2)),
            float(np.sum((L(kp) - L(kf)) ** 2)),
            float(b.mean()))


def cell_dir(out, label, seed):
    return os.path.join(out, f"{label}_seed{seed}")


def is_complete(out, label, arm, seed):
    d = cell_dir(out, label, seed)
    return (os.path.exists(os.path.join(d, f"meta_{arm}.json")) and
            os.path.exists(os.path.join(d, f"pts_{arm}.npy")))


def load_manifest(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=MANIFEST_COLS)


def append_manifest(path, row):
    """Append one row and flush immediately -- a kill costs at most one cell."""
    df = pd.DataFrame([row], columns=MANIFEST_COLS)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-iter", type=int, default=4500)
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/MKL thread cap per run; 0 leaves the environment alone")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these cell labels")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    man = load_manifest(manifest_path)
    done = set(zip(man.get("label", []), man.get("seed", [])))

    cells = [c for c in CELLS if args.only is None or c[0] in args.only]

    # seed-major: complete comparable sets first
    queue = []
    for seed in args.seeds:
        for label, arm, env in cells:
            if args.resume and (label, seed) in done and is_complete(args.out, label, arm, seed):
                continue
            queue.append((seed, label, arm, env))

    est_min = len(queue) * 17
    print(f"{len(queue)} cells queued ({len(done)} already in manifest).")
    print(f"rough estimate: {est_min} min  (~{est_min/60:.1f} h) at ~17 min/cell")
    for seed, label, arm, env in queue:
        p = f"tau={env.get('GP_VALIDITY_PCTL','-')} w={env.get('GP_OOB_WEIGHT','-')} k={env.get('GP_KNN','-')}"
        print(f"    seed {seed}  {label:14s} arm={arm:14s} {p}")
    if args.dry_run:
        return
    if not os.path.exists(TRAINER):
        sys.exit(f"trainer not found: {TRAINER}")

    df = pd.read_csv(args.data).dropna()
    fire = df[["LONGITUDE", "LATITUDE"]].values
    area = ((fire[:, 0].max() - fire[:, 0].min()) *
            (fire[:, 1].max() - fire[:, 1].min()))
    bounds = (fire[:, 0].min() - BBOX_PAD, fire[:, 0].max() + BBOX_PAD,
              fire[:, 1].min() - BBOX_PAD, fire[:, 1].max() + BBOX_PAD)

    for i, (seed, label, arm, env_over) in enumerate(queue, 1):
        d = cell_dir(args.out, label, seed)
        # a half-written cell from a kill is not trustworthy -- start clean
        if os.path.exists(d) and not is_complete(args.out, label, arm, seed):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

        env = dict(os.environ, **env_over)
        if args.threads > 0:
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[v] = str(args.threads)

        print(f"\n[{i}/{len(queue)}] seed {seed} :: {label} (arm={arm})", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, TRAINER, "--data", args.data, "--out", d,
             "--arm", arm, "--seed", str(seed), "--n-iter", str(args.n_iter)],
            env=env, capture_output=True, text=True)
        dt = time.time() - t0

        if r.returncode != 0 or not is_complete(args.out, label, arm, seed):
            print(f"  FAILED after {dt:.0f}s (rc={r.returncode})", flush=True)
            print("  --- stdout tail ---\n" + (r.stdout or "")[-1500:], flush=True)
            print("  --- stderr tail ---\n" + (r.stderr or "")[-1500:], flush=True)
            with open(os.path.join(d, "FAILED.log"), "w") as fh:
                fh.write((r.stdout or "") + "\n=== STDERR ===\n" + (r.stderr or ""))
            continue

        pts = np.load(os.path.join(d, f"pts_{arm}.npy"))
        ksse_raw, ksse_L, bfrac = score(pts, fire, area, bounds)
        meta = json.load(open(os.path.join(d, f"meta_{arm}.json")))

        # the trainer reports border_fraction too; disagreement means a bounds
        # mismatch between trainer and scorer and must not pass silently
        mb = meta.get("border_fraction")
        if mb is not None and abs(mb - bfrac) > 1e-6:
            print(f"  WARNING border mismatch: trainer {mb:.5f} vs scorer {bfrac:.5f}",
                  flush=True)

        append_manifest(manifest_path, {
            "label": label, "arm": arm, "seed": seed,
            "tau_pctl": env_over.get("GP_VALIDITY_PCTL", ""),
            "omega": env_over.get("GP_OOB_WEIGHT", ""),
            "k": env_over.get("GP_KNN", ""),
            "K_SSE_rawK": ksse_raw, "K_SSE_L": ksse_L,
            "border_fraction": bfrac,
            "on_lattice_rate": meta.get("on_lattice_rate"),
            "wall_clock_s": round(dt, 1),
            "pts_file": os.path.join(d, f"pts_{arm}.npy"),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"  done {dt:.0f}s  K-SSE {ksse_raw:.4f}  border {bfrac:.4f}  "
              f"on_lattice {meta.get('on_lattice_rate')}", flush=True)

    # ---- summary across seeds ----
    man = load_manifest(manifest_path)
    if len(man):
        print("\n=== mean +- sd across seeds ===", flush=True)
        g = man.groupby("label").agg(
            n_seeds=("seed", "nunique"),
            K_SSE_mean=("K_SSE_rawK", "mean"), K_SSE_sd=("K_SSE_rawK", "std"),
            border_mean=("border_fraction", "mean"), border_sd=("border_fraction", "std"),
            on_lattice=("on_lattice_rate", "mean")).reset_index()
        print(g.to_string(index=False), flush=True)
        g.to_csv(os.path.join(args.out, "summary_by_arm.csv"), index=False)
        print(f"\nmanifest : {manifest_path}")
        print(f"summary  : {os.path.join(args.out, 'summary_by_arm.csv')}")
        print("\nReport every arm as mean +- sd over seeds. Do not quote a single run.")


if __name__ == "__main__":
    main()
