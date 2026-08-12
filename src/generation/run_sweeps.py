"""
Sweep harness for the domain-aware gradient penalty hyperparameters.

Section 7.5 / 7.9 of the submission list tau, the out-of-manifold weight omega,
and k as un-swept, each requiring retraining. This script runs those sweeps and
also runs the specific stress test that Section 7.5 hypothesises is where
domain-aware weighting should pay off -- "noisier/sparser background pools,
higher-dimensional features, critics prone to drifting off-manifold" -- by
shrinking and corrupting the background pool.

Run on your own hardware. Each cell is one full training run: budget roughly
25 min for uniform-GP cells and 2.5-3 h for domain-aware cells on CPU, matching
Appendix Table tab:table_compcost. Use --dry-run first to see the cell count.

    python run_sweeps.py --data <fire csv> --out sweeps/ --sweep tau --dry-run
    python run_sweeps.py --data <fire csv> --out sweeps/ --sweep tau
    python run_sweeps.py --data <fire csv> --out sweeps/ --sweep all --resume

Results are appended to <out>/sweep_results.csv after every cell, so the run is
interruptible and resumable. Cells already present in that file are skipped
when --resume is passed.

WHAT IS SWEPT
-------------
  tau      validity-radius percentile: 50, 75, 90 (submitted), 97.5, 99
  omega    out-of-manifold weight:     0.0 (hard), 0.05, 0.1 (submitted), 0.3, 1.0
                                       omega = 1.0 is exactly GP-standard, so the
                                       sweep contains its own null hypothesis
  k        manifold-check neighbours:  3, 5, 7 (submitted), 15, 30
  stress   background pool size x noise: {15000, 5000, 1500} x {0.0, 0.1, 0.3}
                                       with omega in {0.1, 1.0} so the
                                       domain-aware vs uniform contrast is
                                       measured in every regime

Each cell reports K-SSE and border fraction against the fire reference, the
on-lattice rate, and wall-clock time.
"""
import argparse, itertools, json, os, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_variants.py")

TAU_GRID = [50.0, 75.0, 90.0, 97.5, 99.0]
OMEGA_GRID = [0.0, 0.05, 0.1, 0.3, 1.0]
K_GRID = [3, 5, 7, 15, 30]
POOL_GRID = [15000, 5000, 1500]
NOISE_GRID = [0.0, 0.1, 0.3]
STRESS_OMEGA = [0.1, 1.0]

DEFAULTS = dict(tau=90.0, omega=0.1, k=7, pool=15000, noise=0.0)


def cells(which):
    out = []
    if which in ("tau", "all"):
        for t in TAU_GRID:
            out.append({**DEFAULTS, "tau": t, "sweep": "tau"})
    if which in ("omega", "all"):
        for w in OMEGA_GRID:
            out.append({**DEFAULTS, "omega": w, "sweep": "omega"})
    if which in ("k", "all"):
        for k in K_GRID:
            out.append({**DEFAULTS, "k": k, "sweep": "k"})
    if which in ("stress", "all"):
        for p, n, w in itertools.product(POOL_GRID, NOISE_GRID, STRESS_OMEGA):
            out.append({**DEFAULTS, "pool": p, "noise": n, "omega": w,
                        "sweep": "stress"})
    # de-duplicate identical configurations across sweeps
    seen, uniq = set(), []
    for c in out:
        key = (c["tau"], c["omega"], c["k"], c["pool"], c["noise"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def cell_id(c):
    return f"tau{c['tau']}_w{c['omega']}_k{c['k']}_pool{c['pool']}_noise{c['noise']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", default="all",
                    choices=["tau", "omega", "k", "stress", "all"])
    ap.add_argument("--n-iter", type=int, default=4500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "sweep_results.csv")

    todo = cells(args.sweep)
    done = set()
    if args.resume and os.path.exists(res_path):
        done = set(pd.read_csv(res_path)["cell_id"].tolist())
        todo = [c for c in todo if cell_id(c) not in done]

    da_cells = sum(1 for c in todo if c["omega"] < 1.0)
    est_h = (da_cells * 2.75 + (len(todo) - da_cells) * 0.42)
    print(f"{len(todo)} cells to run ({len(done)} already done). "
          f"{da_cells} carry the manifold check.")
    print(f"rough estimate: {est_h:.1f} h CPU-only at {args.n_iter} iters/cell")
    for c in todo:
        print("   ", cell_id(c), "|", c["sweep"])
    if args.dry_run:
        return

    if not os.path.exists(TRAINER):
        sys.exit(f"trainer not found: {TRAINER}")

    for i, c in enumerate(todo, 1):
        cid = cell_id(c)
        cdir = os.path.join(args.out, cid)
        os.makedirs(cdir, exist_ok=True)
        env = dict(os.environ,
                   GP_VALIDITY_PCTL=str(c["tau"]),
                   GP_OOB_WEIGHT=str(c["omega"]),
                   GP_KNN=str(c["k"]),
                   N_BG=str(c["pool"]),
                   BG_NOISE=str(c["noise"]))
        arm = "v1_da_gp" if c["omega"] < 1.0 else "gp_standard"
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {cid}  arm={arm}", flush=True)
        r = subprocess.run(
            [sys.executable, TRAINER, "--data", args.data, "--out", cdir,
             "--arm", arm, "--n-iter", str(args.n_iter)],
            env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            print(f"  FAILED {cid}, continuing", flush=True)
            continue

        meta_path = os.path.join(cdir, f"meta_{arm}.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        row = {"cell_id": cid, "sweep": c["sweep"], "arm": arm,
               "tau_pctl": c["tau"], "omega": c["omega"], "k": c["k"],
               "pool": c["pool"], "bg_noise": c["noise"],
               "border_fraction": meta.get("border_fraction"),
               "on_lattice_rate": meta.get("on_lattice_rate"),
               "wall_clock_s": round(time.time() - t0, 1),
               "pts_file": os.path.join(cdir, f"pts_{arm}.npy")}
        pd.DataFrame([row]).to_csv(res_path, mode="a", index=False,
                                   header=not os.path.exists(res_path))
        print(f"  done in {row['wall_clock_s']:.0f}s "
              f"border={row['border_fraction']}", flush=True)

    print(f"\nAll cells written to {res_path}")
    print("Next: score K-SSE for every cell with\n"
          "  python recompute_spatial.py --data <csv> --pa-dir <cell dir> --out <dir>")


if __name__ == "__main__":
    main()
