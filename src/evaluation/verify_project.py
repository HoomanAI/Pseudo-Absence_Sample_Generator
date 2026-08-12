"""Fast, non-destructive integrity checks for the AP-WGAN experiment bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
RETRAIN = HERE / "retrain"
DATA = ROOT / "Pseudo-Absence_Sample_Generator" / "data" / "Fire_points_dataset_final_csv.csv"


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"missing: {path}")


def main() -> int:
    required = [
        DATA,
        RETRAIN / "manifest.csv",
        RESULTS / "e4_dual_artifact.csv",
        RESULTS / "e4_summary.csv",
        RESULTS / "e4_ap_invariance.csv",
        RESULTS / "e4_equivalence.json",
        ROOT / "Overleaf 2" / "figures" / "F24_e4_condition_grid.pdf",
        ROOT / "Overleaf 2" / "figures" / "F25_e4_downstream_auc.pdf",
    ]
    for path in required:
        require(path)

    manifest = pd.read_csv(RETRAIN / "manifest.csv")
    assert len(manifest) == 24, f"expected 24 trained arm/seed cells, found {len(manifest)}"
    assert manifest["label"].nunique() == 8, "expected eight experiment arms"
    assert manifest["seed"].nunique() == 3, "expected three seeds"

    for row in manifest.itertuples():
        run = RETRAIN / f"{row.label}_seed{int(row.seed)}"
        require(run / f"pts_{row.arm}.npy")
        require(run / f"feats_{row.arm}.npy")

    e4 = pd.read_csv(RESULTS / "e4_dual_artifact.csv")
    cells = e4[["arm", "condition", "seed"]].drop_duplicates()
    assert len(cells) == 96, f"expected 96 E4 cells, found {len(cells)}"
    assert set(cells["condition"]) == {"baseline", "lattice", "bw", "lattice_bw"}

    invariance = pd.read_csv(RESULTS / "e4_ap_invariance.csv")
    assert np.allclose(invariance["max_abs_diff"], 0), "AP lattice invariance failed"
    assert invariance["feature_array_equal"].astype(bool).all(), "AP feature arrays are not invariant"

    equivalence = json.loads((RESULTS / "e4_equivalence.json").read_text(encoding="utf-8"))
    differences = [float(v) for key, v in equivalence.items() if "diff" in key]
    assert differences and max(differences) <= 1e-12, f"chunked matcher equivalence failed: {equivalence}"

    print("AP-WGAN verification passed")
    print(f"  trained cells: {len(manifest)} (8 arms x 3 seeds)")
    print(f"  E4 cells:      {len(cells)} (8 arms x 4 conditions x 3 seeds)")
    print("  invariance:    exact")
    print("  figures:       F24 and F25 present")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
