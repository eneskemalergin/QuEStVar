"""
Generate data/demo_full.tsv -- the canonical demo file for QuEStVar.

Produces 1000 proteins (3 reps per condition) with a balanced mix of outcome
classes and a set of controlled sparse missing-value patterns so that the demo
exercises the CV filter, allow_missing handling, and NaN-aware statistics.

Missing-value breakdown (200 proteins total, 800 fully observed):
  c1_full  (30) : all three c1 replicates absent -- fully missing in one cond
  c2_full  (30) : all three c2 replicates absent
  c1_one   (40) : one random c1 replicate absent
  c2_one   (40) : one random c2 replicate absent
  c1_two   (20) : two c1 replicates absent (single value remaining)
  c2_two   (20) : two c2 replicates absent
  both_one (15) : one replicate absent from each condition
  all_full  (5) : all six values absent (completely unmeasured)

Run from the repo root:
    uv run python tools/make_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.synthesize import DataConfig, inject_missingness, make_balanced_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEMO_PATH = REPO_ROOT / "data" / "demo_full.tsv"

# 1000 proteins: 334 EQ + 333 DF + 333 NS, 3 reps per condition
CFG = DataConfig(
    n_eq=334,
    n_df=333,
    n_ns=333,
    n_reps=3,
    cv=0.10,
    seed=2026,
)

COND_1 = [f"c1_{i}" for i in range(CFG.n_reps)]
COND_2 = [f"c2_{i}" for i in range(CFG.n_reps)]

MISSINGNESS_PATTERNS: list[tuple[str, int]] = [
    ("c1_full",  30),
    ("c2_full",  30),
    ("c1_one",   40),
    ("c2_one",   40),
    ("c1_two",   20),
    ("c2_two",   20),
    ("both_one", 15),
    ("all_full",  5),
]

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def main() -> None:
    ds = make_balanced_dataset(CFG)

    # Rename columns from c1_r1/c2_r1 style to c1_0/c2_0 style
    rename = {f"c1_r{i+1}": f"c1_{i}" for i in range(CFG.n_reps)}
    rename.update({f"c2_r{i+1}": f"c2_{i}" for i in range(CFG.n_reps)})
    df = ds.data.rename(rename)

    df_missing = inject_missingness(
        df,
        cond_1=COND_1,
        cond_2=COND_2,
        patterns=MISSINGNESS_PATTERNS,
        seed=42,
    )

    # Reorder columns: intensity columns first, protein_id last
    intensity_cols = COND_1 + COND_2
    df_missing = df_missing.select(intensity_cols + ["protein_id"])

    df_missing.write_csv(DEMO_PATH, separator="\t")

    # Summary report
    total = len(df_missing)
    null_counts = df_missing.null_count()
    total_nulls = sum(null_counts.row(0))
    proteins_with_any_null = df_missing.filter(
        pl.any_horizontal(pl.col(intensity_cols).is_null())
    ).height

    print(f"Written: {DEMO_PATH}")
    print(f"  rows                    : {total}")
    print(f"  columns                 : {df_missing.width}")
    print(f"  total null cells        : {total_nulls}")
    print(f"  proteins with >= 1 null : {proteins_with_any_null} / {total}")
    print(f"  null counts per column  :")
    for col, n in zip(df_missing.columns, null_counts.row(0)):
        if n > 0:
            print(f"    {col}: {n}")


if __name__ == "__main__":
    main()
