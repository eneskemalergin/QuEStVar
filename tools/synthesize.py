"""
Synthetic data generation for QuEStVar testing and benchmarking.

Produces datasets with a controlled, balanced mix of outcome classes:

  EQ (status =  1): |log2FC| well inside eq_thr, low CV -> TOST significant
  DF (status = -1): |log2FC| well above df_thr, low CV  -> t-test significant
  NS (status =  0): |log2FC| in the gray zone (eq_thr < |fc| < df_thr)
                    Status is structurally NS regardless of p-values.

All data is in raw (linear) intensity space.
Pass is_log2=False when running through questvar or the ref implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Parameters for one synthetic dataset."""
    n_eq: int = 100        # proteins that should come out EQ
    n_df: int = 100        # proteins that should come out DF
    n_ns: int = 100        # proteins that should come out NS (gray-zone)
    n_reps: int = 6        # replicates per condition

    # Ground-truth log2FC for each group
    eq_fc: float = 0.00    # EQ: no fold change
    df_fc: float = 1.50    # DF: clearly above df_thr (0.75)
    ns_fc: float = 0.625   # NS: midpoint of gray zone [0.5, 0.75]

    # Within-condition variability: target CV in raw space
    cv: float = 0.08       # 8% CV -> passes cv_thr=0.15

    # Mean log2 intensity distribution across proteins
    int_mu_log2: float = 18.0
    int_sd_log2: float = 0.5

    seed: int | None = 42


# Preset configurations covering a range of sizes / replicate counts
CONFIGS: dict[str, DataConfig] = {
    "small":         DataConfig(n_eq=50,   n_df=50,   n_ns=50,   n_reps=4),
    "medium":        DataConfig(n_eq=200,  n_df=200,  n_ns=200,  n_reps=6),
    "large":         DataConfig(n_eq=1000, n_df=1000, n_ns=1000, n_reps=6),
    "xlarge":        DataConfig(n_eq=2000, n_df=2000, n_ns=2000, n_reps=8),
    "few_reps":      DataConfig(n_eq=200,  n_df=200,  n_ns=200,  n_reps=3),
    "many_reps":     DataConfig(n_eq=200,  n_df=200,  n_ns=200,  n_reps=12),
    "unbalanced_eq": DataConfig(n_eq=400,  n_df=100,  n_ns=100,  n_reps=6),
    "unbalanced_df": DataConfig(n_eq=100,  n_df=400,  n_ns=100,  n_reps=6),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """A generated dataset with its ground truth."""
    data: pl.DataFrame        # columns: protein_id, cond_1_*, cond_2_*
    cond_1: list[str]         # condition-1 column names
    cond_2: list[str]         # condition-2 column names
    truth: np.ndarray         # int8 array: 1=EQ, -1=DF, 0=NS per row
    protein_ids: list[str]
    config: DataConfig

    @property
    def n_proteins(self) -> int:
        return len(self.truth)

    def summary(self) -> dict[str, int]:
        """Count proteins per ground-truth class."""
        from collections import Counter
        c = Counter(self.truth.tolist())
        return {"EQ": c.get(1, 0), "DF": c.get(-1, 0), "NS": c.get(0, 0)}

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (s1_arr, s2_arr) as (n_proteins, n_reps) float64 arrays."""
        s1 = self.data.select(self.cond_1).to_numpy().astype(np.float64)
        s2 = self.data.select(self.cond_2).to_numpy().astype(np.float64)
        return s1, s2


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def _lognormal_replicates(
    means_raw: np.ndarray,     # shape (n_proteins,)
    cv: float,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw (n_proteins, n_reps) log-normal replicates with specified raw-space CV.

    For a lognormal(mu_ln, sigma_ln):
        CV = sqrt(exp(sigma_ln^2) - 1)
        sigma_ln = sqrt(log(1 + CV^2))
        mu_ln    = log(mean) - 0.5 * sigma_ln^2
    """
    sigma_ln = np.sqrt(np.log1p(cv ** 2))
    mu_ln = np.log(means_raw) - 0.5 * sigma_ln ** 2
    # Broadcast: (n_proteins, 1) -> (n_proteins, n_reps)
    return rng.lognormal(
        mu_ln[:, np.newaxis],
        sigma_ln,
        size=(len(means_raw), n_reps),
    )


def _draw_means(
    n: int,
    int_mu_log2: float,
    int_sd_log2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n protein means from a Normal in log2-space, return as raw values."""
    log2_means = rng.normal(int_mu_log2, int_sd_log2, size=n)
    return np.power(2.0, log2_means)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_balanced_dataset(config: DataConfig | None = None, **kwargs) -> SyntheticDataset:
    """
    Generate a synthetic dataset with balanced EQ / DF / NS outcome classes.

    Parameters
    ----------
    config : DataConfig or None
        If None, uses DataConfig defaults. Keyword overrides are merged.
    **kwargs
        Override individual DataConfig fields.

    Returns
    -------
    SyntheticDataset
        Includes a Polars DataFrame (protein_id + replicate columns),
        condition column name lists, and a ground-truth status array.

    Notes
    -----
    * Data is in **raw (linear) intensity space** - set is_log2=False.
    * Three protein groups are generated:
        EQ: log2FC = eq_fc (default 0.0), tight within-condition spread
        DF: log2FC = df_fc (default 1.5), tight within-condition spread
        NS: log2FC = ns_fc (default 0.625) - gray zone, always status=0
    * CV in raw space is DataConfig.cv (default 8%), well below cv_thr=0.15.
    * With n_reps >= 4 and cv=0.08, EQ proteins consistently pass TOST at
      FDR=0.01 and DF proteins consistently pass the t-test.
    """
    if config is None:
        config = DataConfig(**kwargs)
    elif kwargs:
        import dataclasses
        config = dataclasses.replace(config, **kwargs)

    rng = np.random.default_rng(config.seed)
    n_total = config.n_eq + config.n_df + config.n_ns

    # Sample baseline protein means (condition 1)
    means_c1 = _draw_means(n_total, config.int_mu_log2, config.int_sd_log2, rng)

    # Condition-2 means: apply the group-specific log2FC shift
    group_fc = np.concatenate([
        np.full(config.n_eq, config.eq_fc),
        np.full(config.n_df, config.df_fc),
        np.full(config.n_ns, config.ns_fc),
    ])
    means_c2 = means_c1 * np.power(2.0, group_fc)

    # Generate replicates
    s1 = _lognormal_replicates(means_c1, config.cv, config.n_reps, rng)
    s2 = _lognormal_replicates(means_c2, config.cv, config.n_reps, rng)

    # Ground-truth status
    truth = np.concatenate([
        np.ones(config.n_eq,  dtype=np.int8),
        np.full(config.n_df, -1, dtype=np.int8),
        np.zeros(config.n_ns, dtype=np.int8),
    ])

    # Shuffle so EQ/DF/NS are not in order
    order = rng.permutation(n_total)
    s1, s2, truth = s1[order], s2[order], truth[order]

    # Column names
    cond_1 = [f"c1_r{i+1}" for i in range(config.n_reps)]
    cond_2 = [f"c2_r{i+1}" for i in range(config.n_reps)]
    protein_ids = [f"P{i+1:05d}" for i in range(n_total)]

    # Build Polars DataFrame
    rows: dict[str, list] = {"protein_id": protein_ids}
    for j, col in enumerate(cond_1):
        rows[col] = s1[:, j].tolist()
    for j, col in enumerate(cond_2):
        rows[col] = s2[:, j].tolist()

    df = pl.DataFrame(rows)

    return SyntheticDataset(
        data=df,
        cond_1=cond_1,
        cond_2=cond_2,
        truth=truth,
        protein_ids=protein_ids,
        config=config,
    )


def make_all_configs() -> dict[str, SyntheticDataset]:
    """Generate one SyntheticDataset per preset CONFIGS entry."""
    return {name: make_balanced_dataset(cfg) for name, cfg in CONFIGS.items()}


def make_paired_dataset(config: DataConfig | None = None, **kwargs) -> SyntheticDataset:
    """
    Generate a synthetic dataset for paired t-test comparison.

    Each replicate j shares a sample-level random effect (δ_j) between
    conditions, so the within-pair correlation is real and the paired test
    will outperform the unpaired test on noisy data.

    Structure in log2 space:
        x1[i,j] = mu_i + delta_j + eps1[i,j]
        x2[i,j] = mu_i + fc_i + delta_j + eps2[i,j]
        diff[i,j] = fc_i + eps2[i,j] - eps1[i,j]

    delta_j  ~ N(0, sigma_between)  -- shared sample effect
    eps[i,j] ~ N(0, sigma_within)   -- independent technical noise

    sigma_between is set to sigma_within (strong pairing) so a paired test
    captures the cancellation and has much lower effective SE than unpaired.

    Returns
    -------
    SyntheticDataset with the same layout as make_balanced_dataset.
    Data is in raw (linear) space; use is_log2=False.
    """
    if config is None:
        config = DataConfig(**kwargs)
    elif kwargs:
        import dataclasses
        config = dataclasses.replace(config, **kwargs)

    rng = np.random.default_rng(config.seed)
    n_total = config.n_eq + config.n_df + config.n_ns

    # Protein means in log2 space
    mu_log2 = rng.normal(config.int_mu_log2, config.int_sd_log2, size=n_total)

    # Per-protein FC
    fc = np.concatenate([
        np.full(config.n_eq, config.eq_fc),
        np.full(config.n_df, config.df_fc),
        np.full(config.n_ns, config.ns_fc),
    ])

    # Noise scale in log2 space from CV
    sigma_within  = np.sqrt(np.log1p(config.cv ** 2)) / np.log(2.0)
    sigma_between = sigma_within  # strong pairing: same magnitude

    # Sample-level effects: shape (n_reps,)
    delta = rng.normal(0, sigma_between, size=config.n_reps)

    # Per-replicate technical noise: shape (n_total, n_reps)
    eps1 = rng.normal(0, sigma_within, size=(n_total, config.n_reps))
    eps2 = rng.normal(0, sigma_within, size=(n_total, config.n_reps))

    # Build log2 arrays then convert to raw
    x1_log2 = mu_log2[:, np.newaxis] + delta[np.newaxis, :] + eps1
    x2_log2 = mu_log2[:, np.newaxis] + fc[:, np.newaxis] + delta[np.newaxis, :] + eps2

    s1 = np.power(2.0, x1_log2)
    s2 = np.power(2.0, x2_log2)

    # Ground-truth status
    truth = np.concatenate([
        np.ones(config.n_eq,  dtype=np.int8),
        np.full(config.n_df, -1, dtype=np.int8),
        np.zeros(config.n_ns, dtype=np.int8),
    ])

    # Shuffle
    order = rng.permutation(n_total)
    s1, s2, truth = s1[order], s2[order], truth[order]

    cond_1 = [f"c1_r{i+1}" for i in range(config.n_reps)]
    cond_2 = [f"c2_r{i+1}" for i in range(config.n_reps)]
    protein_ids = [f"P{i+1:05d}" for i in range(n_total)]

    rows: dict[str, list] = {"protein_id": protein_ids}
    for j, col in enumerate(cond_1):
        rows[col] = s1[:, j].tolist()
    for j, col in enumerate(cond_2):
        rows[col] = s2[:, j].tolist()

    return SyntheticDataset(
        data=pl.DataFrame(rows),
        cond_1=cond_1,
        cond_2=cond_2,
        truth=truth,
        protein_ids=protein_ids,
        config=config,
    )


# ---------------------------------------------------------------------------
# CLI entry point (quick sanity print)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = make_balanced_dataset()
    print("Dataset summary:", ds.summary())
    print("Shape:", ds.data.shape)
    print(ds.data.head(5))
