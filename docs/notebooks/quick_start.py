# %% [markdown]
# Quick Start
#
# You have two conditions, replicate measurements, and a feature table. You need a decision on each feature: equivalent, different, or inconclusive. A non-significant difference test does not mean two groups are equivalent. QuEStVar adds an equivalence test so that case is explicit instead of implied.
#
# This tutorial runs through the full workflow on realistic proteomics data. 10,000 features, 10 replicates per condition, missing values, a known mixture of equivalent, differential, and high-noise features.

# %%
from contextlib import suppress
from pathlib import Path

import polars as pl

from questvar import QuestVar

_candidates = [Path("data/demo_realistic.tsv"), Path.cwd() / "data" / "demo_realistic.tsv"]
with suppress(NameError):
    _candidates.append(
        Path(__file__).resolve().parent.parent.parent / "data" / "demo_realistic.tsv"
    )
data_path = next((p for p in _candidates if p.exists()), _candidates[0])
df = pl.read_csv(data_path, separator="\t", null_values=["", "NA", "NaN"])
print(f"Features: {df.shape[0]:,}, columns: {df.shape[1]}")
print(f"Missing values: {df.null_count().sum_horizontal().item():,}")

# %% [markdown]
# ## Configure the analysis
#
# Four thresholds control the decision boundary.
#
# `cv_thr` sets the CV filter. Features with CV above this in either condition are excluded before testing. A value of 1.0 keeps everything. Tighten it to 0.3 for low-noise features only.
#
# `eq_thr` defines the equivalence window in log2 fold change units. A feature with |log2FC| below this boundary and a significant TOST result is called equivalent. Start with 0.5.
#
# `df_thr` defines the difference boundary. Features with |log2FC| above this and a significant t-test are called differential. Must be larger than `eq_thr`. The gap between them is the zone where neither test is decisive.
#
# `p_thr` is the adjusted p-value cutoff for both tests. Default 0.05.
#
# `allow_missing` controls whether features with missing values get CV-filtered or tested. True means CV is computed on available replicates per feature. False means any missing value produces a NaN CV and the feature is excluded. The demo data has missing values, so set this to True.
#
# `correction` controls multiple testing correction. BH-FDR is the default. Other options: bonferroni, holm, hochberg, BY, qvalue, or none.

# %%
qv = QuestVar(
    cv_thr=1.0,
    eq_thr=0.5,
    df_thr=1.0,
    p_thr=0.05,
    correction="fdr",
    allow_missing=True,
)

# %% [markdown]
# ## Run the test
#
# `cond_1` and `cond_2` are lists of column names. Each must have at least two replicates. Paired testing is available with `is_paired=True`.

# %%
cond_1 = [f"c1_{i}" for i in range(10)]
cond_2 = [f"c2_{i}" for i in range(10)]
results = qv.test(df, cond_1=cond_1, cond_2=cond_2)

# %% [markdown]
# ## Read the summary
#
# The summary tells you what happened in one block. Out of 10,000 input features, some were excluded by the CV filter, some were tested, and each tested feature got a status.

# %%
print(results.summary())

# %% [markdown]
# Three statuses appear.
#
# Equivalent (+1) means the effect falls inside the equivalence boundary and the TOST p-value is significant. These features are stable across conditions. Your summary tells you the exact count.
#
# Differential (-1) means the effect exceeds the difference boundary and the t-test p-value is significant. These features change between conditions.
#
# Not significant (0) means neither test was decisive. The effect falls between eq_thr and df_thr, or the p-value is above the cutoff. The bulk of the data usually lands here. This is normal.
#
# Excluded features failed the CV filter. They had too much variance within one condition to produce a reliable test. The `info` sidecar tracks why each feature was excluded.

# %% [markdown]
# ## Visualize with the Antler plot
#
# The Antler plot is the main diagnostic figure. The y-axis shows signed -log10 adjusted p-value. Equivalence results appear above zero. Difference results appear below zero. The x-axis is log2 fold change.
#
# Blue dashed lines mark the equivalence boundary. Red dotted lines mark the difference boundary. Features in the upper band between the blue lines are equivalent. Features in the lower band outside the red lines are differential. Everything else is inconclusive.

# %%
import matplotlib.pyplot as plt

fig = results.plot(cond_1_label="Control", cond_2_label="Treatment")
plt.close(fig)
fig

# %% [markdown]
# ## Save and reload
#
# Results save to parquet with two sidecar files: one for the CV filter info table, one for metadata (config, condition labels). Loading them back gives you the same object with the same methods.

# %%
results.save("tmp/quick_start_results.parquet")
reloaded = type(results).load("tmp/quick_start_results.parquet")
print(f"Reloaded: {len(reloaded.data)} features, {len(reloaded.info)} total")

# %% [markdown]
# From here you can adjust thresholds, try different correction methods, run a power analysis to pick an equivalence boundary, or move to a multi-comparison workflow with `compare_all_pairs`.
