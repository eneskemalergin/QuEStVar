# Configuration

## TestConfig

TestConfig controls a single pairwise comparison. Every parameter has a default. You only need to set the ones you want to change.

### Thresholds

`cv_thr` sets the CV filter. Features with a coefficient of variation above this threshold in either condition are excluded before any statistical test. CV is computed on the raw intensity scale regardless of the `is_log2` setting. A value of 1.0 keeps everything. A value of 0.3 excludes features with more than 30 percent variance. Default is 1.0.

`eq_thr` defines the equivalence boundary in log2 fold change units. A feature with |log2FC| below this boundary and a significant TOST result is declared equivalent. Default is 0.5.

`df_thr` defines the difference boundary. Features with |log2FC| above this and a significant t-test are declared differential. Must be strictly larger than `eq_thr`. The zone between eq_thr and df_thr is where neither test is decisive and the status is "not significant". Default is 1.0.

`p_thr` is the adjusted p-value threshold for both tests. A feature is declared equivalent or differential only if its adjusted p-value falls below this cutoff. Default is 0.05.

### Input handling

`is_log2` tells QuEStVar whether the input matrix is raw intensities or already log2-transformed. When False (default), CV is computed on the raw values, then the matrix is clipped at 1e-300 and log2-transformed for the statistical tests. When True, CV is computed on back-transformed values (2 ** x) while the tests run on the original log2 scale. Set this to True only when the input is already in log2 space.

`allow_missing` controls CV computation for features with missing values. When False (default), any missing replicate produces a NaN CV and the feature is excluded. When True, CV is computed on the available replicates per feature using nan-aware mean and standard deviation. Use True when your dataset has scattered missing values and you want to keep features with partial data.

### Test assumptions

`var_equal` controls the variance assumption for the t-test. When False (default), Welch's t-test is used. This does not assume equal variance between conditions. When True, Student's t-test with pooled variance is used. Welch is the safe default.

`is_paired` switches from independent to paired testing. When True, the paired t-test and paired TOST are used instead of the unpaired versions. Paired tests require equal numbers of replicates in each condition. Default is False.

### Multiple testing correction

`correction` selects the multiple testing correction method. Applied uniformly to all p-value series in the pipeline. Valid options:

- `"fdr"` (default): Benjamini-Hochberg FDR
- `"fdr_bh"`: same as "fdr"
- `"bonferroni"`: Bonferroni correction
- `"holm"`: Holm-Bonferroni step-down
- `"hochberg"`: Hochberg step-up
- `"BY"`: Benjamini-Yekutieli FDR
- `"qvalue"`: Storey q-value
- `None` or `"none"`: no correction

### Reference

See the [TestConfig API reference](../api/questvar.md) for the complete parameter listing.

## PowerConfig

PowerConfig controls the power analysis simulation. All parameters have defaults tuned for a typical proteomics experiment. The grid parameters (`eq_boundaries`, `n_reps_grid`, `cv_mean_grid`, `cv_thr_grid`) define the design space to sweep. The scalar parameters (`n_prts`, `n_iterations`, etc.) control simulation behavior.

### Simulation parameters

`n_prts` sets the number of features simulated per Monte Carlo iteration. More features give more precise SEI and power estimates. Default is 5000.

`n_reps` is the default replicate count used when no grid override is provided. The grid parameter `n_reps_grid` determines what gets swept. If the grid has multiple values, the scalar follows from the grid.

`cv_mean` is the default mean CV used when no grid override is provided. The `cv_mean_grid` determines what gets swept. CV is simulated using a gamma distribution scaled to match this mean.

`cv_k` and `cv_theta` are the shape and scale parameters for the gamma distribution that generates per-feature CV values. Default k=2.0, theta=0.5. These control the spread of CVs around the mean.

`eq_thr` is the default equivalence boundary. The grid parameter `eq_boundaries` determines what gets swept.

### Power targets

`target_sei` sets the Stable Equivalence Index target. SEI measures the proportion of truly equivalent features that TOST correctly recovers. Default is 0.80.

`target_power` sets the power target for the design search. The search finds the cheapest design that achieves this power. Default is 0.80.

### Grid parameters

`eq_boundaries` is a tuple of equivalence thresholds to sweep. Default is (0.1, 0.3, 0.5, 0.7, 0.9).

`n_reps_grid` is a tuple of replicate counts to sweep. Default is (3, 5, 10, 20).

`cv_mean_grid` is a tuple of mean CV values to sweep. Default is (0.10, 0.20, 0.30).

`cv_thr_grid` is a tuple of CV thresholds to sweep. Default is (1.0, 1.5, 2.0).

### Reproducibility

`random_seed` makes the simulation deterministic. Pass any integer. Same seed plus same parameters produces identical results across runs. Default is None, which still gives deterministic results (seeds derived from run_id only).

`n_iterations` controls Monte Carlo iterations per design point. More iterations tighten confidence bands. Default is 10.

`n_jobs` controls parallel workers. Default None uses half of the available CPU cores.

### Reference

See the [PowerConfig API reference](../api/power_results.md) for the complete parameter listing.

## YAML config files

Both config objects support YAML serialization. Write a config to a file, edit it, and load it back.

```yaml
cv_thr: 0.3
p_thr: 0.01
eq_thr: 0.5
df_thr: 1.0
correction: bonferroni
is_log2: false
allow_missing: true
```

Load it with:

```python
from questvar import QuestVar

qv = QuestVar.from_yaml("config.yaml")
results = qv.test(data, cond_1=["A1", "A2"], cond_2=["B1", "B2"])
```

CLI flags override YAML values when both are provided. This lets you use a base config file and override specific parameters per run.
