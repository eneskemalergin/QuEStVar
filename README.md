<!-- markdownlint-disable MD033 MD036 MD041 MD045 -->
<p align="center">
    <img src="assets/questvar.jpg" alt="QuEStVar" width="270" />
</p>

<p align="center">
    <strong>Quantitative Exploration of Stability and Variability through statistical hypothesis testing</strong>
</p>

<p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/version-0.1.0-0f766e?style=for-the-badge" alt="Version 0.1.0" /></a>
    <a href="ref/statistical-testing-for-protein-equivalence-identifies-core-functional-modules-conserved-across-360-cancer-cell-lines.pdf"><img src="https://img.shields.io/badge/paper-linked-7c3aed?style=for-the-badge" alt="Paper linked" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/python-3.12%2B-f59e0b?style=for-the-badge" alt="Python 3.12+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/numpy-2.0%2B-2563eb?style=for-the-badge" alt="NumPy 2.0+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/polars-1.0%2B-06b6d4?style=for-the-badge" alt="Polars 1.0+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/scipy-1.10%2B-8b5cf6?style=for-the-badge" alt="SciPy 1.10+" /></a>
</p>

<p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/install-uv%20or%20pip-16a34a?style=for-the-badge&logo=python&logoColor=white" alt="Install with uv or pip" /></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/api-QuestVar-0f766e?style=for-the-badge&logo=python&logoColor=white" alt="Python API" /></a>
    <a href="#cli"><img src="https://img.shields.io/badge/cli-questvar-111827?style=for-the-badge&logo=terminal&logoColor=white" alt="CLI questvar" /></a>
    <a href="#plotting"><img src="https://img.shields.io/badge/plot-antlers-ef4444?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Antlers plot" /></a>
</p>

QuEStVar is a testing utility for paired equivalence and difference testing. It is built around the logic in [Statistical testing for protein equivalence identifies core functional modules conserved across 360 cancer cell lines](ref/statistical-testing-for-protein-equivalence-identifies-core-functional-modules-conserved-across-360-cancer-cell-lines.pdf).

The point is simple. A non-significant difference test does not mean two groups are equivalent. QuEStVar adds an equivalence test so that case is explicit instead of implied.

The package uses NumPy and SciPy for computation and Polars for data I/O. Python API, YAML-backed configuration, and CLI entry points all point at the same analysis engine.

## Features

- Vectorized difference and equivalence testing across entire feature tables
- CV filtering before statistical testing
- Multiple testing correction with BH-FDR, Holm, Bonferroni, Hochberg, BY, and q-value paths
- Antler's plot for signed `-log10(p)` visualisation
- Power analysis for equivalence boundary planning
- Polars-first input and output, with pandas accepted at the boundary

## Requirements

- Python 3.12 or newer
- NumPy 2.0 or newer
- Polars 1.0 or newer
- SciPy 1.10 or newer

Optional extras:

- `matplotlib` for plotting
- `pyyaml` for YAML config loading in the library
- `typer` for an alternate CLI experience later
- `numba` for user-side acceleration experiments

## Installation

From a local checkout:

```bash
uv pip install -e .
```

With extras:

```bash
uv pip install -e ".[plots,yaml,cli,accel]"
```

With pip:

```bash
pip install -e .
```

## Quick Start

```python
import questvar as qv

qv_obj = qv.QuestVar(cv_thr=0.15, p_thr=0.05, df_thr=1.0, eq_thr=0.5)
results = qv_obj.test(
    data,
    cond_1=["Control_1", "Control_2", "Control_3"],
    cond_2=["Treated_1", "Treated_2", "Treated_3"],
)

print(results.summary())
results.save("results.parquet")
results.plot()
```

## YAML Config

The library can build a `QuestVar` object from YAML when `pyyaml` is installed.

```python
import questvar as qv

qv_obj = qv.QuestVar.from_yaml("config.yaml")
results = qv_obj.test(data, cond_1=["Control_1", "Control_2"], cond_2=["Treated_1", "Treated_2"])
```

Example configuration:

```yaml
cv_thr: 0.15
p_thr: 0.05
df_thr: 1.0
eq_thr: 0.5
correction: fdr
is_log2: false
var_equal: false
is_paired: false
allow_missing: false
```

## CLI

The package exposes a `questvar` command.

```bash
questvar test \
  --data input.parquet \
  --cond-1 Control_1,Control_2,Control_3 \
  --cond-2 Treated_1,Treated_2,Treated_3 \
  --output results.parquet
```

Power analysis:

```bash
questvar power \
  --eq-boundaries 0.1,0.3,0.5,0.7,0.9 \
  --n-reps-list 3,5,10,20 \
  --cv-mean-list 15,27.5,40 \
  --output power_results.parquet
```

Plotting:

```bash
questvar plot --type antlers --results results.parquet --output antlers.png
questvar plot --type power --results power_results.parquet --output power_profile.png
```

## Plotting

The main figure is Antler's plot. It uses signed `-log10(p)` on the y axis and `log2 fold change` on the x axis. Equivalent features appear above zero, differential features below zero.

When `matplotlib` is installed, you can also call the plotting helpers directly:

```python
results.plot()
```

## Public API

```python
import questvar as qv

results = qv.test(data, cond_1=["A1", "A2"], cond_2=["B1", "B2"])
power = qv.run_power_analysis(
    eq_boundaries=[0.3, 0.5, 0.7],
    n_reps_list=[3, 5, 10],
    cv_mean_list=[15, 27.5, 40],
    n_prts=10000,
    n_iterations=10,
    target_power=0.8,
)
```

The main objects are `QuestVar`, `TestResults`, and `PowerResults`.

## Project Notes

This repository is a rewrite of the QuEStVar analysis logic from the manuscript above. The current codebase keeps the computation path small and explicit. It does not include imputation, clustering, enrichment analysis, or PCA.

Paired testing requires equal group sizes. Missing values are handled with nan-aware summary statistics at the boundary, not with built-in imputation.

## License

MIT

---

<p align="center">
    <em>Proteins are compared,</em><br />
    <em>boundaries keep the signal honest,</em><br />
    <em>and the plot carries both sides.</em>
</p>
