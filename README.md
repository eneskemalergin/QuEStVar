<!-- markdownlint-disable MD033 MD036 MD041 MD045 -->
<p align="center">
    <img src="https://raw.githubusercontent.com/eneskemalergin/QuEStVar/main/assets/questvar.jpg" alt="QuEStVar" width="270" />
</p>

<p align="center">
    <strong>Quantitative Exploration of Stability and Variability through statistical hypothesis testing</strong>
</p>

<p align="center">
    <a href="https://pypi.org/project/questvar/"><img src="https://img.shields.io/badge/version-v0.1.0-0f766e?style=for-the-badge" alt="v0.1.0" /></a>
    <a href="ref/statistical-testing-for-protein-equivalence-identifies-core-functional-modules-conserved-across-360-cancer-cell-lines.pdf"><img src="https://img.shields.io/badge/paper-linked-7c3aed?style=for-the-badge" alt="Paper" /></a>
    <a href="https://eneskemalergin.github.io/QuEStVar/"><img src="https://img.shields.io/badge/docs-site-0891b2?style=for-the-badge" alt="Documentation" /></a>
    <a href="https://github.com/eneskemalergin/QuEStVar/actions"><img src="https://img.shields.io/github/actions/workflow/status/eneskemalergin/QuEStVar/ci.yml?branch=main&style=for-the-badge&logo=github&label=CI" alt="CI" /></a>
</p>
<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-f59e0b?style=for-the-badge" alt="Python 3.12+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/numpy-2.0%2B-2563eb?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy 2.0+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/scipy-1.10%2B-8b5cf6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy 1.10+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/polars-1.0%2B-cd7c2f?style=for-the-badge&logo=polars&logoColor=white" alt="Polars 1.0+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/matplotlib-3.10%2B-003049?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib 3.10+" /></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/pyyaml-6.0%2B-6b2800?style=for-the-badge&logo=python&logoColor=white" alt="PyYAML 6.0+" /></a>
</p>

QuEStVar is a testing utility for paired equivalence and difference testing. It is packaged and easy to access built around the logic we published with biological application: [Statistical testing for protein equivalence identifies core functional modules conserved across 360 cancer cell lines](https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00131).

The point is simple. A non-significant difference test does not mean two groups are equivalent. QuEStVar adds an equivalence test so that case is explicit instead of implied.

**v0.1.0 is a stable minimum working version.** The core workflow is complete and tested. Future releases will add multi-comparison analysis, expanded visualization, and deeper validation tools.

## Features

- Vectorized difference and equivalence testing across entire feature tables
- CV quality filtering before statistical testing
- Seven multiple testing correction methods: BH-FDR, Bonferroni, Holm, Hochberg, BY, q-value, or none
- Antler plot combining TOST and t-test results
- Eight-panel summary figure
- Power analysis for experiment planning
- CLI, Python API, and YAML config with identical results
- CI-tested on Linux, macOS, and Windows

## Installation

```bash
pip install questvar[plot,yaml]
```

For the core engine only (no plotting, no YAML config):

```bash
pip install questvar
```

## Quick start

```python
import polars as pl
from questvar import QuestVar

df = pl.read_csv("data/demo_realistic.tsv", separator="\t")
qv = QuestVar(cv_thr=1.0, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction="fdr")
results = qv.test(df, cond_1=["c1_0","c1_1","c1_2"], cond_2=["c2_0","c2_1","c2_2"])
print(results.summary())

# Eight-panel summary figure
results.plot(cond_1_label="Control", cond_2_label="Treatment")
```

![Summary plot](https://raw.githubusercontent.com/eneskemalergin/QuEStVar/main/assets/summary_plot.png)

Three statuses appear. Equivalent (+1) means the TOST test is significant and |log2FC| is below the equivalence boundary. Differential (-1) means the t-test is significant and |log2FC| is above the difference boundary. Not significant (0) means neither test was decisive. Excluded features failed the CV quality filter, some were testable but very high CV, and some were untestable due to missing values or zero variance (shows the stats in panel G.)

## Documentation

Full documentation, tutorials, and API reference are at the [documentation site](https://eneskemalergin.github.io/QuEStVar/).

- [Quick start tutorial](https://eneskemalergin.github.io/QuEStVar/notebooks/quick_start/)
- [Power analysis tutorial](https://eneskemalergin.github.io/QuEStVar/notebooks/power_analysis/)
- [Configuration guide](https://eneskemalergin.github.io/QuEStVar/guides/configuration/)
- [Plotting guide](https://eneskemalergin.github.io/QuEStVar/guides/plotting/)
- [CLI reference](https://eneskemalergin.github.io/QuEStVar/guides/cli/)
- [API reference](https://eneskemalergin.github.io/QuEStVar/api/questvar/)

## CLI

```bash
questvar test --data input.parquet --cond-1 A1,A2,A3 --cond-2 B1,B2,B3 --output results.parquet
questvar power --eq-boundaries 0.1,0.3,0.5 --n-reps-list 3,5,10 --output power_results.parquet
questvar plot --type antlers --results results.parquet --output antlers.png
```

## Roadmap

- **Multi-comparison analysis**: metadata-driven pair generation, batch execution, per-pair and per-feature summaries.
- **Visualization expansion**: stacked bar plots, RSM scatter, status heatmaps, agreement matrix for sophisticated summaries across multiple comparisons.
- **Validation tools**: stability analysis, regression golden files, expanded hypothesis tests.
- **Documentation and publishing**: mkdocstrings site, tutorial notebooks, PyPI release.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
    <em>Static cells in ice,</em><br />
    <em>Strict bounds trace the quiet line,</em><br />
    <em>Where twin shapes agree.</em>
</p>
