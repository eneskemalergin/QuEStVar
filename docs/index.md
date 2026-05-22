# QuEStVar

**Quantitative Exploration of Stability and Variability through statistical hypothesis testing**

QuEStVar is a testing utility for paired equivalence and difference testing in quantitative omics data. It labels each measured feature as equivalent, not significant, or different using CV filtering, Welch's t-test, TOST equivalence testing, and multiple testing correction.

## Quick links

- [Quick Start Tutorial](notebooks/quick_start.py) -- 5 minute walkthrough with demo data
- [Power Analysis Tutorial](notebooks/power_analysis.py) -- experiment planning and design sweeps
- [Configuration Guide](guides/configuration.md) -- YAML config, CLI flags, all parameters
- [Plotting Guide](guides/plotting.md) -- Antler's plot, summary figure, power profile
- [API Reference](api/questvar.md) -- full function and class documentation
- [Contributing](contributing.md) -- development setup and PR workflow

## Installation

```bash
pip install questvar[plot,yaml]
```

For the core engine only (no plotting, no YAML):

```bash
pip install questvar
```

## Quick example

```python
import questvar as qv
import polars as pl

df = pl.read_csv("data/demo_realistic.tsv", separator="\t")
result = qv.test(df, cond_1=["c1_0","c1_1","c1_2"], cond_2=["c2_0","c2_1","c2_2"])
print(result.summary())
result.plot()
```
