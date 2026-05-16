# QuEStVar

**Quantitative Exploration of Stability and Variability through statistical hypothesis testing**

QuEStVar is a standalone Python package for equivalence and difference testing in proteomics. It provides a vectorized TOST + Welch's t-test engine with CV-based quality filtering, multiple testing correction (BH-FDR, Storey q-value), and power analysis.

- Pure numpy computation with zero scipy dependency
- Polars-based data I/O, Parquet/CSV/TSV
- Three access paths: Python API, YAML config, CLI

## Requirements

- Python >= 3.12
- numpy >= 2.0
- polars >= 1.0

## Installation

```bash
uv add questvar
```

Or with extras:

```bash
uv add "questvar[plots,yaml,cli]"
```

## Quick Start

```python
import questvar as qv

results = qv.test(
    data,
    cond_1=["Control_1", "Control_2", "Control_3"],
    cond_2=["Treated_1", "Treated_2", "Treated_3"],
    eq_thr=0.5,
)
print(results.summary())
results.save("results.parquet")
results.plot()
```

## License

MIT
