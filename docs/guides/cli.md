# CLI Reference

## Overview

The CLI exposes the same analysis engine as the Python API. The three subcommands mirror the three main workflows: run a test, run a power analysis, generate a plot.

## Global flags

`--version` prints the package version. `--help` on any subcommand prints usage details.

## questvar test

Run a single pairwise comparison.

```bash
questvar test \
  --data input.parquet \
  --cond-1 Control_1,Control_2,Control_3 \
  --cond-2 Treated_1,Treated_2,Treated_3 \
  --output results.parquet
```

### Input files

`--data` accepts parquet, CSV, and TSV files. The format is inferred from the file extension. CSV and TSV files use `null_values=["", "NA", "NaN"]`.

### Condition columns

`--cond-1` and `--cond-2` take comma-separated column names. Each must have at least two columns. Paired testing with `--is-paired` requires equal counts.

### Threshold overrides

`--cv-thr`, `--p-thr`, `--df-thr`, and `--eq-thr` override individual config values. When `--config` is also provided, CLI flags take precedence.

### Input scale

`--input-scale raw` (default) tells QuEStVar to expect raw positive intensities. `--input-scale log2` tells it the input is already log2-transformed. The deprecated `--is-log2` flag is an alias for `--input-scale log2`.

### Correction

`--correction` selects the multiple testing correction method. Valid values: bonferroni, holm, hochberg, fdr, fdr_bh, BY, qvalue, or none.

### Boolean flags

`--allow-missing` and `--no-allow-missing` control missing value handling. `--is-paired` and `--no-is-paired` switch between unpaired and paired testing. `--var-equal` and `--no-var-equal` control the variance assumption.

### Config file

`--config` loads a YAML config file. CLI flags override any values from the file.

### Output

`--output` specifies the result path. Supported formats: .parquet, .csv, .tsv. A sidecar `.info.*` file and a `.meta.json` file are written alongside the main output.

## questvar power

Run a power analysis sweep from the command line.

```bash
questvar power \
  --eq-boundaries 0.1,0.3,0.5,0.7,0.9 \
  --n-reps-list 3,5,10,20 \
  --output power_results.parquet
```

### Grid parameters

`--eq-boundaries`, `--n-reps-list`, `--cv-mean-list`, `--cv-thr-list`, and `--n-prts-list` define the design space. Each takes a comma-separated list of values.

### Scalar overrides

`--eq-thr`, `--n-reps`, `--cv-mean`, and `--cv-thr` set single values. When a scalar is provided without a corresponding grid, it replaces the grid. When a grid is provided, the scalar is derived from the grid's first value.

### Simulation parameters

`--int-mu` and `--int-sd` control the simulated mean log-intensity and its standard deviation. `--cv-k` and `--cv-theta` control the gamma CV distribution shape and scale. `--n-iterations` sets the Monte Carlo iterations per design point.

### Targets

`--target-sei` and `--target-power` set the optimization targets for the design search. Defaults are 0.80 for both.

### Output

`--output` supports .parquet, .csv, .tsv for design-grid-only export, and .json for full payload export (including run metrics, search results, and diagnostics).

## questvar plot

Generate a figure from saved results.

```bash
questvar plot --type antlers --results results.parquet --output antlers.png
questvar plot --type power --results power_results.parquet --output power_profile.png
```

### Types

`--type antlers` generates the standalone Antler plot from a TestResults file. `--type power` generates the power profile from a PowerResults file. Both file types are loaded with their metadata sidecar for full fidelity.

### Output

`--output` accepts any matplotlib-supported image extension: png, svg, pdf, jpg, tiff.
