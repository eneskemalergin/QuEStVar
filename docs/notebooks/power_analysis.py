# %% [markdown]
# Power Analysis
#
# You want to design an experiment. You have a target equivalence boundary in mind, a rough idea of your measurement noise, and a minimum power you need to hit. The question is: how many replicates do you need?
#
# The power analysis module answers that question by simulation. It generates synthetic data where every feature is truly equivalent, then tests how many the TOST pipeline correctly recovers as equivalent. By sweeping across equivalence boundaries, replicate counts, and CV levels, you get a map of statistical power before you run a single real sample.

# %%
import numpy as np

from questvar import run_power_analysis

# %% [markdown]
# ## Configure the sweep
#
# Five parameters define the design space.
#
# `eq_boundaries` lists the equivalence thresholds to test. Wider boundaries make equivalence easier to declare. Narrow boundaries need more data. Typical values range from 0.1 to 1.0 log2 fold change.
#
# `n_reps_list` lists the replicate counts per condition. Power increases with more replicates but with diminishing returns. Three replicates is a bare minimum across most omics. Ten is generous.
#
# `cv_mean_list` lists the mean CV levels to simulate. CV is the ratio of standard deviation to mean intensity. Lower CV means cleaner data. Proteomics data often runs between 0.15 and 0.35. Transcriptomics tends lower.
#
# `n_prts` controls how many features are simulated per Monte Carlo iteration. More features give more precise power estimates. 5,000 is a good balance of speed and accuracy.
#
# `n_iterations` controls how many Monte Carlo iterations run per design point. More iterations tighten the confidence bands. 10 is fast but noisy. 100 is stable.
#
# `target_power` sets the power level the design search aims for. 0.80 is standard across biomedical research.
#
# `random_seed` makes the simulation deterministic. Same seed, same results. This is critical for reproducibility.

# %%
results = run_power_analysis(
    eq_boundaries=np.array([0.1, 0.3, 0.5, 0.7, 1.0]),
    n_reps_list=[3, 5, 10],
    cv_mean_list=[0.15, 0.25],
    n_prts=5000,
    n_iterations=10,
    target_power=0.80,
    random_seed=42,
    n_jobs=1,
)

# %% [markdown]
# ## Read the summary
#
# The summary shows the design grid dimensions, convergence diagnostics, and the search results. It tells you whether any tested design meets your target power.

# %%
print(results.summary())

# %% [markdown]
# Three sections matter.
#
# "Design ranges" groups the results by parameter. Each group shows the range of values tested, the SEI (Stable Equivalence Index) range, the power range, and how many designs in that group are feasible (meet the target power).
#
# "Recommended designs" shows the minimal design found for each axis. If no design meets the target, it tells you why. The most common reason: the tested values did not reach the target power, which means you need more replicates or a wider equivalence boundary.
#
# "Convergence" tells you whether the SEI estimates are stable at the current iteration count. SEI coefficient of variation below 0.10 is a reasonable threshold.

# %% [markdown]
# ## Inspect the design table
#
# The `design_table()` method pivots the results so you can read power across two axes at once. This is useful for finding the cheapest design that meets your target.

# %%
pivot = results.design_table(row_axis="eq_thr", col_axis="n_reps", metric="power")
print(pivot)

# %% [markdown]
# Each cell shows power for one combination of equivalence boundary and replicate count. Values near 1.0 mean the design reliably recovers equivalent features. Values near 0.20 mean the design is barely better than random.
#
# Read across a row to see how power changes with more replicates at a fixed boundary. Read down a column to see how power changes with a wider boundary at a fixed replicate count.

# %% [markdown]
# ## Find the optimal design
#
# The `optimal_design()` method returns the cheapest design that meets the target power for a given axis. It searches the design grid for the minimal n_reps, minimal eq_thr, or maximal cv_mean that achieves the target.

# %%
best = results.optimal_design("n_reps")
if best:
    print(f"Optimal n_reps: {best}")
else:
    print("No design reached target power in the tested range.")

# %% [markdown]
# The result tells you the recommended value, the power achieved, and the fixed parameter settings used for that search axis. If no design is feasible, you need to widen the search range or lower the target.

# %% [markdown]
# ## Visualize the power profile
#
# The power profile figure shows power on the y-axis against equivalence boundary on the x-axis. Each line is a different replicate count. The shaded band shows the 90 percent quantile range across Monte Carlo iterations. The dashed horizontal line marks the target power.

# %%
import matplotlib.pyplot as plt

fig = results.plot(ci_method="quantile", ci=0.90)
plt.close(fig)
fig

# %% [markdown]
# The figure helps you pick the right combination of eq_thr and n_reps for your experiment. If the line for n_reps=5 crosses the target power at eq_thr=0.5, then five replicates with a 0.5 equivalence boundary meet your design goal.
#
# From here you can adjust the sweep parameters, tighten the CV assumptions, or run a larger simulation with more iterations for tighter confidence bands.
