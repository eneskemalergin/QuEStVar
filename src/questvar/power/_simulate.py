from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_data(
    n_prts: int = 10000,
    n_reps: int = 10,
    int_mu: float = 18.0,
    int_sd: float = 1.0,
    int_log2: bool = True,
    cv_mu: float = 0.275,
    cv_k: float = 2.0,
    cv_theta: float = 0.5,
    seed: int | None = None,
    delta: float = 0.0,
) -> NDArray[np.float64]:
    """Simulate log-normal proteomics data.

    cv_mu is the target mean CV as a **ratio** (e.g. 0.275 for 27.5%).

    delta is the true log2 fold-change applied to the second condition
    (columns n_reps // 2 onward). A value of 0.0 (default) simulates
    pure-equivalence data where both conditions share the same mean.
    Positive delta shifts condition 2 upward in log2 space.

    Seed policy
    -----------
    This function uses ``numpy.random.default_rng(seed)`` which creates a
    fresh local ``Generator``.  It never touches NumPy's global RNG state.
    Passing the same ``seed`` always produces identical data on any platform
    within the same NumPy major version.
    """
    rng = np.random.default_rng(seed)

    mean_dist = rng.normal(int_mu, int_sd, n_prts)
    if int_log2:
        mean_dist = np.power(2.0, mean_dist)
    mean_dist = mean_dist[:, np.newaxis]

    cv_dist = rng.gamma(cv_k, cv_theta, n_prts)
    cv_dist = cv_dist * cv_mu / np.mean(cv_dist)  # scale so mean CV == cv_mu
    cv_dist = cv_dist[:, np.newaxis]

    if np.any(mean_dist == 0):
        raise ValueError(
            "Simulated mean intensities derived from parameters 'int_mu'/'int_sd' "
            "must be non-zero for the log-normal distribution."
        )

    sd_dist = mean_dist * cv_dist
    mu_ln = np.log(mean_dist**2 / np.sqrt(sd_dist**2 + mean_dist**2))
    sigma_ln = np.sqrt(np.log1p(cv_dist**2))
    data = rng.lognormal(mu_ln, sigma_ln, (n_prts, n_reps))
    if delta != 0.0:
        # Shift condition 2 by delta in log2 space: multiply intensities by 2^delta.
        n_half = n_reps // 2
        data[:, n_half:] *= 2.0**delta
    return data.astype(np.float64)
