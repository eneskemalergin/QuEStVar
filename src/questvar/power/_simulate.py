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

    Parameters
    ----------
    n_prts : int
        Number of features. Default 10000.
    n_reps : int
        Number of replicates per condition. Default 10.
    int_mu : float
        Mean log-intensity. Default 18.0.
    int_sd : float
        Log-intensity standard deviation. Default 1.0.
    int_log2 : bool
        If True, mean_dist is computed as 2**mean_dist. Default True.
    cv_mu : float
        Target mean CV as a ratio. Default 0.275.
    cv_k : float
        Gamma shape for CV distribution. Default 2.0.
    cv_theta : float
        Gamma scale for CV distribution. Default 0.5.
    seed : int, optional
        Random seed for deterministic generation.
    delta : float
        True log2 fold-change for condition 2. Default 0.0.

    Returns
    -------
    ndarray
        Simulated intensity matrix, shape (n_prts, n_reps).

    Raises
    ------
    ValueError
        If mean intensities are zero (log-normal undefined).

    Notes
    -----
    Seed policy and CV scaling are documented below.
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
