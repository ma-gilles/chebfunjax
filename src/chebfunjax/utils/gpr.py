# uses-numpy: GP kernel matrix construction and Cholesky solve are iterative/not JIT-safe
"""Gaussian process regression.

Translated from MATLAB Chebfun (commit 7574c77): gpr.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
[1] C. E. Rasmussen & C. K. I. Williams, "Gaussian Processes for Machine
    Learning", MIT Press, 2006.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ===========================================================================
# Public API
# ===========================================================================


def gpr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    domain: Optional[tuple[float, float]] = None,
    sigma: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise: float = 0.0,
    trig: bool = False,
    n_samples: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Gaussian process regression.

    Given data (x_i, y_i), compute the posterior mean and variance
    as dense arrays evaluated on a fine grid, using a squared-exponential
    (RBF) kernel.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input data locations.
    y : array_like, shape (n,)
        Observed function values at x.
    domain : (float, float), optional
        Domain [a, b] for the output grid.  Defaults to [min(x), max(x)].
    sigma : float, optional
        Signal standard deviation (prior amplitude).  Defaults to max(|y|).
    length_scale : float, optional
        RBF kernel length scale L.  If not given, L is chosen to maximise
        the log marginal likelihood (eq. (2.30) from [1]).
    noise : float, optional
        i.i.d. observation noise standard deviation sigma_y (default 0).
    trig : bool, optional
        If True, use the periodic squared-exponential kernel
        k(x, x') = sigma^2 * exp(-2/L^2 * sin^2(pi*(x-x')/P)),
        where P = domain length (eq. (4.31) from [1]).
    n_samples : int, optional
        Number of posterior sample paths to draw (default 0).
    rng : numpy.random.Generator, optional
        Random number generator for sample paths.

    Returns
    -------
    result : dict with keys
        ``"x_grid"``    — shape (M,) evaluation grid on the domain
        ``"mean"``      — shape (M,) posterior mean
        ``"variance"``  — shape (M,) posterior variance (>= 0)
        ``"samples"``   — shape (M, n_samples) posterior samples, or None
        ``"length_scale"`` — the length scale used (float)
        ``"sigma"``     — the signal amplitude used (float)

    Notes
    -----
    The implementation uses the Cholesky-based algorithm (Alg. 2.1 from [1]).
    A small nugget 1e-15 * sigma^2 * n is added to the kernel matrix for
    numerical stability when noise == 0.

    Provenance
    ----------
    MATLAB source : gpr.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.uniform(-2, 2, 10)
    >>> y = np.sin(np.exp(x))
    >>> result = gpr(x, y, domain=(-2.0, 2.0))
    >>> result["mean"].shape
    (200,)
    >>> result["variance"].shape
    (200,)

    See Also
    --------
    aaa
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if rng is None:
        rng = np.random.default_rng()

    # --- scaling ---
    if len(y) > 0 and sigma is None:
        scaling = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    else:
        scaling = 1.0

    sigma_val = scaling if sigma is None else float(sigma)
    sigma_y = float(noise)

    # --- domain ---
    if domain is None:
        if len(x) == 0:
            domain = (-1.0, 1.0)
        elif len(x) == 1:
            domain = (float(x[0]) - 1.0, float(x[0]) + 1.0)
        elif trig:
            diff_ = (float(np.max(x)) - float(np.min(x))) / 10.0
            domain = (float(np.min(x)), float(np.max(x)) + diff_)
        else:
            domain = (float(np.min(x)), float(np.max(x)))
    a, b = float(domain[0]), float(domain[1])
    period = b - a

    # --- optimise length scale if not given ---
    if length_scale is None:
        n = len(x)
        if n == 0:
            length_scale = period / 4.0
        else:
            if trig:
                search_lo = 1.0 / (2.0 * max(n, 1))
                search_hi = 10.0
            else:
                search_lo = period / (2.0 * np.pi * max(n, 1))
                search_hi = 10.0 / np.pi * period
            length_scale = _optimise_length_scale(
                x, y / scaling, sigma_val, sigma_y, trig, period,
                search_lo, search_hi,
            )
    length_scale = float(length_scale)

    # --- output grid ---
    M = 200
    x_grid = np.linspace(a, b, M)

    if len(x) == 0:
        mean = np.zeros(M)
        variance = np.full(M, sigma_val ** 2)
        samples = None if n_samples == 0 else np.zeros((M, n_samples))
        return {
            "x_grid": x_grid,
            "mean": mean,
            "variance": variance,
            "samples": samples,
            "length_scale": length_scale,
            "sigma": sigma_val,
        }

    n = len(x)
    yn = y / scaling

    # --- kernel matrix K(x, x) ---
    K = _kernel(x, x, sigma_val, length_scale, sigma_y, trig, period)
    if sigma_y == 0.0:
        K += 1e-15 * scaling ** 2 * n * np.eye(n)
    else:
        K += sigma_y ** 2 * np.eye(n)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))

    # --- posterior mean on grid ---
    Ks = _kernel(x_grid, x, sigma_val, length_scale, sigma_y, trig, period)
    mean = scaling * (Ks @ alpha)

    # --- posterior variance on grid ---
    Kss_diag = sigma_val ** 2 * np.ones(M)  # diagonal of K(x_grid, x_grid)
    v = np.linalg.solve(L, Ks.T)           # L^{-1} * Ks^T
    variance = Kss_diag - np.sum(v ** 2, axis=0)
    variance = np.maximum(variance, 0.0)

    # --- samples ---
    samples = None
    if n_samples > 0:
        Kss_full = _kernel(x_grid, x_grid, sigma_val, length_scale, 0.0, trig, period)
        Cov = Kss_full - v.T @ v
        Cov += 1e-12 * sigma_val ** 2 * n * np.eye(M)
        L_s = np.linalg.cholesky(Cov)
        Z = rng.standard_normal((M, n_samples))
        samples = scaling * (mean[:, None] / scaling + L_s @ Z)

    return {
        "x_grid": x_grid,
        "mean": mean,
        "variance": variance,
        "samples": samples,
        "length_scale": length_scale,
        "sigma": sigma_val,
    }


# ===========================================================================
# Private helpers
# ===========================================================================


def _kernel(
    xa: np.ndarray,
    xb: np.ndarray,
    sigma: float,
    ell: float,
    sigma_y: float,
    trig: bool,
    period: float,
) -> np.ndarray:
    """Evaluate the (cross-)covariance matrix between xa and xb."""
    xa = np.asarray(xa).ravel()
    xb = np.asarray(xb).ravel()
    # r_{ij} = xa_i - xb_j
    R = xa[:, None] - xb[None, :]
    if trig:
        K = sigma ** 2 * np.exp(
            -2.0 / ell ** 2 * np.sin(np.pi / period * R) ** 2
        )
    else:
        K = sigma ** 2 * np.exp(-0.5 / ell ** 2 * R ** 2)
    # noise cross term (only when xa and xb share the same points)
    if sigma_y != 0.0:
        K += sigma_y ** 2 * (R == 0).astype(float)
    return K


def _log_marginal_likelihood(
    ell: float,
    x: np.ndarray,
    yn: np.ndarray,
    sigma: float,
    sigma_y: float,
    trig: bool,
    period: float,
) -> float:
    """Log marginal likelihood for hyperparameter optimisation (eq. 2.30 from [1])."""
    n = len(x)
    K = _kernel(x, x, sigma, ell, 0.0, trig, period)
    reg = (sigma_y ** 2 if sigma_y != 0.0 else 1e-15 * n * sigma ** 2)
    K += reg * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -1e30
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * yn @ alpha - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi))


def _optimise_length_scale(
    x: np.ndarray,
    yn: np.ndarray,
    sigma: float,
    sigma_y: float,
    trig: bool,
    period: float,
    lo: float,
    hi: float,
    n_grid: int = 50,
) -> float:
    """Find the length scale that maximises the log marginal likelihood."""
    if len(x) == 0:
        return (lo + hi) / 2.0
    ells = np.logspace(np.log10(max(lo, 1e-10)), np.log10(max(hi, lo * 2)), n_grid)
    lmls = np.array([
        _log_marginal_likelihood(e, x, yn, sigma, sigma_y, trig, period)
        for e in ells
    ])
    return float(ells[np.argmax(lmls)])
