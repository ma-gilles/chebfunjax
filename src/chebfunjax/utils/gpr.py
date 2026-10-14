# uses-numpy: exception class only; covariance and linear algebra are JAX.
"""Gaussian process regression.

Translated from MATLAB Chebfun (commit 7574c77): gpr.m. The finite
covariance and linear algebra path uses JAX arrays. The public mapping retains its
historical dense 200-point grid. The tuple adapter follows the source mean type, variance Chebyshev grid, and nonempty
sample grid and source mean constructor preferences (eps and splitting) while
using package Chebfun constructors. Full native adaptive-fit, optimizer, and
random-stream parity remains unqualified.

Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.

Provenance
----------
MATLAB source: ``gpr.m``. Chebfun commit:
``7574c77680d7e82b79626300bf255498271a72df``.
"""
from __future__ import annotations

import secrets
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from numpy.linalg import LinAlgError


def gpr(
    x,
    y,
    *,
    domain: Optional[tuple[float, float]] = None,
    sigma: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise: float = 0.0,
    trig: bool = False,
    n_samples: int = 0,
    rng=None,
) -> dict:
    """Gaussian process regression on a dense convenience grid.

    Args:
        x, y: One-dimensional training locations and values.
        domain: Output interval. Defaults to the source GPR domain rules.
        sigma: Signal standard deviation; defaults to ``max(abs(y))`` or 1
            when that maximum is zero.
        length_scale: RBF scale. If omitted, use the existing 50-point
            log-spaced likelihood search; that search is not MATLAB optimizer
            parity.
        noise: Observation noise standard deviation.
        trig: Select the periodic squared-exponential kernel.
        n_samples: Number of posterior sample columns.
        rng: Optional external object implementing ``standard_normal(shape)``.
            If omitted, a host-generated seed initializes JAX's PRNG.

    Returns a mapping with the historical ``x_grid`` (200 uniform points),
    ``mean``, nonnegative ``variance``, optional ``samples``, ``sigma`` and
    ``length_scale``. When data are nonempty it also includes callable
    ``mean_fn``/``variance_fn`` adapters and ``domain``. Numeric values are
    JAX arrays; host-side domain/argument validation remains Python.

    MATLAB's fixed-L source path is ``gpr.m`` at Chebfun commit 7574c77.
    This implementation preserves its separate training, mean-query, and
    predictive covariance noise terms, including the source's second
    ``sigmaY**2 * I`` in the noisy training Cholesky. It does not claim
    adaptive Chebfun fit/grid parity.

    Provenance
    ----------
    MATLAB source: ``gpr.m``. Chebfun commit:
    ``7574c77680d7e82b79626300bf255498271a72df``.
    """
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    n = int(x.shape[0])
    if n and sigma is None:
        yscale = float(jnp.max(jnp.abs(y)))
        scaling = yscale if yscale > 0.0 else 1.0
    else:
        scaling = 1.0
    sigma_val = scaling if sigma is None else float(sigma)
    sigma_y = float(noise)

    if domain is not None and len(domain) == 0:
        # MATLAB parseInputs treats an explicitly empty domain as unspecified.
        domain = None

    if domain is None:
        if n == 0:
            domain = (-1.0, 1.0)
        elif n == 1:
            x0 = float(x[0])
            domain = (x0 - 1.0, x0 + 1.0)
        elif trig:
            xmin, xmax = float(jnp.min(x)), float(jnp.max(x))
            domain = (xmin, xmax + (xmax - xmin) / 10.0)
        else:
            domain = (float(jnp.min(x)), float(jnp.max(x)))
    a, b = float(domain[0]), float(domain[1])
    period = b - a

    if length_scale is None:
        if n == 0:
            length_scale = period / 4.0
        elif trig:
            lo, hi = 1.0 / (2.0 * max(n, 1)), 10.0
        else:
            lo = period / (2.0 * jnp.pi * max(n, 1))
            hi = 10.0 / jnp.pi * period
        if n:
            length_scale = float(_optimise_length_scale(
                x, y, sigma_val, sigma_y, bool(trig), period, float(lo), float(hi)
            ))
    length_scale = float(length_scale)

    m = 200
    x_grid = jnp.linspace(a, b, m, dtype=jnp.float64)
    if n == 0:
        zeros = jnp.zeros((m,), dtype=jnp.float64)
        variance = jnp.full((m,), sigma_val**2, dtype=jnp.float64)
        samples = None if n_samples == 0 else jnp.zeros((m, int(n_samples)), dtype=jnp.float64)
        return {
            "x_grid": x_grid,
            "mean": zeros,
            "variance": variance,
            "samples": samples,
            "length_scale": length_scale,
            "sigma": sigma_val,
            "domain": (a, b),
        }

    L = _checked_factor(_training_cholesky(
        x, sigma_val, length_scale, sigma_y, bool(trig), period, scaling
    ))
    alpha = _solve_cholesky(L, y)

    mean = _mean_at(x_grid, x, alpha, sigma_val, length_scale, sigma_y, bool(trig), period)
    variance = jnp.maximum(_predictive_variance(
        x_grid, x, L, sigma_val, length_scale, sigma_y, bool(trig), period
    ), 0.0)

    def mean_fn(xe):
        query = jnp.asarray(xe, dtype=jnp.float64).reshape(-1)
        return _mean_at(query, x, alpha, sigma_val, length_scale, sigma_y, bool(trig), period)

    def variance_fn(xe):
        query = jnp.asarray(xe, dtype=jnp.float64).reshape(-1)
        return jnp.maximum(_predictive_variance(
            query, x, L, sigma_val, length_scale, sigma_y, bool(trig), period
        ), 0.0)

    samples = None
    if n_samples > 0:
        covariance = _predictive_covariance(
            x_grid, x, L, sigma_val, length_scale, sigma_y, bool(trig), period
        )
        cov_for_samples = covariance + (1e-12 * scaling**2 * n) * jnp.eye(m, dtype=jnp.float64)
        Ls = _checked_cholesky(cov_for_samples)
        shape = (m, int(n_samples))
        if rng is None:
            key = jax.random.PRNGKey(secrets.randbits(32))
            z = jax.random.normal(key, shape, dtype=jnp.float64)
        else:
            z = jnp.asarray(rng.standard_normal(shape), dtype=jnp.float64)
        samples = mean[:, None] + Ls @ z

    return {
        "x_grid": x_grid,
        "mean": mean,
        "variance": variance,
        "samples": samples,
        "length_scale": length_scale,
        "sigma": sigma_val,
        "mean_fn": mean_fn,
        "variance_fn": variance_fn,
        "domain": (a, b),
    }


def gpr_chebfun(x, y, **kwargs):
    """Construct MATLAB-style mean, variance, and sample Chebfuns.

    The fixed-L numerical callbacks come from :func:`gpr`; this adapter follows
    the source construction choices for mean type, variance nodes, and sample
    nodes. It does not claim adaptive optimizer or RNG-stream parity.

    Provenance
    ----------
    MATLAB source: gpr.m. Chebfun commit:
    7574c77680d7e82b79626300bf255498271a72df.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.utils.quadrature import chebpts_ab

    n_samples = int(kwargs.get("n_samples", 0))
    call_kwargs = dict(kwargs)
    call_kwargs["n_samples"] = 0
    rng = call_kwargs.pop("rng", None)
    res = gpr(x, y, **call_kwargs)
    a, b = res["domain"]
    trig = bool(kwargs.get("trig", False))
    noisy = float(kwargs.get("noise", 0.0)) != 0.0
    if "mean_fn" not in res:
        f = chebfun(lambda t: jnp.zeros_like(t), domain=(a, b))
        fvar = chebfun(lambda t: jnp.full_like(t, res["sigma"] ** 2), domain=(a, b))
        return f, fvar, None

    f = chebfun(
        lambda t: jnp.asarray(res["mean_fn"](jnp.asarray(t))),
        domain=(a, b), trig=(trig and not noisy), eps=1e-12, splitting=True,
    )
    # Source gpr.m samples q=min(20*n,2000) Chebyshev points, constructs a
    # non-trig Chebfun from those values, then simplifies it.
    q = min(20 * int(jnp.asarray(x).size), 2000)
    qnodes = chebpts_ab(q, a, b)
    x_array = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y_array = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    sigma_y = float(kwargs.get("noise", 0.0))
    scale = (float(jnp.max(jnp.abs(y_array)))
             if kwargs.get("sigma") is None and y_array.size
             and float(jnp.max(jnp.abs(y_array))) > 0.0 else 1.0)
    factor = _training_cholesky(
        x_array, float(res["sigma"]), float(res["length_scale"]), sigma_y,
        trig, b - a, scale,
    )
    # MATLAB takes spdiags(Kss-v'*v) directly. Preserve negative roundoff
    # values rather than applying the convenience callback's nonnegative cap.
    qvalues = _predictive_variance(
        qnodes, x_array, factor, float(res["sigma"]),
        float(res["length_scale"]), sigma_y, trig, b - a,
    )
    fvar = chebfun(jnp.asarray(qvalues), domain=(a, b)).simplify()

    fsamples = None
    if n_samples > 0:
        xs, ys = _gpr_source_samples(
            x, y, res, kwargs, rng, trig, n_samples, mean_fun=f
        )
        fsamples = chebfun(ys, domain=(a, b))
    return f, fvar, fsamples


def _gpr_source_samples(x, y, result, options, rng, trig, count, *, mean_fun):
    """Draw samples using source node layout and covariance jitter scaling.

    MATLAB and JAX random streams differ; no sample identity claim is made.
    """
    import jax

    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    a, b = result["domain"]
    n = int(x.size)
    q = min(20 * n, 2000)
    from chebfunjax.utils.quadrature import chebpts_ab
    xs = chebpts_ab(q, a, b)
    if n == 0:
        return xs, jnp.zeros((q, count), dtype=jnp.float64)

    sigma = float(result["sigma"])
    ell = float(result["length_scale"])
    sigma_y = float(options.get("noise", 0.0))
    trig = bool(trig)
    period = b - a
    K = _training_cholesky(
        x, sigma, ell, sigma_y, trig, period,
        (float(jnp.max(jnp.abs(y))) if options.get("sigma") is None
         and float(jnp.max(jnp.abs(y))) > 0.0 else 1.0),
    )
    # The source samples from f(xSample), where f is the fitted Chebfun mean.
    mean = jnp.asarray(mean_fun(xs), dtype=jnp.float64).reshape(-1)
    covariance = _predictive_covariance(xs, x, K, sigma, ell, sigma_y, trig, period)
    scale = (float(jnp.max(jnp.abs(y))) if options.get("sigma") is None
             and float(jnp.max(jnp.abs(y))) > 0.0 else 1.0)
    covariance = covariance + 1e-12 * scale**2 * n * jnp.eye(q, dtype=jnp.float64)
    factor = _checked_cholesky(covariance)
    if rng is None:
        seed = secrets.randbits(32)
        noise = jax.random.normal(
            jax.random.PRNGKey(seed), (q, count), dtype=jnp.float64
        )
    else:
        noise = jnp.asarray(rng.standard_normal((q, count)), dtype=jnp.float64)
    return xs, mean[:, None] + factor @ noise


def _base_covariance(xa, xb, sigma, ell, trig, period):
    """Signal-only covariance; no observation noise is applied here."""
    xa = jnp.asarray(xa, dtype=jnp.float64).reshape(-1)
    xb = jnp.asarray(xb, dtype=jnp.float64).reshape(-1)
    r = xa[:, None] - xb[None, :]
    if trig:
        return sigma**2 * jnp.exp(-2.0 / ell**2 * jnp.sin(jnp.pi / period * r) ** 2)
    return sigma**2 * jnp.exp(-0.5 / ell**2 * r**2)


def _training_covariance(x, sigma, ell, sigma_y, trig, period, scaling):
    """Source training covariance before the separate Cholesky noise term."""
    n = x.shape[0]
    base = _base_covariance(x, x, sigma, ell, trig, period)
    return base + sigma_y**2 * jnp.eye(n, dtype=jnp.float64)


def _training_cholesky(x, sigma, ell, sigma_y, trig, period, scaling):
    n = x.shape[0]
    kernel = _training_covariance(x, sigma, ell, sigma_y, trig, period, scaling)
    if sigma_y == 0.0:
        kernel = kernel + 1e-15 * scaling**2 * n * jnp.eye(n, dtype=jnp.float64)
    else:
        # gpr.m forms K with sigmaY^2 I, then adds sigmaY^2 I again to chol.
        kernel = kernel + sigma_y**2 * jnp.eye(n, dtype=jnp.float64)
    return jnp.linalg.cholesky(kernel)


def _solve_cholesky(L, rhs):
    lower = solve_triangular(L, rhs, lower=True)
    return solve_triangular(L.T, lower, lower=False)


def _checked_cholesky(matrix):
    """Eager adapter preserving NumPy's exception for failed factors."""
    return _checked_factor(jnp.linalg.cholesky(matrix))


def _checked_factor(factor):
    """Check a computed factor outside the JIT-compatible numeric core."""
    if not bool(jnp.all(jnp.isfinite(factor))):
        raise LinAlgError("Matrix is not positive definite")
    return factor


def _coincidence_mask(xa, xb):
    return xa.reshape(-1, 1) == xb.reshape(1, -1)


def _mean_covariance(query, x, sigma, ell, sigma_y, trig, period):
    # Source mean() uses sigmaY (not sigmaY^2) at exact training coincidences.
    return _base_covariance(query, x, sigma, ell, trig, period) + sigma_y * _coincidence_mask(query, x)


def _prediction_cross_covariance(query, x, sigma, ell, sigma_y, trig, period):
    return _base_covariance(query, x, sigma, ell, trig, period) + sigma_y**2 * _coincidence_mask(query, x)


def _prediction_self_covariance(query, x, sigma, ell, sigma_y, trig, period):
    in_x = jnp.any(query.reshape(-1, 1) == x.reshape(1, -1), axis=1)
    return _base_covariance(query, query, sigma, ell, trig, period) + sigma_y**2 * jnp.diag(in_x.astype(jnp.float64))


def _mean_at(query, x, alpha, sigma, ell, sigma_y, trig, period):
    return _mean_covariance(query, x, sigma, ell, sigma_y, trig, period) @ alpha


def _predictive_variance(query, x, L, sigma, ell, sigma_y, trig, period):
    """Predictive diagonal without allocating the full query covariance."""
    cross = _prediction_cross_covariance(query, x, sigma, ell, sigma_y, trig, period)
    v = solve_triangular(L, cross.T, lower=True)
    in_x = jnp.any(query.reshape(-1, 1) == x.reshape(1, -1), axis=1)
    prior_diag = sigma**2 + sigma_y**2 * in_x.astype(jnp.float64)
    return prior_diag - jnp.sum(v**2, axis=0)


def _predictive_covariance(query, x, L, sigma, ell, sigma_y, trig, period):
    cross = _prediction_cross_covariance(query, x, sigma, ell, sigma_y, trig, period)
    v = solve_triangular(L, cross.T, lower=True)
    return _prediction_self_covariance(query, x, sigma, ell, sigma_y, trig, period) - v.T @ v


def _log_marginal_likelihood(ell, x, y, sigma, sigma_y, trig, period, scaling):
    # Preserve the current 50-grid objective: it adds one noise diagonal, and
    # uses sigma^2 for its zero-noise nugget. It is not the source gpr.m
    # training Cholesky, which adds the second noise diagonal in that branch.
    n = x.shape[0]
    kernel = _base_covariance(x, x, sigma, ell, trig, period)
    if sigma_y != 0.0:
        kernel = kernel + sigma_y**2 * jnp.eye(n, dtype=jnp.float64)
    else:
        kernel = kernel + 1e-15 * n * sigma**2 * jnp.eye(n, dtype=jnp.float64)
    L = jnp.linalg.cholesky(kernel)
    alpha = _solve_cholesky(L, y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    value = -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * jnp.log(2.0 * jnp.pi)
    return jnp.nan_to_num(value, nan=-1e30, posinf=-1e30, neginf=-1e30)


def _optimise_length_scale(x, y, sigma, sigma_y, trig, period, lo, hi, n_grid=50):
    """Retain the old 50-point search in JAX; no MATLAB optimizer parity claim."""
    lo = max(float(lo), float(jnp.finfo(jnp.float64).tiny))
    hi = max(float(hi), lo * 2.0)
    ells = jnp.logspace(jnp.log10(lo), jnp.log10(hi), int(n_grid), dtype=jnp.float64)
    scale = float(jnp.max(jnp.abs(y))) if y.shape[0] else 1.0
    scale = scale if scale > 0.0 else 1.0
    values = jax.vmap(lambda ell: _log_marginal_likelihood(
        ell, x, y, sigma, sigma_y, trig, period, scale
    ))(ells)
    return ells[jnp.argmax(values)]
