"""Scratch port of MATLAB ``test_power.m`` source assertions 39–41.

Original source formulas and bounds are retained. NumPy MT19937 is a
reproducible stream adapter; exact MATLAB random stream parity remains open.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, passes 39–41
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


def _source_array_sample(seed: int = 6178):
    """Reproducible Python adapter for source ``sort(2*rand(100)-1)``.

    RandomState/MT19937 is not claimed to reproduce MATLAB ``seedRNG``. The
    preceding draws preserve the source test's random-call count through pass
    38; exact stream parity remains an environment-specific oracle gap.
    """
    rng = np.random.RandomState(seed)
    def draw_source_matrix():
        # MATLAB rand(100) is 100x100 and fills columns first. Reproduce that
        # shape/order around the non-equivalent Python MT19937 stream.
        return rng.rand(10_000).reshape((100, 100), order="F")

    rng.rand(100)  # passes 31–33: rand(100,1)
    rng.rand(100)  # passes 34–35: rand(100,1)
    rng.rand(100)  # passes 36–38: rand(100,1)
    return np.sort(2.0 * draw_source_matrix() - 1.0, axis=0)


def _source_max_vscale(result):
    """Python scalar adapter for MATLAB's ``max(eps*vscale(g))``."""
    if hasattr(result, "cols"):
        per_column = [float(np.asarray(col.vscale)) for col in result.cols]
        return max(per_column)
    return float(np.max(np.asarray(result.vscale)))


def _matrix_inf_norm(values):
    """MATLAB ``norm(A,inf)``: maximum row sum of magnitudes."""
    return float(np.linalg.norm(np.asarray(values), ord=np.inf))


def _pass39_result_and_tolerance():
    powers = jnp.asarray(((-0.1, 0.3),))
    base = cj.chebfun(
        lambda t: jnp.stack((t, t + 0.5), axis=-1),
        domain=(-1.0, 1.5),
    )
    result = base**powers
    tol = 1.0e4 * EPS * _source_max_vscale(result)
    assert np.isfinite(tol) and tol > 0
    return result, tol


def test_source_pass39_array_chebfun_to_matched_fractional_power_columns():
    sample = jnp.asarray(_source_array_sample())
    powers = jnp.asarray(((-0.1, 0.3),))
    base = cj.chebfun(
        lambda t: jnp.stack((t, t + 0.5), axis=-1),
        domain=(-1.0, 1.5),
    )
    result = base**powers
    # Source @chebfun/power.m splits matching array-valued columns and returns
    # a Quasimatrix, keeping their branch/singularity structure independent.
    assert result.shape == ("inf", 2)
    z = sample.astype(jnp.complex128)
    exact = jnp.concatenate((jnp.power(z, powers[0, 0]),
                             jnp.power(z + 0.5, powers[0, 1])), axis=1)
    # MATLAB feval(quasimatrix, matrix) horizontally joins each column's
    # same-shaped result, so 100x100 input yields 100x200 output.
    assert result(sample).shape == (100, 200)
    err = _matrix_inf_norm(result(sample) - exact)
    tol = 1.0e4 * EPS * _source_max_vscale(result)
    assert np.isfinite(tol) and tol > 0
    print(f"source39 matrix error={err:.17g} original_bound={tol:.17g}")
    assert err < tol


def test_source_pass40_scalar_chebfun_to_integer_power_vector():
    _, source_tol = _pass39_result_and_tolerance()
    x = cj.chebfun(lambda t: t)
    powers = jnp.arange(6)[None, :]
    result = x**powers
    expected = cj.chebfun(
        lambda t: jnp.stack([t**k for k in range(6)], axis=-1)
    )
    # MATLAB norm(f-g,inf) for array-valued Chebfuns is the continuous
    # maximum row 1-norm; retain the source's inherited pass39 tolerance.
    error = float((result - expected).norm(jnp.inf))
    print(f"continuous norm error={error:.17g} inherited_bound={source_tol:.17g}")
    assert error < source_tol


def test_source_pass41_scalar_chebfun_to_mixed_power_vector():
    _, source_tol = _pass39_result_and_tolerance()
    x = cj.chebfun(lambda t: t)
    powers = jnp.asarray(((-2.0, -0.5, 1.25, 3.0),))
    result = (1.0 + x**2)**powers
    expected = cj.chebfun(
        lambda t: jnp.stack([(1.0 + t**2)**p for p in powers[0]], axis=-1)
    )
    error = float((result - expected).norm(jnp.inf))
    print(f"continuous norm error={error:.17g} inherited_bound={source_tol:.17g}")
    assert error < source_tol
