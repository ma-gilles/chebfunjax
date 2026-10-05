"""Scratch source-derived tests for MATLAB Chebfun power passes 27–30.

Sampling uses NumPy RandomState(MT19937) seed 6178 as a deterministic Python
adapter for the source's 100 random points; it does not claim MATLAB
``seedRNG`` stream equivalence.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, passes 27–30
MATLAB APIs  : @chebfun/power.m, @chebtech/power.m
Chebfun commit: 7574c77
Original copyright: 2017 The University of Oxford and Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
DOMAIN = (-2.0, 7.0)


def _source_sample_adapter():
    """Source expression diff(dom)*rand(100,1)+dom(1), adapted stream."""
    rng = np.random.RandomState(6178)
    return (DOMAIN[1] - DOMAIN[0]) * rng.rand(100) + DOMAIN[0]


def _max_abs_error(actual, exact):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(exact), ord=np.inf))


def _max_abs_scale(exact):
    return float(np.linalg.norm(np.asarray(exact), ord=np.inf))


def _assert_original_bound(actual, exact, multiplier):
    error = _max_abs_error(actual, exact)
    bound = multiplier * EPS * _max_abs_scale(exact)
    assert np.isfinite(bound) and bound > 0
    print(f"source sample error={error:.17g} original_bound={bound:.17g}")
    assert error < bound


def _signed_sine_power_reference(x, exponent):
    # MATLAB's real negative base to noninteger POWER follows the principal
    # complex branch; cast before exponentiation to avoid NumPy's real NaNs.
    values = np.sin(10.0 * np.asarray(x)).astype(np.complex128)
    return values ** exponent


def test_source_pass27_varying_sign_positive_fractional_power():
    exponent = 0.8
    f = cj.chebfun(lambda x: jnp.sin(10.0 * x),
                   domain=DOMAIN, splitting=True)
    result = f ** exponent
    x = jnp.asarray(_source_sample_adapter())
    exact = _signed_sine_power_reference(x, exponent)
    _assert_original_bound(result(x), exact, 1e4)


def test_source_pass28_varying_sign_negative_fractional_power():
    exponent = -0.65
    f = cj.chebfun(lambda x: jnp.sin(10.0 * x),
                   domain=DOMAIN, splitting=True)
    result = f ** exponent
    x = jnp.asarray(_source_sample_adapter())
    exact = _signed_sine_power_reference(x, exponent)
    _assert_original_bound(result(x), exact, 2e4)


def _complex_smooth_values(x):
    return jnp.sin(3.0 * x) + 1j * jnp.cos(2.0 * x)


def _complex_power_reference(x, exponent):
    values = np.sin(3.0 * np.asarray(x)) + 1j * np.cos(2.0 * np.asarray(x))
    return values ** exponent


def test_source_pass29_complex_smooth_positive_power():
    exponent = 1.7
    f = cj.chebfun(_complex_smooth_values, domain=DOMAIN, splitting=True)
    result = f ** exponent
    x = jnp.asarray(_source_sample_adapter())
    exact = _complex_power_reference(x, exponent)
    _assert_original_bound(result(x), exact, 1e2)


def test_source_pass30_complex_smooth_negative_power():
    exponent = -0.77
    f = cj.chebfun(_complex_smooth_values, domain=DOMAIN, splitting=True)
    result = f ** exponent
    x = jnp.asarray(_source_sample_adapter())
    exact = _complex_power_reference(x, exponent)
    _assert_original_bound(result(x), exact, 1e2)
