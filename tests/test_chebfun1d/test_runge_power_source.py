"""MATLAB Runge power source operations 25 and 26.

The source
resets its RNG before creating a 100-point sample; these tests use NumPy
RandomState/MT19937 seed 6178 and explicitly does not claim MATLAB stream
equivalence.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, passes 25 and 26
Chebfun commit: 7574c77
P construction API: ``chebfun(f, *, domain=...)`` (domain is keyword-only)
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
DOMAIN = (-2.0, 7.0)


def _source_sample(seed=6178):
    """MATLAB pass25 sampling expression using a Python MT19937 adapter."""
    rng = np.random.RandomState(seed)
    return (DOMAIN[1] - DOMAIN[0]) * rng.rand(100) + DOMAIN[0]


def _sample_max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def test_source_pass25_runge_positive_fractional_power():
    exponent = 0.6
    f = cj.chebfun(
        lambda x: 1.0 / (1.0 + 25.0 * x**2),
        domain=DOMAIN,
        splitting=True,
    )
    result = f**exponent
    x = jnp.asarray(_source_sample())
    exact = 1.0 / (1.0 + 25.0 * x**2) ** exponent
    err = _sample_max_error(result(x), exact)
    scale = float(np.max(np.abs(np.asarray(exact))))
    assert err < 1e1 * EPS * scale


def test_source_pass26_runge_negative_fractional_power():
    exponent = -1.6
    f = cj.chebfun(
        lambda x: 1.0 / (1.0 + 25.0 * x**2),
        domain=DOMAIN,
        splitting=True,
    )
    result = f**exponent
    x = jnp.asarray(_source_sample())
    exact = 1.0 / (1.0 + 25.0 * x**2) ** exponent
    err = _sample_max_error(result(x), exact)
    scale = float(np.max(np.abs(np.asarray(exact))))
    assert err < 1e3 * EPS * scale
