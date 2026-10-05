"""Independent Airy oracle, differentiation, and ODE checks.

Provenance
----------
MATLAB source : hermpts.m (built-in airy calls)
Chebfun commit: 7574c77
Defining equation and initial values: https://dlmf.nist.gov/9.2
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import airy

from chebfunjax.utils.airy import _airy_negative

EPS = np.finfo(float).eps


def test_local_airy_matches_independent_special_function():
    x = np.linspace(-16.0, 0.0, 1001)
    ai, aip = _airy_negative(x)
    expected_ai, expected_aip, _, _ = airy(x)
    # Both implementations accumulate binary64 errors across the oscillations.
    # This is an absolute bound, including the many zero crossings.
    npt.assert_allclose(ai, expected_ai, atol=64 * EPS, rtol=0)
    npt.assert_allclose(aip, expected_aip, atol=128 * EPS, rtol=0)


def test_oscillatory_airy_matches_oracle_with_phase_rounding_bound():
    x = -np.geomspace(16.001, 10000.0, 1001)
    ai, aip = map(np.asarray, jax.jit(_airy_negative)(jnp.asarray(x)))
    expected_ai, expected_aip, _, _ = airy(x)
    phase = (2 / 3) * (-x)**1.5
    amplitude = (-x)**0.25 / np.sqrt(np.pi)
    # Large oscillatory arguments amplify phase-rounding error. Scaling by
    # phase and envelope is meaningful at zeros, unlike a relative bound.
    bound = 16 * EPS * (1 + phase)
    assert np.all(np.abs(ai - expected_ai) <= bound / (np.sqrt(np.pi) * (-x)**0.25))
    assert np.all(np.abs(aip - expected_aip) <= bound * amplitude)


@pytest.mark.parametrize('x', [0.0, -1.0, -7.234, -15.75, -16.0, -16.001, -100.0, -1000.0])
def test_airy_derivative_and_defining_ode(x):
    ai, aip = _airy_negative(x)
    derivative = jax.grad(lambda z: _airy_negative(z)[0])
    got_first = derivative(x)
    got_second = jax.grad(derivative)(x)
    npt.assert_allclose(got_first, aip, atol=64 * EPS * max(1, abs(float(aip))), rtol=0)
    npt.assert_allclose(got_second, x * ai,
                        atol=64 * EPS * max(1, abs(float(x * ai))), rtol=0)


def test_initial_values_shapes_and_invalid_positive_argument():
    ai, aip = _airy_negative(np.zeros((2, 3)))
    assert ai.shape == aip.shape == (2, 3)
    npt.assert_array_equal(ai, np.full((2, 3), 0.35502805388781723926))
    npt.assert_array_equal(aip, np.full((2, 3), -0.25881940379280679841))
    assert all(np.isnan(float(value)) for value in _airy_negative(1.0))
