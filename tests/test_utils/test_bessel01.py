"""Independent checks for the bounded JAX J0/J1 Laguerre prerequisite.

These are candidate diagnostic bounds, not original MATLAB assertions.
Provenance: DLMF 10.17.E1–E3; downstream motivation is
``lagpts.m:800`` (`asyBessel`, Chebfun source commit 7574c776).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

from chebfunjax.utils.bessel01 import _bessel_j01


def test_j01_matches_scipy_through_middle_branch_and_boundaries():
    eps = np.finfo(np.float64).eps
    x = np.unique(np.concatenate((
        np.array([0.0, np.nextafter(1.0, 0.0), 1.0,
                  np.nextafter(1.0, 2.0),
                  np.nextafter(32.0, 0.0), 32.0,
                  np.nextafter(32.0, np.inf)]),
        np.geomspace(1.0e-12, 1.0, 100),
        np.linspace(1.0, 32.0, 300),
    )))
    got0, got1 = _bessel_j01(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(got0), jv(0, x), rtol=0.0,
                               atol=64.0 * eps)
    np.testing.assert_allclose(np.asarray(got1), jv(1, x), rtol=0.0,
                               atol=64.0 * eps)


def test_j01_far_asymptotic_phase_conditioned_error():
    eps = np.finfo(np.float64).eps
    x = np.geomspace(32.0, 1.0e5, 500)
    got0, got1 = _bessel_j01(jnp.asarray(x))
    scale = np.sqrt(2.0 / (np.pi * x))
    bound = 16.0 * eps * (1.0 + x) * scale
    np.testing.assert_array_less(np.abs(np.asarray(got0) - jv(0, x)), bound)
    np.testing.assert_array_less(np.abs(np.asarray(got1) - jv(1, x)), bound)


def test_j01_zero_and_invalid_arguments():
    j0, j1 = _bessel_j01(jnp.asarray([0.0, -1.0, jnp.inf, jnp.nan]))
    np.testing.assert_array_equal(np.asarray(j0[:1]), np.array([1.0]))
    np.testing.assert_array_equal(np.asarray(j1[:1]), np.array([0.0]))
    assert np.isnan(np.asarray(j0[1:])).all()
    assert np.isnan(np.asarray(j1[1:])).all()


def test_j01_jit_vmap_and_derivative_identities():
    xs = jnp.asarray([0.0, 0.2, 2.0, 32.0, 80.0])
    compiled = jax.jit(_bessel_j01)
    got0, got1 = compiled(xs)
    ref0, ref1 = jax.vmap(lambda z: _bessel_j01(z))(xs)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(np.asarray(got0), np.asarray(ref0),
                               rtol=16.0 * eps, atol=16.0 * eps)
    np.testing.assert_allclose(np.asarray(got1), np.asarray(ref1),
                               rtol=16.0 * eps, atol=16.0 * eps)

    for x in (0.0, 0.2, 1.0, 2.0, 12.0, 32.0,
              np.nextafter(32.0, np.inf), 48.0, 1000.0):
        (j0, j1), (dj0, dj1) = jax.jvp(
            _bessel_j01, (jnp.float64(x),), (jnp.float64(1.0),))
        expected0 = -j1
        expected1 = 0.5 if x == 0.0 else j0 - j1 / x
        assert abs(float(dj0 - expected0)) <= 64.0 * eps
        assert abs(float(dj1 - expected1)) <= 128.0 * eps


@pytest.mark.parametrize("x", [-1.0, -1.0e-12])
def test_j01_negative_arguments_are_nan(x):
    values = _bessel_j01(jnp.asarray(x))
    assert all(np.isnan(float(value)) for value in values)
