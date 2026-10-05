"""Independent tests for half-integer Bessel prerequisites.

Diagnostic bounds only; these are not MATLAB assertions. Sources:
DLMF 10.16.1 (closed forms), 10.2.2 (power series), 10.6.1 (recurrence),
and pinned Chebfun ``lagpts.m``/``besselroots.m`` at commit 7574c776.
"""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.special import jv, jvp

from chebfunjax.utils.bessel_half import (
    _bessel_j_half,
    _bessel_jhalf_orders,
    _bessel_roots_half,
)

ORDERS = (-1.5, -0.5, 0.5, 1.5)
EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("order", ORDERS)
def test_half_integer_values_near_zero_use_stable_series(order):
    # mpmath receives each exact binary64 input promoted at high precision.
    xs = np.array([1.0e-12, 1.0e-8, 1.0e-4, 0.01, 0.5, np.nextafter(1.0, 0.0)])
    with mp.workdps(80):
        expected = np.array([
            float(mp.besselj(mp.mpf(order), mp.mpf(float(x)))) for x in xs
        ])
    actual = np.asarray(_bessel_j_half(order, jnp.asarray(xs)))
    # Same elementwise rtol/atol envelope; NumPy2.4 assertion formats atol as scalar.
    bound = 128.0 * EPS * np.maximum(1.0, np.abs(expected)) + 128.0 * EPS * np.abs(expected)
    error = np.abs(actual-expected)
    assert np.all(error <= bound), (error, bound)


@pytest.mark.parametrize("order", ORDERS)
def test_half_integer_closed_forms_against_scipy(order):
    xs = np.array([1.0, np.nextafter(1.0, np.inf), 1.5, 3.0, 12.0, 100.0, 1.0e5])
    actual = np.asarray(_bessel_j_half(order, jnp.asarray(xs)))
    expected = jv(order, xs)
    # Same elementwise rtol/atol envelope; NumPy2.4 assertion formats atol as scalar.
    bound = 128.0 * EPS * np.maximum(1.0, np.abs(expected)) + 128.0 * EPS * np.abs(expected)
    error = np.abs(actual-expected)
    assert np.all(error <= bound), (error, bound)


@pytest.mark.parametrize("order", ORDERS)
def test_half_integer_jvp_against_scipy_derivative(order):
    xs = (1.0e-5, 0.1, 0.75, 1.0, 2.0, 20.0, 1.0e3)
    for x in xs:
        value, derivative = jax.jvp(
            lambda z: _bessel_j_half(order, z),
            (jnp.float64(x),), (jnp.float64(1.0),))
        reference_value = float(jv(order, x))
        reference_derivative = float(jvp(order, x, n=1))
        scale = max(1.0, abs(reference_value), abs(reference_derivative))
        assert abs(float(value) - reference_value) <= 128.0 * EPS * scale
        assert abs(float(derivative) - reference_derivative) <= 256.0 * EPS * scale


def test_half_integer_shapes_jit_vmap_and_bundle_order():
    xs = jnp.asarray([[0.2, 0.8], [1.0, 4.0]])
    eager = _bessel_jhalf_orders(xs)
    compiled = jax.jit(_bessel_jhalf_orders)(xs)
    vmapped = tuple(jax.vmap(lambda z, nu=nu: _bessel_j_half(nu, z))(
        xs.ravel()).reshape(xs.shape) for nu in ORDERS)
    for a, b, c in zip(eager, compiled, vmapped, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=16.0 * EPS, atol=16.0 * EPS)
        np.testing.assert_allclose(np.asarray(a), np.asarray(c),
                                   rtol=16.0 * EPS, atol=16.0 * EPS)


def test_half_integer_zero_limits_and_invalid_arguments():
    j_m3, j_m1, j_p1, j_p3 = _bessel_jhalf_orders(
        jnp.asarray([0.0, -1.0, jnp.inf, jnp.nan]))
    assert np.isneginf(float(j_m3[0]))
    assert np.isposinf(float(j_m1[0]))
    assert float(j_p1[0]) == 0.0
    assert float(j_p3[0]) == 0.0
    for values in (j_m3, j_m1, j_p1, j_p3):
        assert np.isnan(np.asarray(values[1:])).all()
    with pytest.raises(TypeError, match="real arguments only"):
        _bessel_j_half(0.5, jnp.asarray(1.0 + 1.0j))


@pytest.mark.parametrize("order", (-0.5, 0.5))
def test_half_order_roots_follow_exact_closed_forms(order):
    n = 12
    roots = np.asarray(_bessel_roots_half(order, n))
    k = np.arange(1, n + 1, dtype=np.float64)
    expected = (k - 0.5) * np.pi if order == -0.5 else k * np.pi
    # Preserve the original elementwise8EPS envelope without ndarray atol formatting.
    error = np.abs(roots-expected)
    bound = 8.0 * EPS * expected
    assert np.all(error <= bound), (error, bound)
    # Verify these are genuine zeros with an independent special-function
    # implementation; the MATLAB Piessens root approximations are not used.
    np.testing.assert_allclose(jv(order, roots), 0.0, rtol=0.0, atol=32.0 * EPS)


def test_half_order_empty_roots_and_invalid_orders_counts():
    assert _bessel_roots_half(-0.5, 0).shape == (0,)
    with pytest.raises(ValueError, match=r"orders \+/-1/2"):
        _bessel_roots_half(1.5, 2)
    with pytest.raises(ValueError, match="nonnegative integer"):
        _bessel_roots_half(0.5, -1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        _bessel_roots_half(0.5, 1.5)
    with pytest.raises(ValueError, match="unsupported half-integer"):
        _bessel_j_half(2.5, jnp.asarray(1.0))

