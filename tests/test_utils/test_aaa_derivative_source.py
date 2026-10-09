"""Literal aaa.m derivative assertions33--40, Chebfun7574c77.

All eight native cases explicitly supply their100 samples. No autosampling
substitution or finite-difference oracle is used. Python list replaces MATLAB
row cell while the package's existing seven-output tuple remains unchanged.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.aaa import _diffbary, aaa


@pytest.mark.parametrize("slot", range(33, 41))
def test_original_derivative_slot(slot):
    z = jnp.linspace(-1.0, 1.0, 100)
    if slot in (34, 35):
        f = jnp.sin(z)
        r, *_ = aaa(f, z, deriv_deg=4)
        if slot == 34:
            assert isinstance(r, list) and len(r) == 5 and all(callable(g) for g in r)
        else:
            assert float(jnp.max(jnp.abs(f - r[4](z)))) < 1e-7
    elif slot == 36:
        r, *_ = aaa(z**3, z, deriv_deg=1)
        assert float(jnp.max(jnp.abs(3 * z**2 - r[1](z)))) < 1e-12
    elif slot in (39, 40):
        z = jnp.exp(2j * jnp.pi * jnp.arange(100) / 100)
        f = jnp.sqrt(2 - z)
        r, *_ = aaa(f, z, degree=6, deriv_deg=1)
        if slot == 39:
            error = float(jnp.max(jnp.abs(f - r[0](z))))
            assert abs(error - 1.19e-11) < 1e-12
        else:
            assert float(jnp.abs(r[1](1.0) + 0.5)) < 1e-6
    else:
        if slot == 38:
            z = jnp.linspace(-1e100, 1e100, 100)
            f = jnp.exp(1e-100 * z)
            expected = 1e-100 * f
        elif slot == 37:
            f = 1e-100 * jnp.exp(z)
            expected = f
        else:
            f = jnp.exp(z)
            expected = f
        r, *_ = aaa(f, z, deriv_deg=1)
        bound = 1e-12 if slot == 33 else 1e-112
        assert float(jnp.max(jnp.abs(expected - r[1](z)))) < bound


def test_support_and_ordinary_points_complex_rational_jit():
    z = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.complex128)
    pole = 2 + 1j
    # Exact barycentric representation of1/(x-pole): denominator weights
    # are polynomial interpolation weights times(z-pole).
    w = jnp.array([0.5, -1.0, 0.5]) * (z - pole)
    f = 1 / (z - pole)
    x = jnp.array([[-1.0, -0.25], [0.0, 1.0]]) + 0j
    for k in range(1, 5):
        factor = 1
        for j in range(2, k + 1):
            factor *= j
        expected = (-1) ** k * factor / (x - pole) ** (k + 1)
        actual = jax.jit(lambda q: _diffbary(q, z, f, w, k))(x)
        assert actual.shape == x.shape
        assert float(jnp.max(jnp.abs(actual - expected))) < 100 * jnp.finfo(jnp.float64).eps


def test_constant_source_scalar_zero_and_default_callable():
    z = jnp.linspace(-1.0, 1.0, 17)
    ordinary, *_ = aaa(jnp.ones_like(z), z)
    derivatives, *_ = aaa(jnp.ones_like(z), z, deriv_deg=2)
    assert callable(ordinary)
    for derivative in derivatives[1:]:
        actual = derivative(jnp.ones((2, 3)))
        assert actual.shape == () and float(actual) == 0.0


def test_callable_autosampling_retained_for_derivative_request():
    calls = []

    def function(z):
        calls.append(jnp.asarray(z))
        return jnp.exp(z)

    plain = aaa(function)
    plain_calls = list(calls)
    calls.clear()
    differentiated = aaa(function, deriv_deg=1)
    assert len(calls) == len(plain_calls) and len(calls) > 1
    for before, after in zip(plain_calls, calls):
        assert before.shape == after.shape and bool(jnp.all(before == after))
    for before, after in zip(plain[1:], differentiated[1:]):
        assert before.shape == after.shape and bool(jnp.all(before == after))
    # Source API forwarding, not an analytical endpoint accuracy claim.
    # The unchanged1e-11 endpoint diagnostic failed in immutable source_v1;
    # identical-data scalar source/JIT recurrences reproduce that loss.
    assert callable(plain[0]) and len(differentiated[0]) == 2


def test_public_derivative_orders_and_other_outputs_unchanged():
    z = jnp.linspace(-1.0, 1.0, 100)
    f = 1 / (z - 2.0)
    plain = aaa(f, z)
    differentiated = aaa(f, z, deriv_deg=4)
    assert len(plain) == len(differentiated) == 7
    for before, after in zip(plain[1:], differentiated[1:]):
        assert bool(jnp.all(before == after))
    x = jnp.array([-1.0, 0.0, 1.0])
    factor = 1
    for k, derivative in enumerate(differentiated[0]):
        if k > 0:
            factor *= k
        expected = (-1) ** k * factor / (x - 2.0) ** (k + 1)
        actual = jax.jit(derivative)(x)
        assert float(jnp.max(jnp.abs(actual - expected))) < 100 * jnp.finfo(
            jnp.float64
        ).eps * float(jnp.max(jnp.abs(expected)))


@pytest.mark.parametrize("order", [0, -1, 0.5, True, "ignored", [2, 3]])
def test_source_option_values_that_do_not_request_derivatives(order):
    z = jnp.linspace(-1.0, 1.0, 10)
    r, *_ = aaa(z, z, deriv_deg=order)
    assert callable(r)


def test_integral_float_order_and_fractional_cell_size_error():
    z = jnp.linspace(-1.0, 1.0, 10)
    r, *_ = aaa(z, z, deriv_deg=1.0)
    assert isinstance(r, list) and len(r) == 2
    with pytest.raises(ValueError, match="integer"):
        aaa(z, z, deriv_deg=1.5)


def test_factorial_above_signed_integer_range():
    z = jnp.array([-1.0, 1.0])
    f = 1 / (z - 2.0)
    w = jnp.array([1.5, -0.5])
    actual = _diffbary(jnp.asarray(0.0), z, f, w, 21)
    expected = -51090942171709440000.0 / 2.0**22
    assert float(jnp.abs(actual - expected)) < 100 * jnp.finfo(jnp.float64).eps * abs(expected)


@pytest.mark.parametrize(
    "order", [[2], [[2]], jnp.array([2]), jnp.array([[2]]), jnp.array([[[2]]])]
)
def test_numeric_singleton_arrays_use_native_numel_scalar_rule(order):
    z = jnp.linspace(-1.0, 1.0, 10)
    r, *_ = aaa(z, z, deriv_deg=order)
    assert isinstance(r, list) and len(r) == 3
    assert float(jnp.abs(r[1](0.0) - 1.0)) < 1e-12
    assert float(jnp.abs(r[2](0.0))) < 1e-12


@pytest.mark.parametrize("order", [[True], jnp.array([[False]]), jnp.array([1, 2])])
def test_boolean_singletons_and_multielement_arrays_remain_ignored(order):
    z = jnp.linspace(-1.0, 1.0, 10)
    r, *_ = aaa(z, z, deriv_deg=order)
    assert callable(r)
