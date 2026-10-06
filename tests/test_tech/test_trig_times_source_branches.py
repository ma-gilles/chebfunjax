"""Independent analytic multiplication controls; source tests remain unchanged.

Provenance
----------
MATLAB source : @trigtech/times.m, tests/trigtech/test_times.m
Chebfun commit: 7574c77

The explicit 200*eps absolute bound covers FFT, simplification and evaluation
of these bounded low-degree operands; it is the source scalar-product test
factor, with unit scale and rtol=0. No source output is a constructor input.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = np.finfo(np.float64).eps
X = jnp.asarray([-0.9375, -0.625, -0.125, 0.0625, 0.4375, 0.8125])


def test_complex_square_is_not_modulus_square():
    f = Trigtech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
    h = f * f
    np.testing.assert_allclose(h(X), jnp.exp(2j * jnp.pi * X), rtol=0, atol=200 * EPS)
    assert not h.is_real


def test_negative_product_and_conjugate_product():
    f = Trigtech.from_function(lambda x: 0.5 * jnp.cos(jnp.pi * x))
    h = f * (-f)
    np.testing.assert_allclose(h(X), -0.25 * jnp.cos(jnp.pi * X) ** 2,
                               rtol=0, atol=200 * EPS)
    assert bool(jnp.all(h(X) < 0))
    z = Trigtech.from_function(lambda x: 0.5 * jnp.exp(1j * jnp.pi * x))
    np.testing.assert_allclose((z * z.conj())(X), 0.25, rtol=0, atol=200 * EPS)


def test_array_scalar_column_both_directions():
    f = Trigtech.from_function(
        lambda x: jnp.stack((jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)), axis=-1))
    g = Trigtech.from_function(lambda x: 0.5 * jnp.cos(jnp.pi * x))
    expected = jnp.stack((jnp.cos(jnp.pi * X), jnp.sin(jnp.pi * X)), axis=-1)
    expected = expected * (0.5 * jnp.cos(jnp.pi * X))[:, None]
    for h in (f * g, g * f):
        np.testing.assert_allclose(h(X), expected, rtol=0, atol=200 * EPS)


def test_constant_tech_row_and_numeric_row():
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
    row = jnp.asarray([0.5, 0.25j])
    c = Trigtech.from_values(row[None, :])
    expected = jnp.cos(jnp.pi * X)[:, None] * row[None, :]
    for h in (f * c, c * f, f * row):
        np.testing.assert_allclose(h(X), expected, rtol=0, atol=200 * EPS)


def test_empty_and_incompatible_columns():
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
    assert (f * jnp.asarray([])).isempty()
    assert (Trigtech.empty() * f).isempty()
    two = f * jnp.asarray([1.0, 0.5])
    three = f * jnp.asarray([1.0, 0.5, 0.25])
    with pytest.raises(ValueError, match="dimensions"):
        _ = two * three


def test_equal_representation_ignores_happiness_metadata():
    f = Trigtech.from_function(lambda x: 0.5 * jnp.cos(jnp.pi * x))
    g = Trigtech(coeffs=f.coeffs, is_real=f.is_real, ishappy=False)
    h = f * g
    assert not h.ishappy
    np.testing.assert_allclose(h(X), 0.25 * jnp.cos(jnp.pi * X) ** 2,
                               rtol=0, atol=200 * EPS)
