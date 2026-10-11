"""Supplemental source-body integration controls; no native test_norm.m exists."""

import warnings

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v, _chebtech_integral, _integral_separable
from chebfunjax.tech.chebtech import Chebtech2


def test_integral_warning_control():
    f = Chebtech2(jnp.array([1.0, 1.0, 1.0]))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = _chebtech_integral(f)
    assert jnp.array_equal(jnp.asarray(actual), f.sum())
    assert not seen


def test_complex_integral_control():
    f = Chebfun2.from_function(lambda x, y: (1 + 2j) * (1 + x + y), domain=(-2.0, 3.0, -1.0, 2.0))
    actual = _integral_separable(f.approx)
    assert jnp.allclose(
        actual, f.sum2(), rtol=0, atol=100 * jnp.finfo(jnp.float64).eps * jnp.abs(f.sum2())
    )


def test_periodic_norm_control():
    f = Chebfun2.from_function(lambda x, y: jnp.cos(jnp.pi * x) + 0 * y, trig=True)
    actual = Chebfun2v([f.approx]).norm()
    expected = jnp.sqrt(2.0)
    assert jnp.abs(actual - expected) < 100 * jnp.finfo(jnp.float64).eps * expected


def test_complex_public_norm_control():
    f = Chebfun2.from_function(lambda x, y: (1 + 2j) * (1 + x + y))
    actual = Chebfun2v([f.approx]).norm()
    expected = jnp.sqrt(100 / 3)
    assert jnp.abs(actual - expected) < 100 * jnp.finfo(jnp.float64).eps * expected


def test_physical_integral_control():
    f = Chebfun2.from_function(lambda x, y: 1 + x + y, domain=(-2.0, 3.0, -1.0, 2.0))
    expected = jnp.asarray(30.0)
    assert (
        jnp.abs(_integral_separable(f.approx) - expected)
        < 100 * jnp.finfo(jnp.float64).eps * expected
    )


def test_trig_integral_jit_control():
    import jax

    f = Chebfun2.from_function(lambda x, y: 1 + jnp.cos(jnp.pi * x) + 0 * y, trig=True)
    assert (
        jnp.abs(jax.jit(_integral_separable)(f.approx) - 4) < 100 * jnp.finfo(jnp.float64).eps * 4
    )


def test_vertcat_identity_domain_control():
    f = Chebfun2.from_function(lambda x, y: x, domain=(-2.0, 3.0, -1.0, 2.0))
    g = Chebfun2.from_function(lambda x, y: y, domain=f.domain)
    pair = f.vertcat(g)
    field = pair.vertcat(2j)
    assert field.components[0] is f.approx
    assert field.components[1] is g.approx
    assert field.domain == f.domain
    assert (
        jnp.abs(Chebfun2(approx=field.components[2])(0.0, 0.0) - 2j)
        < 100 * jnp.finfo(jnp.float64).eps
    )


def test_empty_norm_source_contract():
    # @chebfun2v/norm.m (7574c77) returns [] before inspecting components.
    result = Chebfun2v.empty().norm()
    assert result.shape == (0,)
    assert result.dtype == jnp.float64


def test_mixed_zero_component_norm_control():
    f = Chebfun2.from_function(lambda x, y: x + 0 * y)
    zero = Chebfun2.from_function(lambda x, y: 0 * x + 0 * y)
    field = Chebfun2v([f.approx, zero.approx])
    expected = jnp.sqrt(4 / 3)
    assert jnp.abs(field.norm() - expected) < 100 * jnp.finfo(jnp.float64).eps * expected
