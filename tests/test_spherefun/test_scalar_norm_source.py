"""Independent controls derived from the scalar norm source dispatch.

Provenance
----------
MATLAB source : @spherefun/norm.m
Chebfun commit: 7574c77
The 256-eps analytic bound is new, not a replacement for source test bounds.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import Spherefun


@pytest.mark.parametrize("p", [2, jnp.inf, "unknown", 1, -jnp.inf])
def test_empty_precedes_order_validation(p):
    result = Spherefun.empty().norm(p)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (0,)


@pytest.mark.parametrize("p", [4, -2, 0])
def test_even_order_literal_constant_integral(p):
    f = Spherefun.from_function(lambda lam, th: 2 * jnp.ones_like(lam))
    result = f.norm(p)
    if p == 0:
        assert bool(jnp.isposinf(result))
    else:
        expected = (4 * jnp.pi * 2.0 ** p) ** (1.0 / p)
        assert abs(result - expected) <= 256 * jnp.finfo(jnp.float64).eps * abs(expected)


@pytest.mark.parametrize("p", [1, 3, 1.5, -jnp.inf, "-inf", "min", "FRO", 2j, True])
def test_unsupported_orders_report_source_identifier(p):
    f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam))
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:norm:"):
        f.norm(p)


def test_exact_aliases_and_jax_l2_reduction(monkeypatch):
    f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam))
    monkeypatch.setattr(Spherefun, "svd", lambda self: jnp.asarray([3.0, 4.0]))
    monkeypatch.setattr(Spherefun, "minandmax2", lambda self: (jnp.asarray([-7., 2.]), None))
    assert float(f.norm()) == float(f.norm("fro")) == 5.0
    assert float(f.norm(jnp.inf)) == float(f.norm("inf")) == float(f.norm("max")) == 7.0
