"""Exact smoothness and unary demotion, @singfun source at7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


def test_small_nonzero_exponent_remains_singular():
    f = Singfun(Chebtech2.from_coeffs(jnp.array([1.])), (1e-14, 0.))
    assert not f.issmooth
    for name in ("real", "conj"):
        out = getattr(f, name)()
        assert isinstance(out, Singfun)
        assert out.exponents == f.exponents


def test_zero_factor_is_smooth_with_nonzero_exponents():
    f = Singfun(Chebtech2.from_coeffs(jnp.array([0.])), (-0.5, 0.25))
    assert f.issmooth
    assert not Singfun(Chebtech2.from_coeffs(jnp.array([jnp.nan])), (-0.5, 0.)).issmooth
    assert Singfun(Chebtech2.from_coeffs(jnp.array([jnp.nan])), (0., 0.)).issmooth


@pytest.mark.parametrize("name,value", [("real", 2j), ("imag", 2.), ("conj", 0.)])
def test_unary_demotes_after_transform(name, value):
    f = Singfun(Chebtech2.from_coeffs(jnp.array([value])), (-0.5, 0.25))
    out = getattr(f, name)()
    assert isinstance(out, Chebtech2)
    assert bool(out.iszero())
    assert f.exponents == (-0.5, 0.25)


@pytest.mark.parametrize("name", ["real", "imag", "conj"])
def test_unary_empty_preserves_explicit_exponents(name):
    f = Singfun.constructor(jnp.empty((0,)))
    assert getattr(f, name)() is f
    assert getattr(f, name)().exponents == (0., 0.)
