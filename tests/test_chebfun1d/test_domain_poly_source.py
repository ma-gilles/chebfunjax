"""Source @domain/poly.m contracts, Chebfun 7574c77."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import poly
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def test_source_forces_second_kind(monkeypatch):
    monkeypatch.setattr(ChebfunPref, "_defaults", ChebfunPref(tech=Chebtech1))
    f = poly(jnp.asarray([-0.2, 0.4]), (0.0, 2.0))
    assert isinstance(f.funs[0].tech, Chebtech2)
    x = jnp.linspace(0.0, 2.0, 17)
    assert jnp.max(jnp.abs(f(x) - (x + 0.2) * (x - 0.4))) < 1e-14


def test_source_infinite_root_index_failure():
    with pytest.raises(IndexError):
        poly(jnp.asarray([1.0, jnp.inf]))


def test_complex_mapped_roots():
    f = poly(jnp.asarray([1.0 + 2j, -0.5j]), (-2.0, 3.0))
    x = jnp.linspace(-2.0, 3.0, 17)
    assert jnp.max(jnp.abs(f(x) - (x - 1 - 2j) * (x + 0.5j))) < 1e-13


def test_empty_and_nan():
    assert poly(jnp.asarray([])).isempty()
    assert jnp.isnan(poly(jnp.asarray([jnp.nan]))(jnp.asarray(0.0)))
