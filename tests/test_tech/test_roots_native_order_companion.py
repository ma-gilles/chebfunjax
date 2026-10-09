"""Restore literal output-order predicates six/seven from native roots tests.

Provenance
----------
MATLAB source : tests/chebtech/test_roots.m, pass(n,6), pass(n,7)
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_original_six_output_order(Tech):
    f = Tech.from_function(lambda x: 1 + 25*x**2)
    roots = f.roots(complex_roots=True)
    assert roots.shape == (2,)
    assert float(jnp.max(jnp.abs(roots-jnp.asarray([1j, -1j])/5))) < 10*EPS


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_original_seven_output_order(Tech):
    f = Tech.from_function(lambda x: (1 + 25*x**2)*jnp.exp(x))
    roots = f.roots(complex_roots=True, prune=True)
    assert roots.shape == (2,)
    assert float(jnp.max(jnp.abs(roots-jnp.asarray([1j, -1j])/5))) < 10*f.n*EPS
