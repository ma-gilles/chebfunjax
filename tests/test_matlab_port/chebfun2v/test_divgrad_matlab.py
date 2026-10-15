"""Original MATLAB function-norm assertions at source bounds.

Provenance
----------
MATLAB source: tests/chebfun2v/test_divgrad.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

EPS = ChebfunPref().cheb2Prefs.chebfun2eps


def test_original_definition():
    field = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                    lambda x, y: jnp.sin(y))
    f, g = [Chebfun2(approx=c) for c in field.components]
    expected = f.diff(dim=2, k=2) + g.diff(dim=1, k=2)
    assert float((expected-Chebfun2(approx=field.divgrad())).norm()) < 50*EPS
