"""Literal predicates from MATLAB tests/chebfun3/test_divide.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_divide.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

EPS = ChebfunPref().cheb3Prefs.chebfun3eps


@pytest.mark.parametrize("clause", [1, 2, 3, 4])
def test_source_divide(clause):
    f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z))
    g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z)/2)
    result = {
        1: lambda: f.rdivide(2),
        2: lambda: f.mrdivide(2),
        3: lambda: Chebfun3.ldivide(2, f),
        4: lambda: Chebfun3.mldivide(2, f),
    }[clause]()
    assert (result - g).norm() < 1000 * EPS
