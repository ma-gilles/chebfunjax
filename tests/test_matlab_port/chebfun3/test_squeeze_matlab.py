"""All seven original squeeze predicates with continuous norms.

Provenance
----------
MATLAB source : tests/chebfun3/test_squeeze.m
Chebfun commit: 7574c77
"""

import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 10*ChebfunPref().cheb3Prefs.chebfun3eps


@pytest.mark.parametrize("case", range(7))
def test_original_squeeze(case):
    if case == 0:
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z))
        assert (f-f.squeeze()).norm() < TOL
    elif case == 1:
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x)).squeeze()
        g = chebfun(jnp.cos)
        assert (f-g).norm() < TOL
    elif case == 2:
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(y),
                                  domain=(-1., 1., -2., 3., -1., 1.)).squeeze()
        g = chebfun(jnp.cos, domain=(-2., 3.))
        assert (f-g).norm() < TOL
    elif case == 3:
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(z),
                                  domain=(-1., 1., -2., 3., -math.pi, math.pi)).squeeze()
        g = chebfun(jnp.sin, domain=(-math.pi, math.pi))
        assert (f-g).norm() < TOL
    else:
        functions = [lambda x, y, z: jnp.cos(y*z),
                     lambda x, y, z: jnp.cos(x*z),
                     lambda x, y, z: jnp.cos(x*y)]
        f = Chebfun3.from_function(functions[case-4]).squeeze()
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x*y))
        assert (f-g).norm() < TOL
