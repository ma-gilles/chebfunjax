"""Literal predicates from MATLAB tests/chebfun3/test_divide.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_divide.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS


@pytest.mark.parametrize("clause", [1, 2, 3, 4])
def test_source_divide(clause):
    # Python maps scalar right/left and elementwise/matrix division to /.
    f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z))
    g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z)/2)
    assert (f / 2 - g).norm() < 1000 * EPS
