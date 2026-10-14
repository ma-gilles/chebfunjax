"""Port of MATLAB Chebfun tests/chebfun3/test_laplacian.m (Fable 5).

Both original continuous-norm predicates and source bounds are retained.

Provenance
----------
MATLAB source : tests/chebfun3/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 50 * ChebfunPref().cheb3Prefs.chebfun3eps


@pytest.mark.parametrize("case", [0, 1])
def test_original_laplacian(case):
    if case == 0:
        F = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*z)+jnp.sin(y*z))
    else:
        F = Chebfun3.from_function(lambda x, y, z: jnp.cos(x)+y*z+z**2)
    lapF = F.laplacian()
    lapF1 = F.diff(dim=1, k=2) + F.diff(dim=2, k=2) + F.diff(dim=3, k=2)
    assert (lapF1-lapF).norm() < TOL
