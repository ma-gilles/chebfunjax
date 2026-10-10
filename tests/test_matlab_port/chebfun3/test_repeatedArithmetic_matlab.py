"""Original MATLAB Chebfun tests/chebfun3/test_repeatedArithmetic.m assertions.

Provenance
----------
MATLAB source : tests/chebfun3/test_repeatedArithmetic.m
Chebfun commit: 7574c77
Original functions, iteration counts, domains, norm predicates and bounds retained.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e4 * ChebfunPref().cheb3Prefs.chebfun3eps

@pytest.mark.parametrize('statement', range(1, 9))
def test_native_repeated_arithmetic_all_eight(statement):
    fn = jnp.sin if statement in (7, 8) else jnp.cos
    f = chebfun3(lambda x, y, z: fn(x*y*z))
    if statement == 1:
        g = chebfun3(0)
        for _ in range(50):
            g = g+f
        assert (g-50*f).norm() < 10*TOL
        return
    g = f
    count = 20 if statement in (7, 8) else 10
    for _ in range(count):
        if statement in (2, 7):
            g = g*f
        else:
            previous = g  # MATLAB closure captures the preceding g.
            op = lambda x, y, z, previous=previous: previous(x, y, z)*f(x, y, z)
            g = chebfun3(op, fiberDim=statement-3) if statement in (4, 5, 6) else chebfun3(op)
    assert (g-f**(count+1)).norm() < TOL
