"""Port of MATLAB Chebfun tests/chebfun2v/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vCross:
    def test_scalar_cross_2d(self):
        # cross((f,g),(h,k)) = f*k - g*h
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        G = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        c = F.cross(G)
        exact = float(X0) ** 2 + float(Y0) ** 2
        assert abs(float(c(X0, Y0)) - exact) < 1e-10


@pytest.mark.parametrize("count", [2, 3])
def test_original_definition(count):
    fns = [lambda x, y: jnp.cos(x), lambda x, y: jnp.sin(y), lambda x, y: x*y]
    gns = [lambda x, y: x, lambda x, y: y, lambda x, y: x+y]
    field = Chebfun2v.from_functions(*fns[:count])
    other = Chebfun2v.from_functions(*gns[:count])
    f = [Chebfun2(approx=c) for c in field.components]
    g = [Chebfun2(approx=c) for c in other.components]
    if count == 2:
        expected = f[0]*g[1]-f[1]*g[0]
        actual = Chebfun2(approx=field.cross(other))
    else:
        expected = Chebfun2v([(f[1]*g[2]-f[2]*g[1]).approx,
                             (f[2]*g[0]-f[0]*g[2]).approx,
                             (f[0]*g[1]-f[1]*g[0]).approx])
        actual = field.cross(other)
    assert float((actual-expected).norm()) < 10*ChebfunPref().cheb2Prefs.chebfun2eps
