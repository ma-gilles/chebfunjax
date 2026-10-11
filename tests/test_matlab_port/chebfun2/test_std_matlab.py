"""All three literal MATLAB Chebfun2 std predicates.

MATLAB source: tests/chebfun2/test_std.m; @separableApprox/std.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Factory pref.cheb2Prefs.chebfun2eps is eps (@chebfunpref/chebfunpref.m:735).
Native default norm is the public scalar Chebfun L2 norm, not a sample maximum.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1e3 * float(jnp.finfo(jnp.float64).eps)


def _exact():
    return chebfun(lambda x: jnp.sqrt(4.0 / 45.0))


class TestChebfun2Std:
    def test_default_dim(self):
        f1 = Chebfun2.from_function(lambda x, y: y**2 + x)
        assert float((f1.std() - _exact()).norm()) < TOL

    def test_dim1(self):
        f1 = Chebfun2.from_function(lambda x, y: y**2 + x)
        assert float((f1.std(None, 1) - _exact()).norm()) < TOL

    def test_dim2(self):
        f2 = Chebfun2.from_function(lambda x, y: x**2 + y)
        assert float((f2.std(None, 2) - _exact()).norm()) < TOL
