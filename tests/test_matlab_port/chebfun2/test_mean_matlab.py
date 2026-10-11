"""All four literal MATLAB Chebfun2 mean predicates.

MATLAB source: tests/chebfun2/test_mean.m; @separableApprox/mean.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Factory pref.cheb2Prefs.chebfun2eps is eps (@chebfunpref/chebfunpref.m:735).
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = float(jnp.finfo(jnp.float64).eps)


def _source_function(domain=(-1.0, 1.0, -1.0, 1.0)):
    return Chebfun2.from_function(lambda x, y: jnp.sin((x - .1) * y), domain=domain)


class TestChebfun2Mean:
    def test_native_mean2_default_rectangle(self):
        f = _source_function()
        assert abs(float(f.mean2()) - float(f.sum2()) / 4) < TOL

    def test_native_mean2_general_rectangle(self):
        f = _source_function((-2.0, 3.0, -1.0, 4.0))
        assert abs(float(f.mean2()) - float(f.sum2()) / 25) < TOL

    def test_native_mean_dim1(self):
        f = _source_function((-2.0, 3.0, -1.0, 4.0))
        assert float((f.mean(1) - f.sum(1) / 5).norm()) < TOL

    def test_native_mean_dim2(self):
        f = _source_function((-2.0, 3.0, -1.0, 4.0))
        assert float((f.mean(2) - f.sum(2) / 5).norm()) < TOL
