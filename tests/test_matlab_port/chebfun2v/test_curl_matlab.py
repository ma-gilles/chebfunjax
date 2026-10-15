"""Original MATLAB function-norm assertions at source bounds.

Provenance
----------
MATLAB source: tests/chebfun2v/test_curl.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

EPS = ChebfunPref().cheb2Prefs.chebfun2eps


def test_original_two_component_definition():
    field = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                    lambda x, y: jnp.sin(y))
    f, g = [Chebfun2(approx=c) for c in field.components]
    expected = g.diff(dim=2)-f.diff(dim=1)
    assert float((expected-Chebfun2(approx=field.curl())).norm()) < 50*EPS


def test_original_three_component_definition():
    field = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                    lambda x, y: jnp.sin(y), lambda x, y: x*y)
    f, g, h = [Chebfun2(approx=c) for c in field.components]
    expected = Chebfun2v([h.diff(dim=1).approx, (-h.diff(dim=2)).approx,
                         (g.diff(dim=2)-f.diff(dim=1)).approx])
    assert float((expected-field.curl()).norm()) < 50*EPS


def test_original_curl_of_gradient():
    f = Chebfun2.from_function(lambda x, y: jnp.cos(x+y**2)+jnp.sin(y)+y)
    assert float(Chebfun2(approx=f.gradient().curl()).norm()) < 50*EPS


# Retain additional polynomial regression controls.
X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vCurl:
    def test_scalar_curl(self):
        # curl(x^2, xy) = y; curl(-y, x) = 2
        F = Chebfun2v.from_functions(lambda x, y: x * x,
                                     lambda x, y: x * y)
        assert abs(float(F.curl()(X0, Y0)) - float(Y0)) < 1e-9
        G = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        assert abs(float(G.curl()(X0, Y0)) - 2.0) < 1e-10

    def test_curl_of_gradient_is_zero(self):
        F = Chebfun2v.from_functions(lambda x, y: y, lambda x, y: x)
        assert abs(float(F.curl()(X0, Y0))) < 1e-10
