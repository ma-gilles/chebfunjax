"""Original MATLAB function-norm assertions at source bounds.

Provenance
----------
MATLAB source: tests/chebfun2v/test_divergence.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

EPS = ChebfunPref().cheb2Prefs.chebfun2eps


def test_original_two_and_three_component_definitions():
    functions = [lambda x, y: jnp.cos(x), lambda x, y: jnp.sin(y),
                 lambda x, y: jnp.cos(x)]
    for count in [2, 3]:
        field = Chebfun2v.from_functions(*functions[:count])
        f, g = [Chebfun2(approx=c) for c in field.components[:2]]
        expected = f.diff(dim=2)+g.diff(dim=1)
        assert float((expected-Chebfun2(approx=field.divergence())).norm()) < 100*EPS


def test_original_divergence_of_gradient():
    f = Chebfun2.from_function(lambda x, y: jnp.cos(x*y))
    actual = Chebfun2(approx=f.gradient().divergence())
    assert float((f.laplacian()-actual).norm()) < 100*EPS


# Retain additional polynomial regression controls.
X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vDivergence:
    def test_divergence_of_polynomial_field(self):
        # div(x^2, xy) = 2x + x = 3x
        F = Chebfun2v.from_functions(lambda x, y: x * x,
                                     lambda x, y: x * y)
        d = F.div()
        assert abs(float(d(X0, Y0)) - 3 * float(X0)) < 1e-9

    def test_rotation_divergence_free(self):
        F = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        assert abs(float(F.div()(X0, Y0))) < 1e-10
