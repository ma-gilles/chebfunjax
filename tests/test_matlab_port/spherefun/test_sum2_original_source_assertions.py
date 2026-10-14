# uses-numpy: Independent analytic reference values and numerical assertions.
"""The two literal assertions in MATLAB tests/spherefun/test_sum2.m.

Provenance
----------
MATLAB source : tests/spherefun/test_sum2.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


def _cartesian(op):
    return lambda lam, theta: op(
        jnp.cos(lam)*jnp.sin(theta),
        jnp.sin(lam)*jnp.sin(theta),
        jnp.cos(theta),
    )


def test_source_sum2_assertion_one():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    g = Spherefun.from_function(_cartesian(lambda x, y, z: 1+x+y+z))
    assert abs(float(g.sum2()) - 4*jnp.pi) < tol


def test_source_sum2_assertion_two():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps

    def f(x, y, z):
        return (0.75*jnp.exp(-(9*x-2)**2/4-(9*y-2)**2/4-(9*z-2)**2/4)
                + 0.75*jnp.exp(-(9*x+1)**2/49-(9*y+1)/10-(9*z+1)/10)
                + 0.5*jnp.exp(-(9*x-7)**2/4-(9*y-3)**2/4-(9*z-5)**2/4)
                - 0.2*jnp.exp(-(9*x-4)**2-(9*y-7)**2-(9*z-5)**2))

    g = Spherefun.from_function(_cartesian(f))
    assert abs(float(g.sum2()) - 6.6961822200736179523) < tol
