# uses-numpy: Independent analytic reference values and numerical assertions.
"""Actual two assertions in MATLAB tests/spherefun/test_sum2.m, 7574c77.

The factory @chebfunpref/chebfunpref.m sets cheb2Prefs.chebfun2eps=eps;
retain source tol=1000*eps, source operators and source reference literal.
The Cartesian callback is expressed on the public Python angular API.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1000 * np.finfo(np.float64).eps


def cartesian(op):
    return lambda lam, theta: op(jnp.cos(lam)*jnp.sin(theta),
                                 jnp.sin(lam)*jnp.sin(theta), jnp.cos(theta))


def test_source_sum2_assertion_one():
    g = Spherefun.from_function(cartesian(lambda x, y, z: 1+x+y+z))
    assert abs(float(g.sum2())-4*np.pi) < TOL


def test_source_sum2_assertion_two():
    def op(x, y, z):
        return (0.75*jnp.exp(-(9*x-2)**2/4-(9*y-2)**2/4-(9*z-2)**2/4)
                +0.75*jnp.exp(-(9*x+1)**2/49-(9*y+1)/10-(9*z+1)/10)
                +0.5*jnp.exp(-(9*x-7)**2/4-(9*y-3)**2/4-(9*z-5)**2/4)
                -0.2*jnp.exp(-(9*x-4)**2-(9*y-7)**2-(9*z-5)**2))
    g = Spherefun.from_function(cartesian(op))
    assert abs(float(g.sum2())-6.6961822200736179523) < TOL
