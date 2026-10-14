"""The eight source assertions in MATLAB tests/spherefun/test_plus.m.

Provenance
----------
MATLAB source : tests/spherefun/test_plus.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform

from ._cart import sph_xyz


def _standalone_unseeded_test_draws():
    """Documented seed-0 Twister draws for this otherwise unseeded test."""
    state = init_matlab_twister(0)
    values = []
    for _ in range(4):
        value, state = matlab_rand_uniform(state, ())
        values.append(value)
    return values


def test_all_eight_source_plus_assertions():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    # test_plus.m does not seed its RNG. Use a reproducible standalone stream
    # while retaining its four scalar draws and feval(theta, lambda) ordering.
    lambda1, theta1, lambda2, theta2 = _standalone_unseeded_test_draws()

    # pass(1): Cartesian handles; source calls feval(gplus, theta, lambda).
    f1 = sph_xyz(lambda x, y, z: jnp.sin(jnp.pi*x*y))
    f2 = sph_xyz(lambda x, y, z: jnp.sin(jnp.pi*x*z))
    gplus = f1 + f2
    lam, th = theta1, lambda1
    x, y, z = jnp.cos(lam)*jnp.sin(th), jnp.sin(lam)*jnp.sin(th), jnp.cos(th)
    exact = jnp.sin(jnp.pi*x*y) + jnp.sin(jnp.pi*x*z)
    assert abs(float(gplus(lam, th) - exact)) < tol

    # pass(2): another independent pair; source again swaps theta/lambda.
    f1 = Spherefun.from_function(
        lambda l, t: jnp.exp(jnp.cos(l-1)*jnp.sin(t)*jnp.cos(t)))
    f2 = Spherefun.from_function(
        lambda l, t: jnp.exp(jnp.sin(l-0.35)*jnp.sin(t)*jnp.cos(t)))
    gplus = f1 + f2
    lam, th = theta2, lambda2
    exact = (jnp.exp(jnp.cos(lam-1)*jnp.sin(th)*jnp.cos(th))
             + jnp.exp(jnp.sin(lam-0.35)*jnp.sin(th)*jnp.cos(th)))
    assert abs(float(gplus(lam, th) - exact)) < tol

    # pass(3-4): ten repeated additions, source infinity norm and rank.
    f = sph_xyz(lambda x, y, z: x**2 + y**2 + z**2)
    r = int(f.numerical_rank())
    g = f
    for _ in range(10):
        g = g + f
    assert float((g - 11*f).norm(jnp.inf)) < float(g.vscale()) * tol
    assert int(g.numerical_rank()) - r == 0

    # pass(5): cancellation residual in the source infinity norm.
    f = sph_xyz(lambda x, y, z: jnp.sin(x*y*z))
    g = 2*f
    assert float((g - f - f).norm(jnp.inf)) < tol

    # pass(6-8): exact pole metadata for both operand orders.
    f = sph_xyz(lambda x, y, z: z)
    g = sph_xyz(lambda x, y, z: z**3)
    assert not (f - g).nonzero_poles
    f = sph_xyz(lambda x, y, z: z*(1-z**2))
    g = sph_xyz(lambda x, y, z: z)
    assert (f + g).nonzero_poles
    assert (g + f).nonzero_poles
