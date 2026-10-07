"""Eight source clauses, with deterministic queries replacing unseeded rand.

Provenance
----------
MATLAB source : tests/spherefun/test_plus.m, @spherefun/sphf2cartf.m
Chebfun commit: 7574c77
No source test_minus.m exists; subtraction occurs in clauses 3, 5 and 6.
Numeric-inf requires the separately qualified scalar-norm candidate.
"""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


def cart(f):
    return lambda lam, th: f(jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th), jnp.cos(th))


def test_source_eight_plus_clauses():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    f1 = cart(lambda x, y, z: jnp.sin(jnp.pi * x * y))
    f2 = cart(lambda x, y, z: jnp.sin(jnp.pi * x * z))
    g = Spherefun.from_function(f1) + Spherefun.from_function(f2)
    # Source evaluates (theta,lambda), despite these variable names.
    lam, theta = 0.371, 0.683
    assert abs(g(theta, lam) - (f1(theta, lam) + f2(theta, lam))) < tol  # 1

    def f1(lam, th):
        return jnp.exp(jnp.cos(lam - 1) * jnp.sin(th) * jnp.cos(th))

    def f2(lam, th):
        return jnp.exp(jnp.sin(lam - 0.35) * jnp.sin(th) * jnp.cos(th))

    g = Spherefun.from_function(f1) + Spherefun.from_function(f2)
    assert abs(g(theta, lam) - (f1(theta, lam) + f2(theta, lam))) < tol  # 2
    f = Spherefun.from_function(cart(lambda x, y, z: x * x + y * y + z * z))
    rank = f.rank
    g = f
    for _ in range(10):
        g = g + f
    assert (g - 11 * f).norm(jnp.inf) < g.vscale() * tol  # 3
    assert g.rank - rank == 0  # 4
    f = Spherefun.from_function(cart(lambda x, y, z: jnp.sin(x * y * z)))
    g = 2 * f
    assert (g - f - f).norm(jnp.inf) < tol  # 5
    f = Spherefun.from_function(cart(lambda x, y, z: z))
    g = Spherefun.from_function(cart(lambda x, y, z: z**3))
    assert not (f - g).nonzero_poles  # 6
    f = Spherefun.from_function(cart(lambda x, y, z: z * (1 - z * z)))
    g = Spherefun.from_function(cart(lambda x, y, z: z))
    assert (f + g).nonzero_poles  # 7
    assert (g + f).nonzero_poles  # 8
