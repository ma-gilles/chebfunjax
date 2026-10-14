"""All three original assertions in MATLAB tests/spherefun/test_fevalm.m.

Provenance
----------
MATLAB source: tests/spherefun/test_fevalm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The second source case intentionally constructs a Chebfun2, not a Spherefun.
"""

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform


def test_three_original_fevalm_assertions():
    tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
    state = init_matlab_twister(2016)

    def points():
        nonlocal state
        s, state = matlab_rand_uniform(state, (5, 1))
        t, state = matlab_rand_uniform(state, (5, 1))
        return jnp.pi*(2*s-1), jnp.pi/2*t

    s, t = points()
    assert Spherefun.empty().fevalm(s, t).size == 0  # pass(1)

    f = chebfun2(lambda lam, th: jnp.cos(lam)*jnp.sin(th))
    s, t = points()
    ss, tt = jnp.meshgrid(s.reshape(-1), t.reshape(-1))
    assert float(jnp.linalg.norm(f(ss, tt)-f.fevalm(s, t), ord=2)) < tol  # pass(2)

    f = Spherefun.from_function(lambda lam, th: jnp.exp(-(jnp.cos(th)-1)**2))
    s, t = points()
    ss, tt = jnp.meshgrid(s.reshape(-1), t.reshape(-1))
    assert float(jnp.linalg.norm(f(ss, tt)-f.fevalm(s, t), ord=2)) < tol  # pass(3)
