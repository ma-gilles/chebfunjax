"""Retained legacy callable-Neumann adapter coverage, separate from source BC.

These sampled checks retain the former Python port's existing loose bounds;
they do not implement the source ten-clause test or establish source accuracy.

Provenance
----------
MATLAB source : tests/chebop/test_bc.m (first equation; callable adapter)
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-6


def test_legacy_callable_neumann_adapter_sampled_bounds():
    A = Chebop(lambda x, u: u.diff(2) + 4*u.diff() + u,
               domain=(-3.0, 4.0))
    A.lbc = -1.0
    A.rbc = lambda u: u.diff()
    f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(-3.0, 4.0))
    u = A.solve(f)
    xs = jnp.asarray(np.linspace(-2.8, 3.8, 30))
    res = u.diff(2)(xs) + 4*u.diff()(xs) + u(xs) - f(xs)
    assert float(jnp.max(jnp.abs(res))) < 1e3*TOL
    assert abs(float(u(jnp.asarray(-3.0))) + 1) < TOL
    assert abs(float(u.diff()(jnp.asarray(4.0)))) < 1e2*TOL
