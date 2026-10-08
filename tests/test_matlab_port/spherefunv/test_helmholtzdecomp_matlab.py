"""Literal four clauses of the pinned Helmholtz decomposition test.

Provenance
----------
MATLAB source : tests/spherefunv/test_helmholtzdecomp.m
Chebfun commit: 7574c77
Uses the source global vector norm and original preference-derived bound.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


def _sph(function):
    return Spherefun.from_function(
        lambda lam, th: function(
            jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th), jnp.cos(th)
        )
    )


@pytest.mark.parametrize("case", [1, 2, 3])
def test_literal_helmholtz_decomposition(case):
    cases = [
        (
            lambda x, y, z: jnp.cos(x * y * z),
            lambda x, y, z: jnp.sin(x + 0.1 * y + 5 * z**2),
            lambda x, y, z: x * y * z,
        ),
        (
            lambda x, y, z: jnp.sin((x - 0.1) * y * z),
            lambda x, y, z: jnp.cos(x + 0.50 * y - z**2),
            lambda x, y, z: -x * y**2 * z,
        ),
        (
            lambda x, y, z: jnp.sin((x - 0.1) * y * z),
            lambda x, y, z: 1 + 0 * x,
            lambda x, y, z: 0 * x,
        ),
    ]
    f = Spherefunv(*[_sph(fn) for fn in cases[case - 1]]).tangent()
    u, v = f.helmholtzdecomp()
    gradient = u.gradient()
    curl = v.curl()
    residual = f - gradient - curl
    tol = 1e5 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert float(residual.norm()) < tol


def test_literal_empty_helmholtz_decomposition():
    u, v = Spherefunv.empty().helmholtzdecomp()
    assert isinstance(u, Spherefun) and isinstance(v, Spherefun)
    assert u.isempty() and v.isempty()
