"""Literal continuous-norm predicates from MATLAB test_cumsum.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax import cheb
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def _xyz(domain=None):
    if domain is None:
        return cheb.xyz()
    return (
        chebfun3(lambda x, y, z: x, domain),
        chebfun3(lambda x, y, z: y, domain),
        chebfun3(lambda x, y, z: z, domain),
    )


def test_original_cumsum_cube_12_predicates():
    tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
    x, y, z = _xyz()
    assert (x.cumsum() - 0.5 * (x**2 - 1)).norm() < tol
    assert (x.cumsum(1) - 0.5 * (x**2 - 1)).norm() < tol
    assert (x.cumsum(2) - x * (y + 1)).norm() < tol
    assert (x.cumsum(3) - x * (z + 1)).norm() < tol
    assert (y.cumsum() - y * (x + 1)).norm() < tol
    assert (y.cumsum(1) - y * (x + 1)).norm() < tol
    assert (y.cumsum(2) - 0.5 * (y**2 - 1)).norm() < tol
    assert (y.cumsum(3) - y * (z + 1)).norm() < tol
    assert (z.cumsum() - z * (x + 1)).norm() < tol
    assert (z.cumsum(1) - z * (x + 1)).norm() < tol
    assert (z.cumsum(2) - z * (y + 1)).norm() < tol
    assert (z.cumsum(3) - 0.5 * (z**2 - 1)).norm() < tol


def test_original_cumsum_box_12_predicates():
    tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
    domain = (-1.1, 2, -0.2, 3, 5, 6)
    x, y, z = _xyz(domain)
    assert (x.cumsum() - 0.5 * (x**2 - (-1.1) ** 2)).norm() < tol
    assert (x.cumsum(1) - 0.5 * (x**2 - (-1.1) ** 2)).norm() < tol
    assert (x.cumsum(2) - x * (y + 0.2)).norm() < tol
    assert (x.cumsum(3) - x * (z - 5)).norm() < tol
    assert (y.cumsum() - y * (x + 1.1)).norm() < tol
    assert (y.cumsum(1) - y * (x + 1.1)).norm() < tol
    assert (y.cumsum(2) - 0.5 * (y**2 - (-0.2) ** 2)).norm() < tol
    assert (y.cumsum(3) - y * (z - 5)).norm() < tol
    assert (z.cumsum() - z * (x + 1.1)).norm() < tol
    assert (z.cumsum(1) - z * (x + 1.1)).norm() < tol
    assert (z.cumsum(2) - z * (y + 0.2)).norm() < tol
    assert (z.cumsum(3) - 0.5 * (z**2 - 5**2)).norm() < tol


def test_original_cumsum_triple_2_predicates():
    tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
    x, y, z = _xyz((-1.1, 2, -0.2, 3, 5, 6))
    f = ((x - 0.1) * (y + 0.1) * (z + 0.1)).sin()
    assert (f.cumsum().cumsum(2).cumsum(3) - f.cumsum3()).norm() < tol
    assert (f.cumsum(1).cumsum(2).cumsum(3) - f.cumsum3()).norm() < tol


def test_original_cumsum_1d_slice_predicate():
    tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun(lambda x: jnp.exp(x))
    f3 = chebfun3(lambda x, y, z: jnp.exp(x))
    g = f.cumsum()
    g3 = f3.cumsum()
    g3x = g3(slice(None), 0, 0.3)
    assert (g3x - g).norm() < tol
