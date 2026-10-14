"""Literal predicates from MATLAB Chebfun tests/chebfun3/test_diff.m.

Original source constructs each expected derivative as a Chebfun3 and uses
its default continuous L2 norm. MATLAB argument order ``diff(f,k,dim)`` is
mapped to Python's explicit ``f.diff(dim=dim,k=k)``; one-argument order
forms use ``f.diff(k=k)``.

Provenance
----------
MATLAB source : tests/chebfun3/test_diff.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e6 * ChebfunPref().cheb3Prefs.chebfun3eps

# Literal MATLAB ff, ffx, ffy, ffz lists and domains.
FF = [
    lambda x, y, z: x,
    lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z),
    lambda x, y, z: jnp.cos(x * y * z),
    lambda x, y, z: x**2 + x * y**2 * z**3,
]
FFX = [
    lambda x, y, z: 1 + 0 * x,
    lambda x, y, z: -jnp.sin(x) * jnp.exp(y) * jnp.sin(z),
    lambda x, y, z: -y * z * jnp.sin(x * y * z),
    lambda x, y, z: 2 * x + y**2 * z**3,
]
FFY = [
    lambda x, y, z: 0 * x,
    lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z),
    lambda x, y, z: -x * z * jnp.sin(x * y * z),
    lambda x, y, z: 2 * x * y * z**3,
]
FFZ = [
    lambda x, y, z: 0 * x,
    lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.cos(z),
    lambda x, y, z: -x * y * jnp.sin(x * y * z),
    lambda x, y, z: 3 * x * y**2 * z**2,
]
DOMS = [
    (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    (-3.0, 1.0, -1.0, 2.0, 3.0, 5.0),
]


@pytest.mark.parametrize("jj", range(4), ids=[f"fn{j}" for j in range(4)])
@pytest.mark.parametrize("r", range(2), ids=[f"dom{r}" for r in range(2)])
def test_original_first_derivatives(jj, r):
    dom = DOMS[r]
    f = chebfun3(FF[jj], domain=dom)
    fx = chebfun3(FFX[jj], domain=dom)
    fy = chebfun3(FFY[jj], domain=dom)
    fz = chebfun3(FFZ[jj], domain=dom)
    # MATLAB norm(diff(f,1,dim) - exactChebfun3) < tol, dim=1,2,3.
    assert (f.diff(dim=1, k=1) - fx).norm() < TOL
    assert (f.diff(dim=2, k=1) - fy).norm() < TOL
    assert (f.diff(dim=3, k=1) - fz).norm() < TOL


def test_original_diff_syntax_predicates():
    # MATLAB's post-loop r=2, jj=3: f = cos(x*y*z) on the second domain.
    dom = DOMS[1]
    f = chebfun3(FF[2], domain=dom)
    # Source pass 25: norm(diff(f) - diff(f,1,1)) < tol.
    assert (f.diff() - f.diff(dim=1, k=1)).norm() < TOL
    # Source pass 26: norm(diff(f,1) - diff(f,1,1)) < tol.
    assert (f.diff(k=1) - f.diff(dim=1, k=1)).norm() < TOL
    # Source pass 27: norm(diff(f,2) - diff(f,2,1)) < 100*tol.
    assert (f.diff(k=2) - f.diff(dim=1, k=2)).norm() < 100 * TOL
    # Source pass 28: norm(diff(f,2,2) - diff(diff(f,1,2),1,2)) < 10*tol.
    assert (f.diff(dim=2, k=2)
            - f.diff(dim=2, k=1).diff(dim=2, k=1)).norm() < 10 * TOL
