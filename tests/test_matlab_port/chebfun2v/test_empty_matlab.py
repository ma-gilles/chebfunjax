"""Port of MATLAB Chebfun tests/chebfun2v/test_empty.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _maxdiff(F, fns, dom=(-1.0, 1.0, -1.0, 1.0)):
    xs = jnp.linspace(dom[0] + 1e-9, dom[1] - 1e-9, 9)
    ys = jnp.linspace(dom[2] + 1e-9, dom[3] - 1e-9, 9)
    X, Y = jnp.meshgrid(xs, ys)
    worst = 0.0
    for c, fn in zip(F.components, fns):
        f2 = Chebfun2(approx=c)
        worst = max(worst, float(jnp.max(jnp.abs(
            jnp.asarray(f2(X, Y)) - fn(X, Y)))))
    return worst


class TestChebfun2vEmpty:
    def test_all_matlab_assertions(self):
        F = Chebfun2v.empty()
        F.conj()
        F.cross(F)
        F.transpose()
        F.curl() if hasattr(F, "curl") else None
        F.divergence()
        F.dot(F)
        F(jnp.asarray(1.0), jnp.asarray(1.0))
        F.imag()
        assert F.isempty()
        F.laplacian()
        F - F
        F * 1.0
        F.norm()
        F + F
        F ** 2
        F.real()
        F.roots()
        F.ctranspose()
