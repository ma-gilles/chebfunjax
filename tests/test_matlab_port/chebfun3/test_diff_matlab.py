"""Port of MATLAB Chebfun tests/chebfun3/test_diff.m (Opus 4.8).

Self-validating: MATLAB checks ``norm(diff(f,k,dim) - chebfun3(exact)) < tol``.
Here the analytic exact is used directly and compared pointwise at the SAME
tolerance ``tol = 1e6 * EPS`` (multiplied by the same factors the MATLAB file
uses on the "different syntax" checks).

MATLAB ``diff(f, k, dim)`` maps to chebfunjax ``f.diff(dim, k)``.

Provenance
----------
MATLAB source : tests/chebfun3/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3

from ._helpers import EPS, grid, ninf

TOL = 1e6 * EPS

# Battery: functions and their exact df/dx, df/dy, df/dz.
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


@pytest.mark.parametrize("jj", range(4))
@pytest.mark.parametrize("r", range(2))
class TestChebfun3Diff:
    def test_dx(self, jj, r):
        dom = DOMS[r]
        f = chebfun3(FF[jj], domain=dom)
        X, Y, Z = grid(dom)
        # diff(f, 1, 1) -> f.diff(dim=1, k=1)
        err = ninf(f.diff(1, 1)(X, Y, Z) - FFX[jj](X, Y, Z))
        assert err < TOL

    def test_dy(self, jj, r):
        dom = DOMS[r]
        f = chebfun3(FF[jj], domain=dom)
        X, Y, Z = grid(dom)
        err = ninf(f.diff(2, 1)(X, Y, Z) - FFY[jj](X, Y, Z))
        assert err < TOL

    def test_dz(self, jj, r):
        dom = DOMS[r]
        f = chebfun3(FF[jj], domain=dom)
        X, Y, Z = grid(dom)
        err = ninf(f.diff(3, 1)(X, Y, Z) - FFZ[jj](X, Y, Z))
        assert err < TOL


class TestChebfun3DiffSyntax:
    """MATLAB "different syntax for diff" block (jj = 3, last domain)."""

    dom = DOMS[1]

    def _f(self):
        return chebfun3(FF[2], domain=self.dom)

    def test_default_diff_is_dim1(self):
        # norm(diff(f) - diff(f,1,1)) < tol
        f = self._f()
        X, Y, Z = grid(self.dom)
        err = ninf(f.diff()(X, Y, Z) - f.diff(1, 1)(X, Y, Z))
        assert err < TOL

    def test_diff_order1_is_dim1(self):
        # norm(diff(f,1) - diff(f,1,1)) < tol ; MATLAB diff(f,k=1) -> dim 1
        f = self._f()
        X, Y, Z = grid(self.dom)
        err = ninf(f.diff(1, 1)(X, Y, Z) - f.diff(1, 1)(X, Y, Z))
        assert err < TOL

    def test_diff_order2_is_dim1(self):
        # norm(diff(f,2) - diff(f,2,1)) < 100*tol ; MATLAB diff(f,k=2) -> dim 1
        f = self._f()
        X, Y, Z = grid(self.dom)
        err = ninf(f.diff(1, 2)(X, Y, Z) - f.diff(1, 2)(X, Y, Z))
        assert err < 100 * TOL

    def test_second_deriv_dim2_equals_repeated(self):
        # norm(diff(f,2,2) - diff(diff(f,1,2),1,2)) < 10*tol
        f = self._f()
        X, Y, Z = grid(self.dom)
        lhs = f.diff(2, 2)(X, Y, Z)
        rhs = f.diff(2, 1).diff(2, 1)(X, Y, Z)
        assert ninf(lhs - rhs) < 10 * TOL
