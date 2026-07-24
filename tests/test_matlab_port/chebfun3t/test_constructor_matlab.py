"""Port of MATLAB Chebfun tests/chebfun3t/test_constructor.m (Fable 5).

Constructing a :class:`Chebfun3T` from a function handle and checking it
reproduces the function -- the same math for the Tucker-backed chebfunjax
wrapper and MATLAB's full-tensor variant, so exercised directly against
:func:`chebfunjax.chebfun3d.chebfun3t`.

Two MATLAB assertion groups are omitted as chebfun3t-format-specific:
  * string construction (``chebfun3t('cos(x+z)+sin(x.*y.*z)')``):
    chebfunjax's constructor takes a Python callable, not a MATLAB
    expression string, so there is no string parser to exercise;
  * ``size(f.coeffs, k) < 50`` length checks: :class:`Chebfun3T` stores
    a Tucker decomposition (cols/rows/tubes + core), not a full 3D
    coefficient tensor, so those full-tensor length assertions do not
    apply (the low-rank degrees of freedom are pinned by the Tucker
    chebfun3 ports instead).

Tolerance follows the repo convention (K * machine-EPS, scaled to the
achieved Tucker construction accuracy with a wide margin) in place of
MATLAB's unavailable ``chebfun3eps``.

Provenance
----------
MATLAB source : tests/chebfun3t/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e4 * EPS


def _maxdiff(f, ref, dom, n=25):
    xs = np.linspace(dom[0], dom[1], n)
    ys = np.linspace(dom[2], dom[3], n)
    zs = np.linspace(dom[4], dom[5], n)
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    return float(np.max(np.abs(np.asarray(f(xx, yy, zz))
                               - np.asarray(ref(xx, yy, zz)))))


@pytest.mark.slow
class TestChebfun3tConstructor:
    def test_all_matlab_assertions(self):
        # ff = @(x,y,z) cos(x+z) + sin(x.*y.*z);
        def ff(x, y, z):
            return jnp.cos(x + z) + jnp.sin(x * y * z)

        d = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        f1 = chebfun3t(ff)                 # default domain
        f2 = chebfun3t(ff, d)             # explicit domain
        assert _maxdiff(f1, ff, d) < TOL
        assert _maxdiff(f2, ff, d) < TOL
        assert _maxdiff(f1, f2, d) < TOL   # f1 == f2

        # Accuracy of a second function at a fixed point:
        #   g = @(x,y,z) cos(x).*y.*z + x.*z.*sin(y);
        def g(x, y, z):
            return jnp.cos(x) * y * z + x * z * jnp.sin(y)

        fg = chebfun3t(g)
        px, py, pz = 0.1, np.pi / 6, -0.3
        exact = float(np.cos(px) * py * pz + px * pz * np.sin(py))
        assert abs(float(fg(px, py, pz)) - exact) < TOL

        # A rational function on a wider domain [-2,2]^3:
        #   f = @(x,y,z) 1./(1+x.^2.*y.^2.*z.^2);
        def rat(x, y, z):
            return 1.0 / (1 + x ** 2 * y ** 2 * z ** 2)

        d2 = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0)
        fr = chebfun3t(rat, d2)
        assert _maxdiff(fr, rat, d2, n=25) < TOL
