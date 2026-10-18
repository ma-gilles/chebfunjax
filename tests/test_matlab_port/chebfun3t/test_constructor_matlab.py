"""Port of MATLAB Chebfun tests/chebfun3t/test_constructor.m (Fable 5).

Constructing the dense :class:`Chebfun3T` from a function handle and
checking the original source predicates without changing grids or bounds.

The literal native string-construction passes 4–5 and coefficient-length
passes 8–10 are in separate source-focused tests. This file retains native
passes 1–3 and 6–7 with their original grids and tolerance.

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
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * ChebfunPref().cheb3Prefs.chebfun3eps


def _maxdiff(f, ref, dom, n=100):
    xs = np.linspace(dom[0], dom[1], n)
    ys = np.linspace(dom[2], dom[3], n)
    zs = np.linspace(dom[4], dom[5], n)
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    return float(np.max(np.abs(np.asarray(f(xx, yy, zz))
                               - np.asarray(ref(xx, yy, zz)))))


@pytest.mark.slow
class TestChebfun3tConstructor:
    def test_native_predicates_1_2_3_6(self):
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

    def test_native_predicate_7_rational_tensor(self):
        # The real length724 C2R/Rader family retains this original bound.
        # Its coefficient words and crop can differ from native; no fitted
        # cutoff, padding or wider tolerance is used.
        # A rational function on a wider domain [-2,2]^3:
        #   f = @(x,y,z) 1./(1+x.^2.*y.^2.*z.^2);
        def rat(x, y, z):
            return 1.0 / (1 + x ** 2 * y ** 2 * z ** 2)

        d2 = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0)
        fr = chebfun3t(rat, d2)
        assert _maxdiff(fr, rat, d2) < TOL
