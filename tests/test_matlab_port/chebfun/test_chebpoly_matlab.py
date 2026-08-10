"""Port of MATLAB Chebfun tests/chebfun/test_chebpoly.m (Fable 5).

MATLAB ``chebpoly(f, n)`` is a (deprecated) accessor for the first ``n``
Chebyshev coefficients of ``f`` (returned highest-degree-first via ``rev``).
chebfunjax exposes the ascending coefficients via the ``.coeffs`` property; the
scalar case is already covered by test_chebcoeffs, and array-valued CHEBFUN now
gives an ``(n, m)`` coefficient matrix, so the 1st-kind smooth cases (MATLAB
pass 1 and 2) are ported here directly (FIXED, Fable 5, Big-Three array-valued
epic).

All MATLAB cases are ported: ``Chebfun.chebpoly(n, kind)`` is the
deprecated MATLAB accessor, forwarding to ``chebcoeffs`` and reversing +
transposing the result, so it covers the 2nd-kind and piecewise cases too.

Provenance
----------
MATLAB source : tests/chebfun/test_chebpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)

# First 5 first-kind Chebyshev coefficients of cos(x) (ascending, c_0 first).
# MATLAB's c_exact = [J0(1); 0; -2 J2(1); 0; 2(-23 J0(1) + 40 J1(1))].
_C_EXACT = np.array(
    [jv(0, 1), 0.0, -2 * jv(2, 1), 0.0, 2 * (-23 * jv(0, 1) + 40 * jv(1, 1))]
)


class TestChebfunChebpoly:
    def test_scalar_1st_kind(self):
        # pass(1): first 5 first-kind Chebyshev coeffs of cos(x).
        f = cj.chebfun(np.cos)
        c = np.asarray(f.coeffs)[:5]
        assert float(np.max(np.abs(c - _C_EXACT))) < 1e2 * f.vscale * EPS

    def test_array_valued_1st_kind(self):
        # pass(2): array-valued [cos cos] -> coeffs match per column.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        import jax.numpy as jnp

        g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), jnp.cos(x)], axis=-1))
        c = np.asarray(g.coeffs)[:5]
        assert c.shape == (5, 2)
        assert float(np.max(np.abs(c - _C_EXACT[:, None]))) < 1e2 * g.vscale * EPS

    def test_kind_2(self):
        # pass(3, 4): chebpoly(f, 5, 'kind', 2) on cos(x), scalar and
        # array-valued.  MATLAB's rev() reverses and transposes.
        import jax.numpy as jnp

        exact = np.array([2 * jv(1, 1), 0.0, -6 * jv(3, 1), 0.0,
                          2 * (-235 * jv(1, 1) + 900 * jv(2, 1))])
        f = cj.chebfun(jnp.cos)
        c = np.asarray(f.chebpoly(5, kind=2))
        assert c.shape == (5,)
        assert float(np.max(np.abs(c - exact[::-1]))) < 1e3 * f.vscale * EPS

        g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), jnp.cos(x)],
                                           axis=-1))
        c = np.asarray(g.chebpoly(5, kind=2))
        assert c.shape == (2, 5)
        assert float(np.max(np.abs(c - exact[::-1][None, :]))) \
            < 1e3 * g.vscale * EPS

    def test_piecewise(self):
        # pass(5-8): chebpoly of piecewise abs(x) on [-1 0 1], both kinds
        # and both scalar and array-valued.
        import jax.numpy as jnp

        exact1 = np.array([2 / np.pi, 0.0, 4 / (3 * np.pi), 0.0,
                           -4 / (15 * np.pi), 0.0, 4 / (35 * np.pi)])
        exact2 = np.array([4 / (3 * np.pi), 0.0, 4 / (5 * np.pi), 0.0,
                           -4 / (21 * np.pi), 0.0, 4 / (45 * np.pi)])
        f = cj.chebfun(jnp.abs, domain=[-1.0, 0.0, 1.0])
        g = cj.chebfun(lambda x: jnp.stack([jnp.abs(x), jnp.abs(x)],
                                           axis=-1), domain=[-1.0, 0.0, 1.0])
        c = np.asarray(f.chebpoly(7))
        assert float(np.max(np.abs(c - exact1[::-1]))) < 10 * f.vscale * EPS
        c = np.asarray(g.chebpoly(7))
        assert c.shape == (2, 7)
        assert float(np.max(np.abs(c - exact1[::-1][None, :]))) \
            < 10 * g.vscale * EPS
        c = np.asarray(f.chebpoly(7, kind=2))
        assert float(np.max(np.abs(c - exact2[::-1]))) < 1e2 * f.vscale * EPS
        c = np.asarray(g.chebpoly(7, kind=2))
        assert float(np.max(np.abs(c - exact2[::-1][None, :]))) \
            < 1e2 * g.vscale * EPS
