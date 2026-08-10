"""Port of MATLAB Chebfun tests/chebfun/test_chebcoeffs.m (Fable 5).

chebfunjax exposes coefficients via .coeffs (single-piece); MATLAB's
Bessel closed forms for the Chebyshev coefficients of cos(x) are the
reference.

Provenance
----------
MATLAB source : tests/chebfun/test_chebcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunChebcoeffs:
    def test_cos_coefficients_bessel_closed_form(self):
        f = cj.chebfun(jnp.cos)
        c = np.asarray(f.coeffs())[:5] if callable(
            getattr(f, "coeffs", None)) else np.asarray(f.coeffs)[:5]
        c_exact = np.array([jv(0, 1), 0.0, -2 * jv(2, 1), 0.0,
                            2 * jv(4, 1)])
        assert float(np.max(np.abs(c - c_exact))) < 100 * EPS

    def test_truncation_argument(self):
        # MATLAB pass(1): chebcoeffs(f, 5) of cos(x).  The 5th exact
        # coefficient 2*J_4(1) is written as 2*(-23*J_0(1) + 40*J_1(1))
        # in the MATLAB test (Bessel recurrence at x = 1).
        f = cj.chebfun(jnp.cos)
        c = np.asarray(f.chebcoeffs(5))
        c_exact = np.array([jv(0, 1), 0.0, -2 * jv(2, 1), 0.0,
                            2 * (-23 * jv(0, 1) + 40 * jv(1, 1))])
        assert c.shape == (5,)
        assert float(np.max(np.abs(c - c_exact))) < 1e2 * f.vscale * EPS

    def test_truncation_array_valued(self):
        # MATLAB pass(2): chebcoeffs([f f], 5).
        g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), jnp.cos(x)],
                                           axis=-1))
        c = np.asarray(g.chebcoeffs(5))
        col = np.array([jv(0, 1), 0.0, -2 * jv(2, 1), 0.0,
                        2 * (-23 * jv(0, 1) + 40 * jv(1, 1))])
        c_exact = np.stack([col, col], axis=-1)
        assert c.shape == (5, 2)
        assert float(np.max(np.abs(c - c_exact))) < 1e2 * g.vscale * EPS

    def test_second_kind_smooth(self):
        # MATLAB pass(3): chebcoeffs(f, 5, 'kind', 2) of cos(x).
        f = cj.chebfun(jnp.cos)
        c = np.asarray(f.chebcoeffs(5, kind=2))
        c_exact = np.array([2 * jv(1, 1), 0.0, -6 * jv(3, 1), 0.0,
                            2 * (-235 * jv(1, 1) + 900 * jv(2, 1))])
        assert float(np.max(np.abs(c - c_exact))) < 1e3 * f.vscale * EPS

    def test_second_kind_smooth_array_valued(self):
        # MATLAB pass(4): chebcoeffs([f f], 5, 'kind', 2).
        g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), jnp.cos(x)],
                                           axis=-1))
        c = np.asarray(g.chebcoeffs(5, kind=2))
        col = np.array([2 * jv(1, 1), 0.0, -6 * jv(3, 1), 0.0,
                        2 * (-235 * jv(1, 1) + 900 * jv(2, 1))])
        c_exact = np.stack([col, col], axis=-1)
        assert float(np.max(np.abs(c - c_exact))) < 1e3 * g.vscale * EPS

    def test_piecewise_first_kind(self):
        # MATLAB pass(5): chebcoeffs(abs(x), 7) on [-1 0 1].
        f = cj.chebfun(jnp.abs, domain=[-1.0, 0.0, 1.0])
        c = np.asarray(f.chebcoeffs(7))
        c_exact = np.array([2 / np.pi, 0.0, 4 / (3 * np.pi), 0.0,
                            -4 / (15 * np.pi), 0.0, 4 / (35 * np.pi)])
        assert float(np.max(np.abs(c - c_exact))) < 10 * f.vscale * EPS

    def test_piecewise_second_kind(self):
        # MATLAB pass(7): chebcoeffs(abs(x), 7, 'kind', 2) on [-1 0 1].
        f = cj.chebfun(jnp.abs, domain=[-1.0, 0.0, 1.0])
        c = np.asarray(f.chebcoeffs(7, kind=2))
        c_exact = np.array([4 / (3 * np.pi), 0.0, 4 / (5 * np.pi), 0.0,
                            -4 / (21 * np.pi), 0.0, 4 / (45 * np.pi)])
        assert float(np.max(np.abs(c - c_exact))) < 1e2 * f.vscale * EPS

    def test_piecewise_array_valued(self):
        # MATLAB pass(6, 8): the array-valued piecewise cases.
        g = cj.chebfun(lambda x: jnp.stack([jnp.abs(x), jnp.abs(x)],
                                           axis=-1), domain=[-1.0, 0.0, 1.0])
        col1 = np.array([2 / np.pi, 0.0, 4 / (3 * np.pi), 0.0,
                         -4 / (15 * np.pi), 0.0, 4 / (35 * np.pi)])
        c = np.asarray(g.chebcoeffs(7))
        assert c.shape == (7, 2)
        assert float(np.max(np.abs(c - np.stack([col1, col1], -1)))) \
            < 10 * g.vscale * EPS
        col2 = np.array([4 / (3 * np.pi), 0.0, 4 / (5 * np.pi), 0.0,
                         -4 / (21 * np.pi), 0.0, 4 / (45 * np.pi)])
        c = np.asarray(g.chebcoeffs(7, kind=2))
        assert float(np.max(np.abs(c - np.stack([col2, col2], -1)))) \
            < 1e2 * g.vscale * EPS

    def test_second_kind_of_U5(self):
        # MATLAB pass(9) (issue #1589): chebcoeffs(chebpoly(5, 2),
        # 'kind', 2) is the unit vector e_6.
        from chebfunjax.utils.polynomials import chebpoly
        f = cj.chebfun(chebpoly(5, kind=2), coeffs=True)
        c = np.asarray(f.chebcoeffs(kind=2))
        exact = np.concatenate([np.zeros(5), [1.0]])
        assert float(np.max(np.abs(c[:6] - exact))) < 1e-14

    def test_piecewise_without_n_raises(self):
        # MATLAB errors 'CHEBFUN:CHEBFUN:chebcoeffs:inputN' when N is
        # omitted for a piecewise chebfun.
        f = cj.chebfun(jnp.abs, domain=[-1.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            f.chebcoeffs()
