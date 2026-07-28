"""Port of MATLAB Chebfun tests/chebfun/test_aaa.m (Fable 5).

Focuses on the AAA-Lawson assertions (pass 21-29, 32).  chebfunjax's
``aaa`` is array-based, so the MATLAB function-handle calls ``aaa(f, ...)``
that rely on the internal auto-sampling are reproduced here by sampling on
the same grid MATLAB's ``aaa_autoZ`` uses (an unresolved function drives it
to n = 14).  The ``'sign'`` cases (pass 30, 31) are skipped: chebfunjax
does not expose the sign-function improvement.

Provenance
----------
MATLAB source : tests/chebfun/test_aaa.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.quadrature import chebpts


def _auto_z(a, b, n):
    # MATLAB aaa_autoZ sampling grid.
    length = b - a
    return np.linspace(a + 1.37e-8 * length, b - 3.08e-9 * length, 1 + 2 ** n)


ERR_MINIMAX = 1.550669058714149e-7


class TestChebfunAaa:
    def test_lawson_matches_minimax_degree3(self):
        # pass(21): degree-3 (Lawson on by default) reaches near-minimax.
        xx = np.linspace(-1, 1, 100)
        Z = _auto_z(-1.0, 1.0, 14)
        r, *_ = aaa(np.exp(Z), Z, degree=3)
        err = np.max(np.abs(np.exp(xx) - np.real(r(xx))))
        assert err / ERR_MINIMAX < 1.1

    def test_mmax_does_not_run_lawson(self):
        # pass(22): mmax (no Lawson) stays well above the minimax error.
        xx = np.linspace(-1, 1, 100)
        Z = _auto_z(-1.0, 1.0, 14)
        r, *_ = aaa(np.exp(Z), Z, mmax=4)
        err = np.max(np.abs(np.exp(xx) - np.real(r(xx))))
        assert err / ERR_MINIMAX > 1.1

    def test_lawson_bailout_symmetry(self):
        # pass(24): on a symmetric problem mmax (=> no Lawson) matches
        # lawson=0 exactly.
        Z = np.exp(2j * np.pi * np.arange(1, 501) / 500)
        F = np.log(2 - Z ** 4)
        n = 15
        r, *_ = aaa(F, Z, mmax=n + 1, lawson=0)
        e1 = np.max(np.abs(F - r(Z)))
        r, *_ = aaa(F, Z, mmax=n + 1)
        e2 = np.max(np.abs(F - r(Z)))
        assert abs(e2 / e1 - 1) < 1.01

    def test_lawson_bailout_troublesome_poles(self):
        # pass(25): sign data on two intervals; mmax => no Lawson == lawson=0.
        za = np.linspace(-3, -1, 1000)
        zb = np.linspace(1, 3, 1000)
        Z = np.concatenate([za, zb])
        F = np.concatenate([np.sign(za), np.sign(zb)])
        n = 12
        r, *_ = aaa(F, Z, mmax=n + 1, lawson=0)
        e1 = np.max(np.abs(F - r(Z)))
        r, *_ = aaa(F, Z, mmax=n + 1)
        e2 = np.max(np.abs(F - r(Z)))
        assert abs(e2 / e1 - 1) < 1.01

    def test_degree_pole_count(self):
        # pass(26)/(27): both degree=3 and mmax=4 yield exactly 3 poles.
        Z = np.linspace(-1, 1, 1000)
        F = np.exp(Z)
        _, pol, *_ = aaa(F, Z, degree=3)
        assert len(pol) == 3
        _, pol2, *_ = aaa(F, Z, mmax=4)
        assert len(pol2) == 3

    def test_exact_interpolation_small_data(self):
        # pass(28): F = [1, 0, 0] at X = [1, 2, 3] is interpolated exactly.
        X = np.array([1.0, 2.0, 3.0])
        F = np.array([1.0, 0.0, 0.0])
        r, *_ = aaa(F, X)
        assert np.max(np.abs(F - np.real(r(X)))) == 0.0

    def test_lawson_relu_degree4(self):
        # pass(29): max(x, 0) on chebpts(200), degree 4, 100 Lawson steps,
        # damping 0.2 -> error near 0.006.
        X = np.asarray(chebpts(200, kind=2))
        F = np.maximum(X, 0.0)
        r, *_ = aaa(F, X, degree=4, lawson=100, damping=0.2)
        err = np.max(np.abs(F - np.real(r(X))))
        assert abs(err - 0.006) < 0.002

    def test_lawson_relu_degree8(self):
        # pass(32): max(x, 0), degree 8, damping 0.5, 200 Lawson steps
        # -> error near 0.0006.
        Z = _auto_z(-1.0, 1.0, 14)
        F = np.maximum(Z, 0.0)
        r, *_ = aaa(F, Z, degree=8, damping=0.5, lawson=200)
        xx = np.linspace(-1, 1, 300)
        err = np.max(np.abs(np.maximum(xx, 0.0) - np.real(r(xx))))
        assert abs(err - 0.0006) < 0.001
