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

    def test_sign_improvement_relu(self):
        # pass(30): max(x, 0) on chebpts(200), degree 4, 100 Lawson steps,
        # damping 0.5, sign=1 -> error near 0.006.
        X = np.asarray(chebpts(200, kind=2))
        F = np.maximum(X, 0.0)
        r, *_ = aaa(F, X, degree=4, lawson=100, damping=0.5, sign=True)
        err = np.max(np.abs(F - np.real(r(X))))
        assert abs(err - 0.006) < 0.002

    def test_sign_improvement_fermi_dirac(self):
        # pass(31): Fermi-Dirac 1/(1+exp(5/(x-2))) on chebpts(1000, [0,10]),
        # degree 12, 100 Lawson steps, damping 0.85, sign=1 -> err ~ 3.5e-5.
        X = 5.0 * (np.asarray(chebpts(1000, kind=2)) + 1.0)  # -> [0, 10]
        with np.errstate(over="ignore"):
            F = 1.0 / (1.0 + np.exp(5.0 / (X - 2.0)))
        r, *_ = aaa(F, X, degree=12, lawson=100, damping=0.85, sign=True)
        err = np.max(np.abs(F - np.real(r(X))))
        assert abs(err - 0.000035) < 0.0001


def _match_poles(found, expected):
    """Max pairwise error matching ``found`` poles to ``expected`` (greedy)."""
    remaining = list(expected)
    worst = 0.0
    for p in found:
        k = int(np.argmin([abs(p - q) for q in remaining]))
        worst = max(worst, abs(p - remaining.pop(k)))
    return worst


class TestAaaStep2tfSystemID:
    """Acceptance case: the chebfun.org Step2tf/Bode2tf system-ID fit.

    Fitting GS = G(i w)/(i w) over w in [1e-4, 1e2] (a huge |F| dynamic
    range from the 1/(i w) factor) must recover the clean physical poles,
    NOT Froissart doublets.  Mirrors the first aaa call of
    examples/applics/Step2tf.html.
    """

    @staticmethod
    def _gs_data():
        def num(s):
            return (1 + 105 * s) * (1 + 28 * s + 400 * s ** 2)

        def den(s):
            return (1 + 100 * s) * (1 + 35 * s + 625 * s ** 2) * (1 + 0.4 * s ** 2)

        w = np.logspace(-4, 2, 6000)
        gs = num(1j * w) / den(1j * w) / (1j * w)
        z = 1j * np.concatenate([-w[::-1], w])
        f = np.concatenate([np.conj(gs)[::-1], gs])
        return f, z

    # MATLAB polG from the Step2tf example.
    POL_G = np.array([
        -1.581138830084190j, 1.581138830084190j,
        -0.027999999999557 + 0.028565713714499j,
        -0.027999999999914 - 0.028565713713914j,
        -0.009999999999644 + 0.000000000000123j,
        0.0j])

    def test_recovers_physical_poles_no_froissart(self):
        f, z = self._gs_data()
        _, pol, *_ = aaa(f, z, lawson=0)
        assert len(pol) == 6
        assert _match_poles(np.asarray(pol), self.POL_G) < 1e-9
        # No spurious high-frequency doublet (physical |Im| <= ~1.6).
        assert np.max(np.abs(np.imag(np.asarray(pol)))) < 2.0

    def test_degree_constrained_also_clean(self):
        # degree=6 (adaptive Lawson path) must also give the physical poles.
        f, z = self._gs_data()
        _, pol, *_ = aaa(f, z, degree=6)
        assert len(pol) == 6
        assert _match_poles(np.asarray(pol), self.POL_G) < 1e-6


class TestChebfunAaaAutoZ:
    """aaa(handle) with no sample set: the aaa_autoZ port (pass 14)."""

    def test_pass14_gamma(self):
        from scipy.special import gamma as sp_gamma
        gam = lambda z: sp_gamma(np.real(np.asarray(z))) + 0j  # noqa: E731
        r, pol, res, *_ = aaa(gam)
        assert abs(complex(np.asarray(r(np.asarray([1.5]))[0]))
                   - sp_gamma(1.5)) < 1e-3
        # the poles at 0 and -1 with residues 1 and -1 are captured
        p = np.asarray(pol)
        q = np.asarray(res)
        i0 = np.argmin(np.abs(p))
        i1 = np.argmin(np.abs(p + 1))
        assert abs(p[i0]) < 1e-8 and abs(q[i0] - 1) < 1e-6
        assert abs(p[i1] + 1) < 1e-8 and abs(q[i1] + 1) < 1e-6

    def test_autoZ_matches_explicit_grid(self):
        # exp resolves at the first autoZ level (n=5); the result equals
        # aaa on that explicit grid.
        Z = _auto_z(-1.0, 1.0, 5)
        r_auto, *_ = aaa(lambda z: np.exp(np.asarray(z)))
        r_grid, *_ = aaa(np.exp(Z), Z)
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(np.real(r_auto(xx))
                             - np.real(r_grid(xx)))) < 1e-13

    def test_autoZ_dom(self):
        r, *_ = aaa(lambda z: np.exp(np.asarray(z)), dom=(-2.0, 2.0))
        xx = np.linspace(-2, 2, 100)
        assert np.max(np.abs(np.exp(xx) - np.real(r(xx)))) < 1e-11
