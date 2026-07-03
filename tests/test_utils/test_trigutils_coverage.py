"""Correctness tests for :mod:`chebfunjax.utils.trigutils`.

``trigutils`` provides trigonometric-polynomial helpers (``trigpoly``) and the
derivative of a trigonometric barycentric rational (``diffbarytrig``), plus the
scalar building blocks that supply the n-th derivatives of csc / cot / sin /
tan used by the barycentric formula.

The individual derivative helpers are verified against closed-form
derivatives / finite differences (all correct). ``diffbarytrig`` itself is
exercised against analytic derivatives of an entire periodic function whose
AAAtrig rational is accurate to machine precision; those tests are marked
These document bugs that were fixed (previously off by a constant factor
(see :class:`TestDiffBaryTrigBug`).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils import trigutils as tu
from chebfunjax.utils.aaa import aaatrig
from chebfunjax.utils.trigutils import diffbarytrig, trigpoly

_T = np.array([0.4, 1.1, 2.3, -0.7, 0.95])


def _fd(f, t, h=1e-6):
    return (f(t + h) - f(t - h)) / (2.0 * h)


# ---------------------------------------------------------------------------
# trigpoly
# ---------------------------------------------------------------------------

class TestTrigpolyMore:
    def test_multi_degree_columns(self):
        # A vector of degrees yields one column per degree, each exp(i pi n x).
        out = np.asarray(trigpoly([0, 1, 2]))
        assert out.shape == (5, 3)
        # Column 0 is degree 0 -> all ones.
        npt.assert_allclose(out[:, 0], np.ones(5), atol=1e-14)
        # |exp(...)| == 1 everywhere.
        npt.assert_allclose(np.abs(out), 1.0, atol=1e-14)

    def test_degree_zero_is_constant_one(self):
        out = np.asarray(trigpoly(0))
        npt.assert_allclose(out, np.ones(1), atol=1e-14)

    def test_domain_scaling_period(self):
        # exp(i*2pi/L*k*(x-a)) has unit modulus and correct grid length.
        out = np.asarray(trigpoly(2, domain=(0.0, 4.0)))
        assert out.shape == (5,)
        npt.assert_allclose(np.abs(out), 1.0, atol=1e-14)
        # First sample sits at x==a -> value exp(0) == 1.
        npt.assert_allclose(out[0], 1.0 + 0.0j, atol=1e-14)

    def test_negative_degree_conjugates(self):
        pos = np.asarray(trigpoly(3))
        neg = np.asarray(trigpoly(-3))
        npt.assert_allclose(neg, np.conj(pos), atol=1e-13)


# ---------------------------------------------------------------------------
# Scalar n-th derivative helpers (all correct)
# ---------------------------------------------------------------------------

class TestDerivHelpers:
    def test_diff_sin(self):
        npt.assert_allclose(tu._diff_sin(_T, 1), np.cos(_T), atol=1e-14)
        npt.assert_allclose(tu._diff_sin(_T, 2), -np.sin(_T), atol=1e-14)
        npt.assert_allclose(tu._diff_sin(_T, 4), np.sin(_T), atol=1e-14)

    def test_diff_cot_first_derivative(self):
        # d/dt cot(t) = -csc^2(t).
        npt.assert_allclose(tu._diff_cot_scalar(_T, 1),
                            -1.0 / np.sin(_T) ** 2, atol=1e-13)

    def test_diff_cot_second_derivative_matches_fd(self):
        d2 = tu._diff_cot_scalar(_T, 2)
        npt.assert_allclose(d2, _fd(lambda x: tu._diff_cot_scalar(x, 1), _T),
                            atol=1e-6)

    def test_diff_tan_first_derivative(self):
        # d/dt tan(t) = sec^2(t) = 1 + tan^2(t).
        npt.assert_allclose(tu._diff_tan(_T, 1), 1.0 / np.cos(_T) ** 2, atol=1e-13)

    def test_diff_csc_first_derivative(self):
        # d/dt csc(t) = -csc(t) cot(t).
        expected = -1.0 / np.sin(_T) * (1.0 / np.tan(_T))
        npt.assert_allclose(tu._diff_csc(_T, 1), expected, atol=1e-13)

    def test_diff_cst_dispatch(self):
        npt.assert_allclose(tu._diff_cst(_T, 2, "even"),
                            tu._diff_cot_scalar(_T, 2), atol=0)
        npt.assert_allclose(tu._diff_cst(_T, 2, "odd"),
                            tu._diff_csc(_T, 2), atol=0)

    def test_diff_cst_inv_dispatch(self):
        npt.assert_allclose(tu._diff_cst_inv(_T, 1, "even"),
                            tu._diff_tan(_T, 1), atol=0)
        npt.assert_allclose(tu._diff_cst_inv(_T, 1, "odd"),
                            tu._diff_sin(_T, 1), atol=0)

    def test_binom(self):
        assert tu._binom(5, 2) == 10.0
        assert tu._binom(4, 0) == 1.0


# ---------------------------------------------------------------------------
# diffbarytrig input validation (correct)
# ---------------------------------------------------------------------------

class TestDiffBaryTrigValidation:
    def _build(self, form="odd"):
        Z = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, *_rest, zj, fj, wj, _ = aaatrig(np.exp(np.sin(Z)), Z, form=form)
        return zj, fj, wj

    def test_order_zero_raises(self):
        zj, fj, wj = self._build()
        with pytest.raises(ValueError, match="revaltrig"):
            diffbarytrig(jnp.asarray([0.5, 1.0]), zj, fj, wj, 0, "odd")

    def test_output_shape_preserved(self):
        zj, fj, wj = self._build()
        zz = np.array([[0.5, 1.3], [2.7, 4.1]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = diffbarytrig(jnp.asarray(zz), zj, fj, wj, 1, "odd")
        assert d.shape == zz.shape


# ---------------------------------------------------------------------------
# Confirmed bug: diffbarytrig is off by a factor of 4**N.
#
# MATLAB diffbarytrig.m line 65 uses 2.^(q-p); the Python port wrote
# 0.5 ** (q - p) (== 2 ** (p - q)), inverting the chain-rule half-angle
# factor.  For a machine-precision AAAtrig rational of exp(sin z) the first
# derivative comes out at exactly 4x the true value, the second at 16x, etc.
# The tests below assert the mathematically correct derivative and run the
# full assembly (so it stays covered); they xpass once the factor is fixed.
# ---------------------------------------------------------------------------

class TestDiffBaryTrigBug:
    def _build(self, form="odd"):
        Z = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, *_rest, zj, fj, wj, _ = aaatrig(np.exp(np.sin(Z)), Z, form=form)
        return r, zj, fj, wj

    @pytest.mark.parametrize("form", ["odd", "even"])
    def test_first_derivative(self, form):
        r, zj, fj, wj = self._build(form)
        zz = np.array([0.5, 1.3, 2.7, 4.1, 5.5])
        true = np.cos(zz) * np.exp(np.sin(zz))  # d/dz exp(sin z)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.asarray(diffbarytrig(jnp.asarray(zz), zj, fj, wj, 1, form)).real
        npt.assert_allclose(d, true, rtol=0, atol=1e-6)

    def test_second_derivative(self):
        r, zj, fj, wj = self._build("odd")
        zz = np.array([0.5, 1.3, 2.7, 4.1])
        # d2/dz2 exp(sin z) = (cos^2 z - sin z) exp(sin z).
        true = (np.cos(zz) ** 2 - np.sin(zz)) * np.exp(np.sin(zz))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.asarray(diffbarytrig(jnp.asarray(zz), zj, fj, wj, 2, "odd")).real
        npt.assert_allclose(d, true, rtol=0, atol=1e-5)

    def test_derivative_at_support_point(self):
        r, zj, fj, wj = self._build("odd")
        # Evaluate exactly at a support point to trigger the NaN 0/0 fix-up.
        zpt = np.real(np.asarray(zj)[1])
        true = np.cos(zpt) * np.exp(np.sin(zpt))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.asarray(diffbarytrig(jnp.asarray([zpt]), zj, fj, wj, 1, "odd")).real
        npt.assert_allclose(d, [true], rtol=0, atol=1e-6)
