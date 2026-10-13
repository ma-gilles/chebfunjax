"""Correctness tests for :mod:`chebfunjax.utils.trigutils`.

``trigutils`` provides trigonometric-polynomial helpers (``trigpoly``) and the
derivative of a trigonometric barycentric rational (``diffbarytrig``), plus the
scalar building blocks that supply the n-th derivatives of csc / cot / sin /
tan used by the barycentric formula.

The individual derivative helpers are verified against closed-form
derivatives / finite differences (all correct). ``diffbarytrig`` itself is
exercised against analytic derivatives of an entire periodic function whose
AAAtrig rational is accurate to machine precision. The original MATLAB
half-angle formula is covered by the analytic derivative checks in
:class:`TestDiffBaryTrigAccuracy`.
"""

from __future__ import annotations

import warnings

import jax
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
    def test_multi_degree_coefficients_and_values(self):
        f = trigpoly(np.array([[-1, 0, 2]]))
        assert np.asarray(f.coeffs).shape == (5, 3)
        npt.assert_array_equal(np.asarray(f.coeffs), np.eye(5)[:, [3, 2, 0]])
        values = np.asarray(f(np.array([-1.0, 0.0, 1.0])))
        expected = np.exp(-1j * np.pi * np.array([-1.0, 0.0, 1.0])[:, None]
                          * np.array([-1, 0, 2])[None, :])
        npt.assert_allclose(values, expected, atol=1e-14)
        assert not f.is_transposed

    def test_degree_zero_is_constant_one(self):
        f = trigpoly(0)
        values = np.asarray(f(np.linspace(-1.0, 1.0, 9))).reshape(-1)
        npt.assert_allclose(values, np.ones(9), atol=1e-14)

    def test_unit_modulus_and_degree_based_representation_length(self):
        points = np.linspace(-1.0, 1.0, 37)
        for degree in [1, 2, 3, 5]:
            f = trigpoly(degree)
            assert f.coeffs.shape[0] == 2 * degree + 1
            values = np.asarray(f(points)).reshape(-1)
            npt.assert_allclose(np.abs(values), 1.0, atol=1e-14)

    def test_negative_degree_is_conjugate(self):
        points = np.linspace(-1.0, 1.0, 37)
        positive = np.asarray(trigpoly(3)(points)).reshape(-1)
        negative = np.asarray(trigpoly(-3)(points)).reshape(-1)
        npt.assert_allclose(negative, np.conj(positive), atol=1e-14)

    def test_column_input_transposes_and_empty_is_chebfun(self):
        f = trigpoly(np.array([[-1], [0], [2]]))
        assert f.is_transposed
        assert f.__class__.__name__ == "Chebfun"
        empty = trigpoly(np.array([]))
        assert empty.isempty()
        assert empty.__class__.__name__ == "Chebfun"

    def test_default_domain_uses_chebfunpref_and_restores_defaults(self):
        from chebfunjax.chebpref import ChebfunPref
        saved = ChebfunPref()
        try:
            ChebfunPref.setDefaults("domain", (2.0, 5.0))
            f = trigpoly(1)
            npt.assert_array_equal(np.asarray(f.domain.breakpoints), [2.0, 5.0])
        finally:
            ChebfunPref.setDefaults(saved)

    def test_domain_and_source_errors(self):
        f = trigpoly(1, domain=(2.0, 5.0))
        npt.assert_allclose(np.asarray(f([2.0, 3.5, 5.0])).reshape(-1),
                            [-1, 1, -1], atol=1e-14)
        with pytest.raises(ValueError, match="CHEBFUN:trigpoly:infdomain"):
            trigpoly(1, domain=(-np.inf, np.inf))
        with pytest.raises(ValueError, match="CHEBFUN:trigpoly:breakpoints"):
            trigpoly(1, domain=(-1.0, 0.0, 1.0))


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
# End-to-end derivative accuracy for the source half-angle formula.
# The implementation uses the native 2**(q-p) coefficient. Keep these
# analytic bounds unchanged while the kernel is migrated to JAX.
# ---------------------------------------------------------------------------

class TestDiffBaryTrigAccuracy:
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


class TestDiffBaryTrigJaxContract:
    def test_native_form_prefixes_and_optional_argument_order(self):
        zj = jnp.array([0.2, 1.1, 2.4, 4.6])
        fj = jnp.array([1.0, -0.7, 0.4, 1.2])
        wj = jnp.array([0.7, -1.2, 0.8, 0.3])
        zz = jnp.array([0.35, 1.7, 3.1])
        even = diffbarytrig(zz, zj, fj, wj, 2, "even")
        for form in ("Even", "evening"):
            npt.assert_allclose(diffbarytrig(zz, zj, fj, wj, form, 2), even,
                                rtol=0, atol=0)
        odd = diffbarytrig(zz, zj, fj, wj, 2, "odd")
        npt.assert_allclose(diffbarytrig(zz, zj, fj, wj, "ODDtail", 2), odd,
                            rtol=0, atol=0)

    def test_static_order_form_jit_including_support_repair(self):
        zj = jnp.array([0.2, 1.1, 2.4, 4.6])
        fj = jnp.array([1 + 0.3j, -0.7 + 0.2j, 0.4 - 0.8j, 1.2])
        wj = jnp.array([0.7, -1.2 + 0.1j, 0.8, 0.3 - 0.2j])
        zz = jnp.array([0.35, 0.2, 3.1])
        compiled = jax.jit(lambda x, y, f, w: diffbarytrig(x, y, f, w, 2, "odd"))
        npt.assert_allclose(compiled(zz, zj, fj, wj),
                            diffbarytrig(zz, zj, fj, wj, 2, "odd"),
                            rtol=2e-13, atol=2e-13)
