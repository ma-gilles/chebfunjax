"""Scratch source controls for ratinterp coefficient trimming; not installed/run.

Provenance
----------
MATLAB source : ratinterp.m, trimCoeffs (Chebfun commit 7574c77)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.ratapprox import _qr_to_cheb_basis, _trim_coeffs, ratinterp


@pytest.mark.parametrize("which", ["numerator", "denominator"])
def test_trim_preserves_small_nonzero_imaginary_coefficient(which):
    tiny = 5e-15
    a = jnp.asarray([1.0 + (1j * tiny if which == "numerator" else 0.0j)])
    b = jnp.asarray([1.0 + (1j * tiny if which == "denominator" else 0.0j)])
    at, bt = _trim_coeffs(a, b, tol=1e-14, ts=1e-14)
    np.testing.assert_allclose(
        np.imag(np.asarray(at)), [tiny if which == "numerator" else 0.0],
        atol=1e-30, rtol=0,
    )
    np.testing.assert_allclose(
        np.imag(np.asarray(bt)), [tiny if which == "denominator" else 0.0],
        atol=1e-30, rtol=0,
    )


@pytest.mark.parametrize("grid", ["type0", "type1", "type2", "equi"])
def test_public_small_complex_constant_is_not_realified(grid):
    tiny = 5e-15

    def constant(x):
        return jnp.ones_like(x, dtype=jnp.complex128) * (1.0 + 1j * tiny)

    r, a, b, mu, nu, _, _ = ratinterp(
        constant, 0, 0, NN=8, xi=grid, tol=1e-14
    )
    assert (mu, nu) == (0, 0)
    assert np.iscomplexobj(np.asarray(a))
    np.testing.assert_allclose(np.imag(np.asarray(a)[0] / np.asarray(b)[0]), tiny, atol=1e-15, rtol=0)
    values = np.asarray(r(np.asarray([-0.61, 0.13, 0.72])))
    np.testing.assert_allclose(values.imag, np.full(3, tiny), atol=1e-15, rtol=0)


def test_trim_tol_zero_preserves_values_dtype_and_lengths():
    a = jnp.asarray([1e-20 + 3e-15j, 0.0 + 0.0j])
    b = jnp.asarray([2e-20 - 5e-15j, 0.0 + 0.0j])
    at, bt = _trim_coeffs(a, b, tol=0.0, ts=0.0)
    np.testing.assert_array_equal(np.asarray(at), np.asarray(a))
    np.testing.assert_array_equal(np.asarray(bt), np.asarray(b))
    assert at.shape == a.shape and at.dtype == a.dtype
    assert bt.shape == b.shape and bt.dtype == b.dtype


def test_trim_uses_strict_source_thresholds_for_tail_equality():
    threshold = 1e-14
    at, bt = _trim_coeffs(
        jnp.asarray([1.0, threshold]), jnp.asarray([1.0, threshold]),
        tol=threshold, ts=threshold,
    )
    np.testing.assert_array_equal(np.asarray(at), [1.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0])


def test_trim_leading_removal_stops_when_either_leading_value_is_large():
    threshold = 1e-14
    at, bt = _trim_coeffs(
        jnp.asarray([0.5 * threshold, 2.0]),
        jnp.asarray([2.0 * threshold, 1.0]),
        tol=threshold, ts=threshold,
    )
    np.testing.assert_array_equal(np.asarray(at), [0.5 * threshold, 2.0])
    np.testing.assert_array_equal(np.asarray(bt), [2.0 * threshold, 1.0])


def test_trim_all_underthreshold_numerator_uses_zero_function_fallback():
    at, bt = _trim_coeffs(
        jnp.asarray([0.5e-14]), jnp.asarray([2.0]), tol=1e-14, ts=1e-14
    )
    np.testing.assert_array_equal(np.asarray(at), [0.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0])


def test_empty_numerator_uses_zero_fallback_even_when_tolerance_is_zero():
    at, bt = _trim_coeffs(jnp.asarray([]), jnp.asarray([2.0]), tol=0.0, ts=0.0)
    np.testing.assert_array_equal(np.asarray(at), [0.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0])


def test_empty_denominator_does_not_trigger_zero_numerator_fallback():
    at, bt = _trim_coeffs(jnp.asarray([2.0]), jnp.asarray([]), tol=0.0, ts=0.0)
    np.testing.assert_array_equal(np.asarray(at), [2.0])
    assert bt.shape == (0,)


def test_trim_zero_function_returns_source_zero_rational_pair():
    at, bt = _trim_coeffs(
        jnp.asarray([0.0 + 0.0j, 0.0 + 0.0j]),
        jnp.asarray([0.0 + 0.0j, 0.0 + 0.0j]),
        tol=1e-14, ts=0.0,
    )
    np.testing.assert_array_equal(np.asarray(at), [0.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0])


def test_trim_removes_successive_leading_terms_only_as_joint_small_pairs():
    threshold = 1e-14
    at, bt = _trim_coeffs(
        jnp.asarray([0.5 * threshold, 0.25 * threshold, 2.0]),
        jnp.asarray([0.2 * threshold, 0.5 * threshold, 1.0]),
        tol=threshold, ts=threshold,
    )
    np.testing.assert_array_equal(np.asarray(at), [2.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0])


def test_trim_uses_numerator_ts_and_denominator_tol_for_tail_rules():
    at, bt = _trim_coeffs(
        jnp.asarray([1.0, 1.5e-14]),
        jnp.asarray([1.0, 1.5e-14]),
        tol=1e-14, ts=2e-14,
    )
    np.testing.assert_array_equal(np.asarray(at), [1.0])
    np.testing.assert_array_equal(np.asarray(bt), [1.0, 1.5e-14])


def test_qr_conversion_keeps_complex_basis_with_real_input_coefficients():
    # constructRatApproxArb divides by R even if the orthogonal coefficients
    # are real; the basis can carry the complex phase.
    a, b = _qr_to_cheb_basis(jnp.asarray([2.0]), jnp.asarray([1.0]),
                            jnp.asarray([[1 + 0.5j]]), 1)
    np.testing.assert_allclose(a, [2 / (1 + 0.5j)], atol=1e-15, rtol=0)
    np.testing.assert_allclose(b, [1 / (1 + 0.5j)], atol=1e-15, rtol=0)


def test_qr_conversion_jit_and_derivative_preserve_tiny_complex_coefficients():
    R = jnp.asarray([[2.0, 0.5], [0.0, 4.0]])
    expected_a = jnp.asarray([1 + 5e-15j, 0.25 - 3e-15j])
    expected_b = jnp.asarray([1.0])
    a_hat = R @ expected_a

    def numerator(values):
        return _qr_to_cheb_basis(values, jnp.asarray([2.0]), R, 2)[0]

    a = jax.jit(numerator)(a_hat)
    _, b = jax.jit(lambda values: _qr_to_cheb_basis(values, jnp.asarray([2.0]), R, 2))(a_hat)
    np.testing.assert_allclose(a.real, expected_a.real, atol=1e-15, rtol=0)
    np.testing.assert_allclose(a.imag, expected_a.imag, atol=1e-30, rtol=0)
    np.testing.assert_allclose(b, expected_b, atol=1e-15, rtol=0)
    # The independent inverse of this two-by-two triangular matrix.
    derivative = jax.jacfwd(numerator, holomorphic=True)(a_hat)
    np.testing.assert_allclose(derivative, [[0.5, -0.0625], [0.0, 0.25]],
                               atol=1e-15, rtol=0)
