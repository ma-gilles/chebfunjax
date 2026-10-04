"""Focused tests for Chebyshev coefficients sampled at Legendre points."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.utils import chebcoeffs2legvals


def _independent_chebyshev_values(coefficients, nodes):
    coefficients = np.asarray(coefficients)
    if coefficients.ndim == 1:
        return np.polynomial.chebyshev.chebval(nodes, coefficients)
    return np.column_stack([
        np.polynomial.chebyshev.chebval(nodes, coefficients[:, column])
        for column in range(coefficients.shape[1])
    ])


def test_chebcoeffs2legvals_matches_independent_polynomial_evaluation():
    rng = np.random.default_rng(781)
    coefficients = rng.normal(size=13)
    nodes, _ = np.polynomial.legendre.leggauss(coefficients.size)
    actual = np.asarray(chebcoeffs2legvals(jnp.asarray(coefficients)))
    expected = _independent_chebyshev_values(coefficients, nodes)
    npt.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_chebcoeffs2legvals_supports_complex_batched_columns():
    rng = np.random.default_rng(782)
    coefficients = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    nodes, _ = np.polynomial.legendre.leggauss(coefficients.shape[0])
    actual = np.asarray(chebcoeffs2legvals(jnp.asarray(coefficients)))
    expected = _independent_chebyshev_values(coefficients, nodes)
    assert actual.shape == coefficients.shape
    npt.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_chebcoeffs2legvals_empty_and_singleton_inputs():
    empty = jnp.asarray([], dtype=jnp.float64)
    singleton = jnp.asarray([2.5 - 0.75j])
    assert chebcoeffs2legvals(empty).shape == (0,)
    npt.assert_array_equal(
        np.asarray(chebcoeffs2legvals(singleton)), np.asarray(singleton)
    )


def test_chebcoeffs2legvals_jit_matches_eager():
    coefficients = jnp.asarray([1.0, -0.5, 0.125, 2.0, -1.25])
    eager = chebcoeffs2legvals(coefficients)
    compiled = jax.jit(chebcoeffs2legvals)(coefficients)
    npt.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-14, atol=1e-14)
