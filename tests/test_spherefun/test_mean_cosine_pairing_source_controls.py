# uses-numpy: Independent reference fixtures and numeric assertions in tests.
"""Static test draft for the MATLAB source coefficient pairing contract.

Provenance: MATLAB Chebfun 7574c77680d7e82b79626300bf255498271a72df,
@chebfun/trigcoeffs.m (two outputs), @trigtech/prolong.m, and
@spherefun/sum.m. Independent coefficient fixtures cover odd/even lengths,
Nyquist handling, common-length prolongation, trailing columns and complex
coefficient preservation. Draft only; intentionally not executed here.
"""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech


def cosine_coeffs(coeffs, n=None, known_real=False):
    mod = importlib.import_module("chebfunjax.spherefun.spherefun")
    tech = Trigtech.from_coeffs(jnp.asarray(coeffs), is_real=known_real)
    return mod._sphere_mean_cosine_coefficients(
        tech, tech.n if n is None else n, known_real=known_real
    )


@pytest.mark.parametrize(
    "coeffs,expected",
    [
        ([0.5, 0.0, 0.5], [0.0, 1.0]),  # cos(theta), no even cosine term
        ([0.5, 0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0]),
        ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  # unsplit Nyquist cos(2theta)
        ([2.0], [2.0]),
        ([3.0, 2.0], [2.0, 3.0]),  # even n2, DC then Nyquist
    ],
)
def test_cosine_pairing_and_nyquist(coeffs, expected):
    np.testing.assert_array_equal(cosine_coeffs(coeffs), expected)


def test_prolonged_nyquist_is_not_counted_twice():
    # Source prolong splits -2/+2; source trigcoeffs adds them back once.
    np.testing.assert_array_equal(cosine_coeffs([1.0, 0.0, 0.0, 0.0], n=7), [0.0, 0.0, 1.0, 0.0])


def test_axis_zero_preserves_array_columns_and_complex_pairs():
    c = np.array([[1 + 2j, 3 - 1j], [2 - 1j, 4 + 3j], [5 + 7j, 6 - 2j]])
    expected = np.stack((c[1], c[0] + c[2]))
    np.testing.assert_array_equal(cosine_coeffs(c), expected)
    np.testing.assert_array_equal(cosine_coeffs(c, known_real=True), expected.real)
