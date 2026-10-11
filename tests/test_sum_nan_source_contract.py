"""Supplemental sumFullDom NaN cleanup; not original sum36 predicates."""

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


def test_complex_nan_cleanup():
    f = cj.Chebfun.from_coeffs(jnp.asarray([complex(float("nan"), 1)]))
    for result in (f.sum(), jax.jit(lambda: f.sum())()):
        assert np.isnan(np.real(result))
        assert np.imag(result) == 0


def test_finite_real_complex_row_array():
    for coefficients in ([2.0, 1.0], [2 + 3j, 1 - 2j], [[2 + 3j, 4 - 5j], [1, 2]]):
        f = cj.Chebfun.from_coeffs(jnp.asarray(coefficients))
        expected = 2 * jnp.asarray(coefficients)[0]
        np.testing.assert_array_equal(f.sum(), expected)
        np.testing.assert_array_equal(
            f.T.sum(), expected.reshape(-1, 1) if expected.ndim else expected
        )
