"""Native small-N Jacobi-to-Chebyshev FFT preserves columns and complex data."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_jacobi

from chebfunjax.utils.transforms import jac2cheb


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("complex_input", [False, True])
def test_jacobi_polynomial_columns(matrix, complex_input):
    n = 23
    coefficients = np.zeros(n, dtype=complex if complex_input else float)
    coefficients[[0, 3, 7]] = [1, .3, -.2]
    if complex_input:
        coefficients[[1, 5]] = [2j, -.4j]
    if matrix:
        coefficients = np.column_stack((coefficients, .5*coefficients[::-1]))
    x = np.linspace(-.9, .9, 31)
    polynomials = np.column_stack([eval_jacobi(k, -.3, .7, x) for k in range(n)])
    expected = polynomials @ coefficients
    converted = np.asarray(jac2cheb(jnp.asarray(coefficients), -.3, .7))
    actual = np.polynomial.chebyshev.chebval(x, converted)
    if matrix:
        expected = expected.T
    assert converted.shape == coefficients.shape
    assert np.iscomplexobj(converted) == complex_input
    assert np.max(np.abs(actual-expected)) < n*n*np.finfo(float).eps
