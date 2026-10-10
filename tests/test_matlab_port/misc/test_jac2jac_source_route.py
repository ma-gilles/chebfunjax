"""Public jac2jac source-route controls against independent value evaluation."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_jacobi

from chebfunjax.utils.transforms import jac2jac


def _values(coeffs, a, b, x):
    coeffs = np.asarray(coeffs)
    mat = np.stack([eval_jacobi(k, a, b, x) for k in range(coeffs.shape[0])], axis=1)
    return mat @ coeffs


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("matrix_input", [False, True])
def test_public_jac2jac_preserves_polynomial_values(complex_input, matrix_input):
    n = 18
    k = np.arange(n, dtype=float)
    c = np.column_stack((np.sin(k + 0.2), np.cos(0.3 * k))) if matrix_input else np.sin(k + 0.2)
    if complex_input:
        c = c + 1j * (0.25 * np.cos(k + 0.1)[:, None] if matrix_input else 0.25 * np.cos(k + 0.1))
    x = np.linspace(-0.93, 0.91, 43)
    out = np.asarray(jac2jac(jnp.asarray(c), 0.2, -0.35, -0.4, 0.3))
    np.testing.assert_allclose(_values(out, -0.4, 0.3, x), _values(c, 0.2, -0.35, x), rtol=0, atol=2e-10)
    assert out.shape == c.shape
    assert np.iscomplexobj(out) == complex_input
