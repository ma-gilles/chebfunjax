"""Native stored point ownership for scalar and complex denominators."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d.chebfun import Chebfun


@pytest.mark.parametrize(
    "index,coefficients",
    enumerate(([0.0, 1.0], [0.0 + 0j, 1.0 + 0j], [0.0, 1.0 + 1j], [1j, 1.0]), 1),
)
def test_native_complex_denominator_pointvalues(index, coefficients):
    r = Path(__file__).parent / "fixtures" / "rdivide_pointvalues_native" / "native_complex_denominator_v1"
    native = loadmat(r / f"case{index}.mat", simplify_cells=True)["rec"]
    quotient = 1 / Chebfun.from_coeffs(jnp.asarray(coefficients))
    np.testing.assert_allclose(quotient.point_values, native["pointValues"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        quotient(jnp.asarray(native["domain"])), native["pointValues"], rtol=0, atol=2e-14
    )


@pytest.mark.parametrize("index,numerator", enumerate((1, -1, 1j, 1 + 1j, complex(1, 0), 0), 1))
def test_native_scalar_numerator_zero_pointvalues(index, numerator):
    r = Path(__file__).parent / "fixtures" / "rdivide_pointvalues_native" / "native_zero_pointvalues_v1"
    native = loadmat(r / f"case{index}.mat", simplify_cells=True)["rec"]
    quotient = numerator / Chebfun.from_coeffs(jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(quotient.domain.breakpoints, native["domain"], rtol=0, atol=0)
    np.testing.assert_allclose(quotient.point_values, native["pointValues"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        quotient(jnp.asarray(native["domain"])), native["pointValues"], rtol=0, atol=2e-14
    )
