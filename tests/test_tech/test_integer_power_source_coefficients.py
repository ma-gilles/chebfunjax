"""Independent nonvanishing polynomial controls for source integer powers.

Provenance
----------
MATLAB source : @chebtech/power.m, @chebtech/compose.m
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech_cls", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("power", [3, 4])
@pytest.mark.parametrize("array_valued", [False, True])
def test_nonvanishing_dyadic_chebyshev_power(tech_cls, power, array_valued):
    # Expand (A+B*T32)^p analytically using Tn^2=(1+T2n)/2.
    # Both columns stay strictly positive, satisfying power.m's zero guard.
    a = np.array([2.0, 3.0]) if array_valued else np.array(2.0)
    b = np.array([0.25, -0.5]) if array_valued else np.array(0.25)
    coeffs = np.zeros((33,) + a.shape)
    coeffs[0], coeffs[32] = a, b
    expected = np.zeros((32 * power + 1,) + a.shape)
    if power == 3:
        expected[0] = a**3 + 1.5 * a * b**2
        expected[32] = 3.0 * a**2 * b + 0.75 * b**3
        expected[64] = 1.5 * a * b**2
        expected[96] = 0.25 * b**3
    else:
        expected[0] = a**4 + 3.0 * a**2 * b**2 + 0.375 * b**4
        expected[32] = 4.0 * a**3 * b + 3.0 * a * b**3
        expected[64] = 3.0 * a**2 * b**2 + 0.5 * b**4
        expected[96] = a * b**3
        expected[128] = 0.125 * b**4
    f = tech_cls.from_coeffs(jnp.asarray(coeffs))
    result = f**power
    assert type(result) is tech_cls
    assert result.n == len(expected)
    assert result.coeffs.shape == expected.shape
    error = float(np.max(np.abs(np.asarray(result.coeffs) - expected)))
    assert error < 20.0 * np.finfo(np.float64).eps * np.max(np.abs(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), coeffs)
