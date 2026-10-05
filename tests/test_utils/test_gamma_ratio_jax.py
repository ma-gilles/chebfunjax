"""Independent high-precision values and JIT checks for gamma ratios.

Provenance
----------
MATLAB source : gammaratio.m
Chebfun commit: 7574c77
Oracles: mpmath 80-digit gamma evaluations; inputs use the shown binary64
values, not rounded decimal sums. Constants are rounded only at comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils import gammaratio

EPS = np.finfo(float).eps


@pytest.mark.parametrize('m,delta,expected', [
    (16.0, 0.5, 3.9688767938289397464678926660568045),
    (1e8, 0.5, 9999.999987500000007812500048828125),
    (50.0, -0.5, 0.1424931821187255217273898519739587),
    (20.0, 3.25, 20152.4290450933927966631960138706),
    (10.0, 0.3, 1.9742909200352050991792459026127745),
    (1.5, 2.5, 6.770275002573075443376953418729271),
    (16.0, -1.0, 0.06666666666666666666666666666666667),
    (1e8, -0.5, 0.0001000000003750000019531250102539063),
])
def test_gamma_ratio_high_precision_oracle(m, delta, expected):
    result = jax.jit(gammaratio)(m, delta)
    assert result.shape == ()
    npt.assert_allclose(result, expected, rtol=64 * EPS, atol=0)


@pytest.mark.parametrize('delta', [0, 1, 2, 3, 5])
def test_exact_integer_ratios_with_dynamic_vmap(delta):
    m = jnp.array([16.0, 50.0, 1000.0])
    expected = np.ones(3)
    for k in range(delta):
        expected *= np.asarray(m) + k
    result = jax.jit(jax.vmap(gammaratio, in_axes=(0, None)))(m, float(delta))
    npt.assert_allclose(result, expected, rtol=16 * EPS, atol=0)


def test_gamma_ratio_rejects_nonscalar_inputs():
    with pytest.raises(ValueError, match='scalar'):
        gammaratio(jnp.ones(2), 0.5)
