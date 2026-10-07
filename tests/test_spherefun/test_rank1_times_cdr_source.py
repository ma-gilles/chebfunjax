"""Source rank-one CDR scaling, including zero and overflow reciprocals.

Provenance
----------
MATLAB source : @spherefun/times.m, @separableApprox/cdr.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('pivot,expected_inverse', [
    (4.0, 0.25), (-4.0, -0.25), (0.0, 0.0),
    (np.nextafter(0.0, 1.0), 0.0), (np.inf, 0.0), (2j, -0.5j),
], ids=['positive', 'negative', 'zero', 'tiny-overflow', 'infinite', 'complex'])
def test_source_cdr_factor_scaling_and_right_metadata(pivot, expected_inverse, disable):
    def tech(value):
        return Trigtech.from_coeffs(jnp.asarray([value]), is_real=True)

    f = Spherefun(cols=[tech(2.0)], rows=[tech(1.0)],
                  pivots=jnp.asarray([pivot]), idx_plus=(0,), idx_minus=(),
                  nonzero_poles=True, pivot_locations=((0.1, 0.2),))
    g = Spherefun(cols=[tech(3.0)], rows=[tech(1.0)],
                  pivots=jnp.asarray([5.0]), idx_plus=(0,), idx_minus=(),
                  nonzero_poles=True, pivot_locations=((0.3, 0.4),))
    with jax.disable_jit(disable):
        h = f*g
        value = h(jnp.asarray(0.7), jnp.asarray(1.1))
    np.testing.assert_array_equal(h.pivots, g.pivots)
    assert h.pivot_locations == g.pivot_locations
    assert h.nonzero_poles
    np.testing.assert_allclose(h.cols[0].coeffs, [6*np.sqrt(abs(expected_inverse))],
                               rtol=0, atol=2e-15)
    expected_row = 0 if expected_inverse == 0 else expected_inverse/abs(expected_inverse)*np.sqrt(abs(expected_inverse))
    np.testing.assert_allclose(h.rows[0].coeffs, [expected_row], rtol=0, atol=2e-15)
    np.testing.assert_allclose(value, 6*expected_inverse/5, rtol=0, atol=2e-15)
