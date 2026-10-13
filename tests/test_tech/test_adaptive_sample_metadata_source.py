"""Source nonfinite metadata used before/after extrapolation.

Chebfun 7574c77, @chebtech/populate.m: finite-only column scales,
whole-row NaN/Inf flags, then NaN followed by Inf restoration.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import (
    _adaptive_sample_metadata,
    _restore_refinement_sentinels,
)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
def test_mixed_nonfinite_rows_keep_finite_column_scales_and_inf_priority(dtype):
    sampled = jnp.asarray([[3., 4.], [jnp.nan, 6.], [7., jnp.inf],
                           [jnp.nan, -jnp.inf]], dtype=dtype)
    scales, bad_nan, bad_inf, finite = _adaptive_sample_metadata(
        sampled, jnp.asarray([2., 8.]))
    np.testing.assert_array_equal(scales, [7., 8.])
    np.testing.assert_array_equal(bad_nan, [[False], [True], [False], [True]])
    np.testing.assert_array_equal(bad_inf, [[False], [False], [True], [True]])
    assert not bool(finite)
    restored = np.asarray(_restore_refinement_sentinels(
        jnp.ones_like(sampled), bad_nan, bad_inf))
    np.testing.assert_array_equal(restored[0], [1., 1.])
    assert np.isnan(restored[1]).all()
    assert np.isposinf(restored[2:].real).all()
    if np.iscomplexobj(restored):
        np.testing.assert_array_equal(restored.imag, np.zeros((4, 2)))


def test_complex_magnitudes_and_finite_sample_bits():
    sampled = jnp.asarray([3.+4j, -5.+12j, complex(-0., -0.)])
    scales, nan, inf, finite = _adaptive_sample_metadata(sampled, jnp.asarray(2.))
    assert float(scales) == 13.
    assert bool(finite)
    restored = _restore_refinement_sentinels(sampled, nan, inf)
    assert np.asarray(restored).tobytes() == np.asarray(sampled).tobytes()
