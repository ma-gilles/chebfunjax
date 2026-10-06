"""Independent real atanh contracts supplement unchanged source trig bounds.

Provenance
----------
MATLAB source : @chebfun/atanh.m, @chebfun/acoth.m; tests/chebfun/test_trig.m
Chebfun commit: 7574c77
Primitive identities are a numerical adaptation, not MATLAB builtin internals.
"""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from chebfunjax.utils.elementary import _atanh_log1p_real


@pytest.mark.parametrize("compiled", [False, True])
def test_real_atanh_against_high_precision(compiled):
    rng = np.random.default_rng(7681)
    values = np.concatenate(
        (
            rng.uniform(-1, 1, 128),
            np.array(
                [
                    0.4117737317053608,
                    -0.4117737317053608,
                    0.5,
                    np.nextafter(0.5, 0),
                    np.nextafter(0.5, 1),
                    np.nextafter(1.0, 0),
                    np.nextafter(-1.0, 0),
                    2.0**-28,
                    np.nextafter(2.0**-28, 0),
                ]
            ),
        )
    )
    with mp.workdps(100):
        expected = np.array([float(mp.atanh(mp.mpf(float(x)))) for x in values])
    fn = jax.jit(_atanh_log1p_real) if compiled else _atanh_log1p_real
    actual = np.asarray(fn(jnp.asarray(values)))
    # Primitive bound is independent; unchanged Chebfun source assertion
    # remains the final contract. Do not claim universally correct rounding.
    assert np.all(np.abs(actual - expected) <= 2 * np.abs(np.spacing(expected)))


def test_tiny_values_and_poles():
    bits = np.array([0, 1, (1 << 52) - 1, 1 << 63, (1 << 63) | 1], dtype=np.uint64)
    for fn in (_atanh_log1p_real, jax.jit(_atanh_log1p_real)):
        got = np.asarray(fn(jnp.asarray(bits.view(np.float64))))
        np.testing.assert_array_equal(got.view(np.uint64), bits)
        np.testing.assert_array_equal(np.asarray(fn(jnp.asarray([-1.0, 1.0]))), [-np.inf, np.inf])
        assert bool(jnp.all(jnp.isnan(fn(jnp.asarray([-2.0, 2.0])))))
