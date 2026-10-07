"""Exceptional reciprocal sampling semantics, not arbitrary nonfinite addition.

Provenance
----------
MATLAB source : @separableApprox/cdr.m, @separableApprox/vscale.m,
    @separableApprox/iszero.m, @spherefun/sample.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Expected samples follow constant factors and literal Inf-only reciprocal mask.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._plus import _iszero, _scale
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def field(pivots, column=1.0):
    c = Trigtech.from_coeffs(jnp.asarray([column]), is_real=True)
    r = Trigtech.from_coeffs(jnp.asarray([1.0]), is_real=True)
    return Spherefun(cols=[c] * len(pivots), rows=[r] * len(pivots),
                     pivots=jnp.asarray(pivots), idx_plus=tuple(range(len(pivots))),
                     idx_minus=())


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('pivots,expected', [
    ([1e-320], 0.0),
    ([1e-320, 2.0], 0.5),
    ([0.0, 2.0], 0.5),
    ([float('nan')], float('nan')),
], ids=['tiny', 'tiny-and-finite', 'zero-and-finite', 'nan-retained'])
def test_sampled_scale_source_inverse_mask(disable, pivots, expected):
    f = field(pivots)
    with jax.disable_jit(disable):
        actual = _scale(f)
    if np.isnan(expected):
        assert bool(jnp.isnan(actual))
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(f.pivots, np.asarray(pivots))


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('pivot,column,expected', [
    (1e-320, 1.0, False),
    (1e-320, 0.0, True),
    (0.0, 1.0, False),
    (float('nan'), 1.0, False),
    (float('nan'), 0.0, True),
], ids=['tiny-nonzero-slice', 'tiny-zero-slice', 'zero-nonzero-slice',
        'nan-nonzero-slice', 'nan-zero-slice'])
def test_source_iszero_retains_raw_check_and_slice_fallback(disable, pivot, column, expected):
    # Source raw initial reciprocal norm is NOT the masked CDR norm.
    # When all sampled values are zero/NaN, it still examines factor slices.
    # Thus zero-valued sampled CDR with nonzero slices need not be iszero=True.
    f = field([pivot], column)
    with jax.disable_jit(disable):
        assert _iszero(f) is expected
