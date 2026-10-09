"""Analytic legacy hist controls from installed MATLAB source.

Provenance
----------
MATLAB source : R2017a toolbox/matlab/datafun/hist.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax
import jax.numpy as jnp
import numpy as np  # uses-numpy: independent source arithmetic/counting test oracle
import pytest

from chebfunjax.utils.matlab_hist import matlab_hist_counts


@pytest.mark.parametrize('disabled', [False, True])
def test_scalar_bins_centers_and_midpoint_tie(disabled):
    with jax.disable_jit(disabled):
        counts, centers = matlab_hist_counts(jnp.asarray([0., 1., 2.]), 2)
    np.testing.assert_array_equal(centers, [.5, 1.5])
    np.testing.assert_array_equal(counts, [2, 1])


def test_negative_midpoint_and_adjacent_value():
    x = jnp.asarray([-2., -1., jnp.nextafter(-1., jnp.inf), 0.])
    counts, centers = matlab_hist_counts(x, 2)
    np.testing.assert_array_equal(centers, [-1.5, -.5])
    # eps(-1)=2**-52: the one-nextafter value is still below shifted edge.
    np.testing.assert_array_equal(counts, [3, 1])


def test_constant_values_and_single_bin():
    counts, centers = matlab_hist_counts(jnp.ones(4)*3, 3)
    np.testing.assert_array_equal(centers, [2., 3., 4.])
    np.testing.assert_array_equal(counts, [0, 4, 0])
    counts, centers = matlab_hist_counts(jnp.asarray([0., 2., 4.]), 1)
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_array_equal(centers, [2.])


@pytest.mark.parametrize('values', [[], [jnp.nan], [jnp.inf], [1j]])
def test_unsupported_data_rejected(values):
    with pytest.raises(ValueError):
        matlab_hist_counts(values, 2)


@pytest.mark.parametrize('disabled', [False, True])
def test_zero_edge_signed_zero_and_subnormal_data(disabled):
    # Source shifted interior is +minimum-subnormal, so zero is left and
    # positive minimum-subnormal itself is right (histc left-closed rule).
    words = np.array([0xbff0000000000000, 0x8000000000000001,
                      0x8000000000000000, 0, 1, 0x3ff0000000000000],
                     dtype=np.uint64)
    data = jax.lax.bitcast_convert_type(jnp.asarray(words), jnp.float64)
    with jax.disable_jit(disabled):
        counts, centers = matlab_hist_counts(data, 2)
    np.testing.assert_array_equal(counts, [4, 2])
    np.testing.assert_array_equal(centers, [-.5, .5])


def test_literal_symmetric_zero_midpoint():
    counts, centers = matlab_hist_counts(jnp.array([-1., 0., 1.]), 2)
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_array_equal(centers, [-.5, .5])


@pytest.mark.parametrize('compiled', [False, True])
def test_edge_plus_eps_binary64_boundary_table(compiled):
    from chebfunjax.utils.matlab_hist import _shifted_edge_keys
    # Literal IEEE source addition cases, not answers copied from implementation.
    # Columns: edge words, RN(edge+eps(abs(edge))) words.
    pairs = np.array([
        [0, 1], [0x8000000000000000, 1], [1, 2],
        [0x8000000000000001, 0],
        [0x000fffffffffffff, 0x0010000000000000],
        [0x0010000000000000, 0x0010000000000001],
        [0x8010000000000000, 0x800fffffffffffff],
        [0xbff0000000000000, 0xbfeffffffffffffe],
        [0xbfe0000000000000, 0xbfdffffffffffffe],
        [0x3ff0000000000000, 0x3ff0000000000001],
    ], dtype=np.uint64)
    values = jax.lax.bitcast_convert_type(jnp.asarray(pairs[:, 0]), jnp.float64)
    fn = jax.jit(_shifted_edge_keys) if compiled else _shifted_edge_keys
    # Independent Python integer total-order construction from literal outputs.
    expected = np.array([(~int(w) & ((1 << 64)-1)) if int(w) >> 63
                         else int(w) | (1 << 63) for w in pairs[:, 1]], dtype=np.uint64)
    np.testing.assert_array_equal(fn(values), expected)


def _source_scalar_bins_oracle(values, bins, *, version):
    # Independent scalar Python binary64 arithmetic; no production helper use.
    import bisect
    import math
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= bins // 2 + .5
        hi += (bins + 1) // 2 - .5
    delta = hi - lo
    if version == 2013:
        width = delta / bins
        edges = [lo + width*k for k in range(bins+1)]
        edges[-1] = hi
    else:
        edges = [lo + (k*delta)/bins for k in range(bins+1)]
        edges[0], edges[-1] = lo, hi
        width = edges[1] - edges[0]
    centers = [v + width/2 for v in edges[:-1]]
    shifted = [e + math.ulp(abs(e)) for e in edges[1:-1]]
    counts = [0]*bins
    for value in values:
        counts[bisect.bisect_right(shifted, value)] += 1
    return counts, centers


def test_source_36_bins_edges_centers_and_last_bin():
    # Exactly representable half-unit grid, deliberately asymmetric domain.
    values = [-7.25 + .5*k for k in range(37)]
    expected_centers = [-7.0 + .5*k for k in range(36)]
    expected_counts = [2] + [1]*35
    counts, centers = matlab_hist_counts(jnp.asarray(values), 36)
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_array_equal(counts, expected_counts)
    assert int(jnp.sum(counts)) == len(values)
    # Both historical formulas coincide for this exactly representable grid.
    for version in (2013, 2017):
        oracle_counts, oracle_centers = _source_scalar_bins_oracle(values, 36, version=version)
        assert oracle_counts == expected_counts
        assert oracle_centers == expected_centers


def test_r2017_asymmetric_grid_rounding_and_adjacent_ties():
    import math
    lo, hi, bins = -.73, 1.91, 36
    edges = [lo + (k*(hi-lo))/bins for k in range(bins+1)]
    values = [lo, hi]
    for edge in edges[1:-1]:
        values.extend([math.nextafter(edge, -math.inf), edge,
                       math.nextafter(edge, math.inf),
                       edge + math.ulp(abs(edge))])
    expected_counts, expected_centers = _source_scalar_bins_oracle(values, bins, version=2017)
    counts, centers = matlab_hist_counts(jnp.asarray(values), bins)
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize('bins', [0, -1, True, 2.5, [.5, 1.5]])
def test_outside_scalar_bin_contract_rejected(bins):
    # Vector-center MATLAB overload is explicitly outside this page adapter.
    with pytest.raises((ValueError, TypeError)):
        matlab_hist_counts(jnp.asarray([0., 1., 2.]), bins)


def test_constant_even_bins_and_endpoint_assignment():
    counts, centers = matlab_hist_counts(jnp.full((5,), 3.), 4)
    np.testing.assert_array_equal(centers, [1., 2., 3., 4.])
    np.testing.assert_array_equal(counts, [0, 0, 5, 0])


@pytest.mark.parametrize('left,right,count', [(-1.e308, 1.e308, 4),
                                               (1.e308, 1.1e308, 36)])
def test_source_linspace_overflow_branches(left, right, count):
    import math

    from chebfunjax.utils.matlab_hist import _source_linspace
    delta = right-left
    if math.isinf(delta):
        expected = [left+(right/count)*k-(left/count)*k for k in range(count+1)]
    else:
        assert math.isinf(delta*(count-1))
        expected = [left+k*(delta/count) for k in range(count+1)]
    expected[0], expected[-1] = left, right
    actual = _source_linspace(jnp.asarray(left), jnp.asarray(right), count)
    np.testing.assert_array_equal(actual, expected)
    assert bool(jnp.all(jnp.isfinite(actual)))
