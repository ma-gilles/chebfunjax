"""RECW source recurrence and binary64 stopping controls.

Pinned lagpts.m195-255 at7574c77. The independent host recurrence is source
emulation, not a fresh MATLAB execution. Public barycentric length uses the
established RHW x.size adaptation to the source truncated-array dimension bug.
"""

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.laguerre_rec import _lag_rec, _recw_raw_weight, _recw_weight_is_zero
from chebfunjax.utils.quadrature import lagpts

EPS = np.finfo(np.float64).eps


@lru_cache(maxsize=None)
def _literal_scalar_recurrence(n, alpha):
    """Independent scalar transcription; raw weights include the first zero."""
    x, w = [], []
    if alpha == -0.5:
        ratio = math.sqrt(math.pi) * math.prod((k - .5) / k for k in range(1, n + 1))
    else:
        ratio = 1.0 if alpha == 0 else math.gamma(n + 1 + alpha) / math.gamma(n + 1)

    def recurrence(z):
        p1, p2 = np.float64(1.0), np.float64(0.0)
        for j in range(n):
            p3, p2 = p2, p1
            p1 = ((-z + alpha + 2 * j + 1) * p2 - (alpha + j) * p3) / (j + 1)
        return p1, (n * p1 - (n + alpha) * p2) / z

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for i in range(n):
            if i == 0:
                z = (1 + alpha) * (3 + .92 * alpha) / (1 + 2.4 * n + 1.8 * alpha)
            elif i == 1:
                z = x[0] + (15 + 6.25 * alpha) / (1 + .9 * alpha + 2.5 * n)
            else:
                ai = i - 1
                z = x[i - 1] + ((1 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alpha / (1 + 3.5 * ai)) * (x[i - 1] - x[i - 2]) / (1 + .3 * alpha)
            z = np.float64(z)
            for _ in range(10):
                p1, pp = recurrence(z)
                old = z
                z = old - p1 / pp
                if abs(z - old) < 1e-15:
                    break
            _, pp = recurrence(z)
            # Parentheses and explicit products retain source staged overflow.
            weight = np.float64(ratio) / (z * np.multiply(pp, pp))
            x.append(z)
            w.append(weight)
            if weight == 0:
                break
    return np.asarray(x), np.asarray(w)


@pytest.mark.parametrize('n', [400, 1000])
def test_high_order_source_prefix_includes_first_zero(n):
    x, w, length = _lag_rec(n, flag=True)
    length = int(length)
    x, w = np.asarray(x), np.asarray(w)
    expected_x, expected_w = _literal_scalar_recurrence(n, 0.0)
    assert length == expected_x.size < n
    assert np.isfinite(x[:length]).all() and np.isfinite(w[:length]).all()
    npt.assert_allclose(x[:length], expected_x, atol=1000 * EPS, rtol=100 * EPS)
    # Forward recurrence conditioning grows with degree; no absolute floor
    # hides the subnormal weights or the first-zero stopping decision.
    npt.assert_allclose(w[:length], expected_w, rtol=64 * n * EPS, atol=0)
    bits = w[:length].view(np.uint64)
    assert np.all(bits[:-1] != 0) and bits[-1] == 0
    # These alpha=0 source rules jump from normal weight to overflow zero.
    # Compare the observed pattern; do not assume a subnormal must occur.
    reference_bits = expected_w.view(np.uint64)
    npt.assert_array_equal((bits != 0) & (bits < np.uint64(1 << 52)),
                           (reference_bits != 0) & (reference_bits < np.uint64(1 << 52)))
    assert np.all(x[length:] == 0) and np.all(w[length:].view(np.uint64) == 0)
    full_x, _ = _lag_rec(n)
    npt.assert_allclose(np.asarray(full_x)[:length], x[:length], atol=1000 * EPS, rtol=100 * EPS)
    actual_x, actual_w, bary = map(np.asarray, lagpts(n, method='RECW', bary=True))
    assert actual_x.size == actual_w.size == bary.size == length
    npt.assert_array_equal(actual_x, x[:length])
    expected_normalized = expected_w / np.sum(expected_w)
    npt.assert_allclose(actual_w, expected_normalized, rtol=64 * n * EPS, atol=0)
    assert actual_w[-1] == 0 and bary[-1] == 0
    assert np.all(np.isfinite(bary))
    magnitude = np.sqrt(actual_w * actual_x)
    expected_bary = (-1.0)**np.arange(length) * magnitude / np.max(magnitude)
    npt.assert_allclose(bary, expected_bary, rtol=100 * EPS, atol=0)
    # Mapping is after barycentric construction, including the trailing zero.
    for interval in [(1.0, np.inf), (-np.inf, -1.0)]:
        mx, mw, mv = map(np.asarray, lagpts(n, interval=interval, method='RECW', bary=True))
        npt.assert_array_equal(mv, bary)
        npt.assert_array_equal(mx, actual_x + 1 if np.isinf(interval[1]) else -actual_x - 1)
        npt.assert_allclose(mw, actual_w * math.exp(-1), rtol=100 * EPS, atol=0)


@pytest.mark.parametrize('alpha', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 4, 17])
def test_small_rule_matches_independent_source_recurrence(n, alpha):
    x, w, length = _lag_rec(n, alpha, flag=True)
    assert int(length) == n
    ex, ew = _literal_scalar_recurrence(n, alpha)
    npt.assert_allclose(x, ex, atol=1000 * EPS, rtol=100 * EPS)
    npt.assert_allclose(w, ew, atol=1000 * EPS * math.gamma(alpha + 1), rtol=0)


@pytest.mark.parametrize('alpha', [0.0, 0.5])
def test_recurrence_flag_with_jit_disabled(alpha):
    with jax.disable_jit():
        x, w, length = _lag_rec(4, alpha, flag=True)
    ex, ew = _literal_scalar_recurrence(4, alpha)
    assert int(length) == 4
    npt.assert_allclose(x, ex, atol=1000 * EPS, rtol=100 * EPS)
    npt.assert_allclose(w, ew, atol=1000 * EPS, rtol=0)


@pytest.mark.parametrize('disabled', [True, False])
def test_exact_zero_does_not_flush_subnormal_or_stop_on_nan(disabled):
    bits = np.asarray([0, 1 << 63, 1, (1 << 63) + 1, (1 << 52) - 1,
                       1 << 52, 0x7ff0000000000000, 0x7ff8000000000000], dtype=np.uint64)
    with jax.disable_jit(disabled):
        values = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.float64)
        zero = jax.jit(_recw_weight_is_zero)(values)
    npt.assert_array_equal(zero, [True, True, False, False, False, False, False, False])


@pytest.mark.parametrize('disabled', [True, False])
def test_raw_weight_stages_preserve_subnormal_and_square_overflow(disabled):
    ratio = np.asarray([1., 2., 1., 1., 1., 1., 0.])
    z = np.asarray([2., 2., 4., 1e-200, 1., 0., 1.])
    pp = np.asarray([2.**511, 2.**511, 2.**511, 1e200, 0., np.inf, 1.])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        expected = ratio / (z * (pp * pp))
    with jax.disable_jit(disabled):
        got = np.asarray(jax.jit(_recw_raw_weight)(jnp.asarray(ratio), jnp.asarray(z), jnp.asarray(pp)))
    good = ~np.isnan(expected)
    npt.assert_array_equal(got[good].view(np.uint64), expected[good].view(np.uint64))
    assert np.isnan(got[~good]).all()
    assert 0 < got[0] < np.finfo(float).tiny
    assert got[2] == got[3] == 0  # Source overflow, not a log-weight substitute.


def test_zero_order_flag_has_zero_length():
    x, w, length = _lag_rec(0, flag=True)
    assert x.size == w.size == int(length) == 0


def test_fixed_capacity_kernel_accepts_dynamic_alpha():
    operation = jax.jit(lambda alpha: _lag_rec(17, alpha, flag=True))
    for alpha in [0.0, 0.5]:
        x, w, length = operation(alpha)
        assert x.shape == w.shape == (17,) and int(length) == 17
        ex, ew = _literal_scalar_recurrence(17, alpha)
        npt.assert_allclose(x, ex, atol=1000 * EPS, rtol=100 * EPS)
        npt.assert_allclose(w, ew, atol=1000 * EPS, rtol=0)


def test_public_variable_length_requires_eager_execution():
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda alpha: lagpts(17, alpha, method='RECW'))(0.0)


def test_demonstrated_subnormal_prefix_negative_half_order1000():
    # Independent scalar source probe found exactly two subnormals here;
    # n400/alpha=-.5 has none, so it is not asserted to exercise this case.
    n, alpha = 1000, -0.5
    ex, ew = _literal_scalar_recurrence(n, alpha)
    reference_bits = ew.view(np.uint64)
    subnormal = (reference_bits != 0) & (reference_bits < np.uint64(1 << 52))
    assert ex.size == 522 and int(np.sum(subnormal)) == 2
    x, w, length = _lag_rec(n, alpha, flag=True)
    length = int(length)
    x, w = np.asarray(x), np.asarray(w)
    assert length == ex.size
    npt.assert_allclose(x[:length], ex, atol=1000 * EPS, rtol=100 * EPS)
    npt.assert_allclose(w[:length], ew, rtol=64 * n * EPS, atol=0)
    bits = w[:length].view(np.uint64)
    assert np.all(bits[:-1] != 0) and bits[-1] == 0
    npt.assert_array_equal((bits != 0) & (bits < np.uint64(1 << 52)), subnormal)
    px, pw, pv = map(np.asarray, lagpts(n, alpha, method='RECW', bary=True))
    npt.assert_array_equal(px, x[:length])
    npt.assert_allclose(pw, ew * (math.sqrt(math.pi) / np.sum(ew)), rtol=64 * n * EPS, atol=0)
    magnitude = np.sqrt(pw * px)
    expected_bary = (-1.0)**np.arange(length) * magnitude / np.max(magnitude)
    npt.assert_allclose(pv, expected_bary, rtol=100 * EPS, atol=0)
    assert np.all(pv[subnormal] != 0) and pv[-1] == 0
    for shift in [1.0, 1000.0]:
        mx, mw, mv = map(np.asarray, lagpts(n, alpha, (shift, np.inf), method='RECW', bary=True))
        assert mx.size == length
        npt.assert_array_equal(mx, px + shift)
        npt.assert_array_equal(mv, pv)
        npt.assert_allclose(mw, pw * math.exp(-shift), rtol=100 * EPS, atol=0)
