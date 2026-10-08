"""Source RHW capacity/stopping; lagpts.m407-505, Chebfun7574c77."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import laguerre_rh_general as rh
from chebfunjax.utils.quadrature import lagpts


@pytest.mark.parametrize('n', [3000, 10000, 100000])
def test_source_truncated_initial_capacity(n):
    x, _, igatt = rh._rh_initial_guesses_general(n, .3, True)
    capacity = min(n, math.ceil(17 * math.sqrt(n)))
    assert x.size == capacity
    assert igatt == math.floor(capacity + 1.31 * n**.4 - n + .5)


@pytest.mark.parametrize('alpha', [-.5, 0., .3, .5])
def test_raw_compressed_rule_retains_full_rule_prefix(alpha):
    x, w, length = rh._laguerre_rh_general(3000, alpha, comp_repr=True)
    xf, wf = rh._laguerre_rh_general(3000, alpha)
    length = int(length)
    assert 0 < length <= math.ceil(17 * math.sqrt(3000))
    assert x.size == math.ceil(17 * math.sqrt(3000))
    np.testing.assert_allclose(x[:length], xf[:length], rtol=0, atol=8e-10)
    np.testing.assert_allclose(w[:length], wf[:length], rtol=4e-9, atol=0)
    np.testing.assert_allclose([np.asarray(w[:length]) @ np.asarray(x[:length])**k for k in range(5)],
                              [math.gamma(alpha+k+1) for k in range(5)], rtol=2e-9, atol=0)


def test_first_zero_after_positive_is_excluded(monkeypatch):
    monkeypatch.setattr(rh, '_rh_initial_guesses_general',
                        lambda n, a, c: (jnp.arange(1., 6.), 0, 0))
    monkeypatch.setattr(rh, '_rh_factors_general', lambda n, a: (1., -1.))
    monkeypatch.setattr(rh, '_poly_asy_rh_general',
                        lambda n, x, a, t: jnp.asarray(0. if n == 3000 else 1.))
    monkeypatch.setattr(rh, '_source_rh_weight',
                        lambda x, f, l, r: jnp.where(x < 2, 1., 0.))
    with jax.disable_jit():
        x, w, length = rh._laguerre_rh_general(3000, .3, comp_repr=True)
    assert int(length) == 1
    np.testing.assert_array_equal(w, [1., 0., 0., 0., 0.])
    np.testing.assert_array_equal(x, [1., 2., 3., 4., 5.])


def test_public_barycentric_and_interval_order():
    x, w, v = map(np.asarray, lagpts(3000, .3, method='RHW', bary=True))
    xm, wm, vm = map(np.asarray, lagpts(3000, .3, (2., np.inf), method='RHW', bary=True))
    np.testing.assert_allclose(w.sum(), math.gamma(1.3), rtol=2e-14)
    np.testing.assert_allclose(xm, x+2, rtol=0, atol=2e-13)
    np.testing.assert_allclose(wm, w*math.exp(-2), rtol=2e-14, atol=0)
    np.testing.assert_array_equal(vm, v)
    assert len(x) <= math.ceil(17*math.sqrt(3000))
    assert np.isfinite(v).all() and np.max(np.abs(v)) == 1


@pytest.mark.parametrize('n', [10000, 100000])
def test_large_degree_compressed_capacity_and_moments(n):
    x, w, v = map(np.asarray, lagpts(n, .3, method='RHW', bary=True))
    assert 0 < len(x) <= math.ceil(17*math.sqrt(n)) < n
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(np.diff(x)>0) and np.all(w>=0)
    np.testing.assert_allclose([w@x**k for k in range(5)],
                              [math.gamma(1.3+k) for k in range(5)], rtol=2e-9, atol=0)
