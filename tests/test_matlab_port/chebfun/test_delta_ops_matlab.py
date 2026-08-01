"""Unified Dirac-delta semantics on Chebfun operations.

MATLAB reference: @deltafun arithmetic and convolution semantics
(@deltafun/plus.m, mtimes.m, conv.m) -- deltas merge under +/-, scale
under scalar *, convolve exactly (delta_u * delta_v = delta_{u+v},
delta_u * g = g(x-u)), and integrate to their magnitudes via sum().
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


def _delta_train(domain, deltas):
    z = cj.chebfun(lambda t: 0.0 * t, domain=list(domain))
    return cj.Chebfun(funs=z.funs, domain=z.domain, deltas=tuple(deltas))


def test_sum_dirac_is_one():
    d = cj.chebfun(lambda t: t, domain=[-1, 1]).dirac()
    assert float(d.sum()) == pytest.approx(1.0, abs=1e-13)


def test_dirac_weights_inverse_slope():
    # dirac(sin) on [2, 8]: root at pi with weight 1/|cos(pi)| = 1.
    d = cj.chebfun(lambda t: jnp.sin(t), domain=[2, 8]).dirac()
    locs = [loc for loc, _ in d.deltas]
    assert locs == pytest.approx([np.pi, 2 * np.pi], abs=1e-12)
    assert float(d.sum()) == pytest.approx(2.0, abs=1e-12)


def test_arithmetic_merges_and_scales():
    d = _delta_train((-1, 1), [(0.0, 1.0)])
    e = d * 2.0 + d
    assert e.deltas == ((0.0, 3.0),)
    assert (-d).deltas == ((0.0, -1.0),)
    f = d - d
    assert f.deltas == ()


def test_conv_delta_trains_binomial():
    coin = _delta_train((-0.5, 1.5), [(0.0, 0.5), (1.0, 0.5)])
    two = coin.conv(coin)
    assert two.deltas == ((0.0, 0.25), (1.0, 0.5), (2.0, 0.25))
    assert float(two.sum()) == pytest.approx(1.0, abs=1e-13)


def test_conv_delta_with_smooth_is_shifted_copy():
    g = cj.chebfun(lambda t: jnp.exp(-t ** 2), domain=[-1, 1])
    d = _delta_train((-0.25, 0.25), [(0.0, 1.0)])
    h = d.conv(g)
    xs = np.linspace(-0.7, 0.7, 13)
    np.testing.assert_allclose(np.asarray(h(jnp.asarray(xs))),
                               np.exp(-xs ** 2), atol=1e-12)


def test_smooth_conv_unaffected():
    f = cj.chebfun(lambda t: jnp.exp(-t ** 2), domain=[-1, 1])
    h = f.conv(f)
    # (e^{-t^2} * e^{-t^2})(0) = sqrt(pi/2)*erf(sqrt(2))
    from scipy.special import erf
    want = np.sqrt(np.pi / 2) * erf(np.sqrt(2.0))
    assert float(np.asarray(h(jnp.asarray([0.0])))[0]) == pytest.approx(
        want, rel=1e-10)


def test_pointwise_product_scales_deltas():
    # delta_{0.5} * g = g(0.5) * delta_{0.5} (@deltafun/times.m).
    g = cj.chebfun(lambda t: 2.0 + t, domain=[-1, 1])
    d = _delta_train((-1, 1), [(0.5, 1.0)])
    h = d * g
    (loc, mag), = h.deltas
    assert loc == pytest.approx(0.5) and mag == pytest.approx(2.5)
    assert float(h.sum()) == pytest.approx(2.5, abs=1e-12)


def test_delta_aware_extrema_and_norms():
    # @deltafun: positive deltas -> max Inf; negative -> min -Inf;
    # 1-norm adds |mass|; other norms infinite (calc/DeltaDerivs).
    d = _delta_train((-1, 1), [(0.0, 2.0), (0.5, -1.0)])
    g = cj.chebfun(lambda t: 0 * t + 1.0)
    f = d + g
    assert f.max()[1] == float("inf")
    assert f.min()[1] == float("-inf")
    assert float(f.norm(1)) == pytest.approx(2.0 + 1.0 + 2.0, abs=1e-12)
    assert float(f.norm(2)) == float("inf")
