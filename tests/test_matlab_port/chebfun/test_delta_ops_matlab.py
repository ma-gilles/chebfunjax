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


# ---------------------------------------------------------------------------
# Derivative-order deltas (MATLAB dirac(f, n) = diff(dirac(f), n);
# @deltafun deltaMag rows carry the distributional-derivative order).
# ---------------------------------------------------------------------------


def test_dirac_derivative_order():
    x = cj.chebfun(lambda t: t, domain=[-1, 1])
    dp = x.dirac(1)
    assert dp.deltas == ((0.0, 1.0, 1),)
    assert x.dirac(2).deltas == ((0.0, 1.0, 2),)
    # diff promotes the order.
    assert x.dirac().diff().deltas == ((0.0, 1.0, 1),)


def test_delta_derivative_integrates_to_zero():
    x = cj.chebfun(lambda t: t, domain=[-1, 1])
    dp = x.dirac(1)
    assert float(dp.sum()) == pytest.approx(0.0, abs=1e-14)
    # cumsum lowers the order: int delta' = delta.
    cd = dp.cumsum()
    assert cd.deltas == ((0.0, 1.0),)
    assert float(cd.sum()) == pytest.approx(1.0, abs=1e-12)


def test_delta_derivative_product_leibniz():
    # g(x) delta'(x) = g(0) delta' - g'(0) delta  (funTimesDelta).
    x = cj.chebfun(lambda t: t, domain=[-1, 1])
    g = cj.chebfun(lambda t: jnp.exp(t), domain=[-1, 1])
    h = g * x.dirac(1)
    rows = {(_r[0], _r[2] if len(_r) > 2 else 0): float(_r[1])
            for _r in h.deltas}
    assert rows[(0.0, 1)] == pytest.approx(1.0, rel=1e-12)
    assert rows[(0.0, 0)] == pytest.approx(-1.0, rel=1e-12)
    # int g delta' = -g'(0) = -1.
    assert float(h.sum()) == pytest.approx(-1.0, rel=1e-12)


def test_delta_derivative_conv_differentiates():
    # (delta'_0 * g)(x) = g'(x)  (@deltafun/conv).
    x = cj.chebfun(lambda t: t, domain=[-1, 1])
    g = cj.chebfun(lambda t: jnp.exp(t), domain=[-1, 1])
    c = x.dirac(1).conv(g)
    assert float(np.asarray(c(jnp.asarray([0.5])))[0]) == pytest.approx(
        float(np.exp(0.5)), rel=1e-12)


def test_delta_derivative_unbounded_extrema():
    # A derivative row is unbounded both ways; every norm is infinite.
    x = cj.chebfun(lambda t: t, domain=[-1, 1])
    dp = x.dirac(1)
    assert dp.max()[1] == float("inf")
    assert dp.min()[1] == float("-inf")
    assert float(dp.norm(1)) == float("inf")
