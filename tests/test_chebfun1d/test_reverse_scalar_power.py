"""Bounded scalar reverse-power adapters beyond the source power tests.

Provenance
----------
MATLAB source : @chebfun/power.m, @chebtech/power.m
Chebfun commit: 7574c77
Singular/unbounded/trig exponent pieces, array bases and quasimatrix dispatch
remain unqualified. The zero-base control has a strictly positive exponent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 100))

def _source_normest(f):
    """Bounded Chebfun source normest: sum onefun estimates by piece.

    Pinned @chebfun/normest.m sums piece normest values; @chebtech/normest.m
    returns max(vscale(tech)). This helper intentionally covers only the
    bounded Chebtech1/2 pieces admitted by the candidate method.
    """
    return sum(float(piece.tech.vscale) for piece in f.funs)

def test_negative_real_base_uses_principal_complex_branch():
    f = cj.chebfun(lambda x: 0.2 + 0.4 * x, domain=(-1.0, 1.0))
    g = (-2.0) ** f
    expected = cj.chebfun(
        lambda x: jnp.exp(f(x) * (jnp.log(2.0) + 1j * jnp.pi)),
        domain=(-1.0, 1.0),
    )
    assert jnp.iscomplexobj(g(X))
    assert _source_normest(g - expected) < 20 * EPS

def test_empty_exponent_returns_empty_chebfun():
    f = Chebfun.empty()
    result = 2.0 ** f
    assert result.isempty()

def test_zero_base_for_strictly_positive_exponent():
    f = cj.chebfun(lambda x: 1.5 + 0.2 * x)
    result = 0.0 ** f
    assert _source_normest(result) == 0.0

def test_scalar_base_preserves_domain_and_piece_intervals():
    f = cj.chebfun(jnp.sin, domain=(2.0, 2.5, 3.0))
    result = 3.0 ** f
    assert result.domain == f.domain
    assert [p.interval for p in result.funs] == [p.interval for p in f.funs]
    expected = cj.chebfun(lambda x: jnp.exp(f(x) * jnp.log(3.0)),
                          domain=(2.0, 3.0))
    assert _source_normest(result - expected) < 20 * EPS

def test_nonscalar_base_defers_for_array_dispatch():
    f = cj.chebfun(jnp.sin)
    assert f.__rpow__(jnp.asarray([2.0, 3.0])) is NotImplemented

@pytest.mark.parametrize("base", [-2.0, 2.0j])
def test_constructed_reverse_power_supports_jit_and_jvp(base):
    f = cj.chebfun(lambda x: .15 + .2*jnp.sin(x))
    result = base ** f
    points = jnp.asarray([-.8, -.1, .2, .9])
    actual = jax.jit(lambda x: result(x))(points)
    _, derivative = jax.jvp(lambda x: result(x), (points,), (jnp.ones_like(points),))
    logbase = jnp.log(jnp.asarray(base, dtype=jnp.complex128))
    expected = jnp.exp(logbase*(.15+.2*jnp.sin(points)))
    expected_derivative = expected*logbase*.2*jnp.cos(points)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                               rtol=20*EPS, atol=20*EPS)
    # Differentiating the interpolant amplifies coefficient roundoff.
    np.testing.assert_allclose(np.asarray(derivative), np.asarray(expected_derivative),
                               rtol=128*EPS, atol=128*EPS)

@pytest.mark.parametrize("base", [2.0, 2.0j])
def test_reverse_power_preserves_row_array_orientation(base):
    f = cj.chebfun(lambda x: jnp.stack([.2*x, .3+x*.1], axis=-1)).T
    result = base ** f
    assert result.is_transposed and result.size() == f.size()
    points = jnp.asarray([-.7, .1, .8])
    np.testing.assert_allclose(np.asarray(result(points)),
                               np.asarray(jnp.power(base, f(points))),
                               rtol=20*EPS, atol=20*EPS)
    # Internal pointValues stay breakpoint-major even for row arrays.
    np.testing.assert_allclose(np.asarray(result._breakpoint_values()),
                               np.asarray(jnp.power(base, f._breakpoint_values())),
                               rtol=20*EPS, atol=20*EPS)


def test_reverse_power_maps_explicit_breakpoint_values():
    f = cj.chebfun(lambda x: .2*x, domain=(-1., 0., 1.))
    f = f.set_point_values(jnp.asarray([-1., 2., 3.]))
    result = 2.0 ** f
    np.testing.assert_array_equal(np.asarray(result.point_values), [.5, 4., 8.])
    np.testing.assert_array_equal(np.asarray(result(jnp.asarray([-1., 0., 1.]))),
                                  [.5, 4., 8.])


def test_reverse_power_maps_mean_at_discontinuous_breakpoint():
    f = cj.chebfun([lambda x: -jnp.ones_like(x), jnp.ones_like],
                   domain=(-1., 0., 1.))
    assert float(f(jnp.asarray(0.))) == 0.
    result = 2.0 ** f
    # Source compose maps f.pointValues, rather than averaging mapped limits.
    assert float(result(jnp.asarray(0.))) == 1.
    np.testing.assert_array_equal(np.asarray(result.point_values), [.5, 1., 2.])
