"""Proposed source-branch controls, not previously executed native tests.

Provenance
----------
MATLAB source : @diskfun/norm.m, tests/diskfun/test_norm.m (precision scale)
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun


class SourceProbe:
    """Observe dispatch order without substituting a numerical Diskfun test."""

    def __init__(self, integral=4.0, empty=False):
        self.integral = integral
        self.empty = empty
        self.calls = []

    def isempty(self):
        return self.empty

    def __pow__(self, order):
        self.calls.append(('power', float(order)))
        return self

    def sum2(self):
        self.calls.append(('sum2',))
        return jnp.asarray(self.integral)

    def minandmax2(self):
        self.calls.append(('minandmax2',))
        return jnp.asarray([-2.0, 3.0]), jnp.asarray([0.0, 1.0])


@pytest.mark.parametrize('order', [4, 6, -2, 0])
def test_source_even_dispatch_order(order):
    f = SourceProbe()
    value = Diskfun.norm(f, order)
    assert f.calls == [('power', float(order)), ('sum2',)]
    if order == 0:
        assert bool(jnp.isinf(value)) and bool(value > 0)
    else:
        assert value == jnp.asarray(4.0) ** (jnp.asarray(1.0) / order)


@pytest.mark.parametrize('order', [3, -3])
def test_source_odd_error_follows_power(order):
    f = SourceProbe()
    with pytest.raises(ValueError, match='p-norm must have p even'):
        Diskfun.norm(f, order)
    assert f.calls == [('power', float(order))]


@pytest.mark.parametrize('order,message', [
    (1, 'L1-norm'), (2.5, 'does not support'),
    (float('nan'), 'does not support'),
    (-float('inf'), 'does not support'), ('min', 'does not support'),
    (4+1j, 'Unknown norm'), ('4', 'Unknown norm'),
])
def test_source_rejected_orders(order, message):
    f = SourceProbe()
    with pytest.raises(ValueError, match=message):
        Diskfun.norm(f, order)
    assert not f.calls


@pytest.mark.parametrize('order', [float('inf'), 'inf', 'max'])
def test_numeric_infinity_uses_existing_extrema(order):
    f = SourceProbe()
    assert Diskfun.norm(f, order) == 3.0
    assert f.calls == [('minandmax2',)]


@pytest.mark.parametrize('order', [2, 4, 'unknown', 4+1j])
def test_empty_precedes_dispatch(order):
    f = SourceProbe(empty=True)
    assert Diskfun.norm(f, order).shape == (0,)
    assert not f.calls


@pytest.mark.parametrize('order', [4, 6, -2, 0])
def test_actual_constant_disk(order):
    f = Diskfun.from_function(lambda theta, r: -2.0 + jnp.zeros_like(theta+r))
    value = f.norm(order)
    if order == 0:
        assert bool(jnp.isinf(value)) and bool(value > 0)
    else:
        expected = 2.0 * jnp.pi ** (1.0/order)
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        assert jnp.abs(value-expected) < tol


@pytest.mark.parametrize('order', [4, 6])
def test_actual_cartesian_x_disk(order):
    f = Diskfun.from_function(lambda theta, r: r*jnp.cos(theta))
    value = f.norm(order)
    expected_integral = jnp.pi/8 if order == 4 else 5*jnp.pi/64
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert jnp.abs(value-expected_integral**(1.0/order)) < tol


@pytest.mark.parametrize('shape', [(), (1,), (1, 1)])
def test_numeric_singleton_adapter(shape):
    f = SourceProbe()
    value = Diskfun.norm(f, jnp.full(shape, 4.0))
    assert value.shape == ()
    assert f.calls == [('power', 4.0), ('sum2',)]


def test_near_zero_strict_integer_threshold():
    # Native accepts |round(p)-p| < eps, even when p is not exactly integral.
    # eps/2 enters the rounded zero branch; exactly eps must be rejected.
    f = SourceProbe()
    value = Diskfun.norm(f, 2.0**-53)
    assert bool(jnp.isinf(value)) and bool(value > 0)
    assert f.calls == [('power', 0.0), ('sum2',)]
    boundary = SourceProbe()
    with pytest.raises(ValueError, match='does not support'):
        Diskfun.norm(boundary, 2.0**-52)
    assert not boundary.calls
