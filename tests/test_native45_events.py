"""Independent analytic event controls; no native executable equality claim."""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.native_ode45 import _interpolate, native_ode45


def unit(t, y):
    return jnp.ones_like(y)


def test_quartic_extension_polynomial():
    # y=t^4 has exact DP stages for its time-only cubic derivative.
    t, h = .25, .5
    nodes = jnp.asarray([0., 1/5, 3/10, 4/5, 8/9, 1., 1.])
    f = (4*(t+h*nodes)**3)[None, :]
    query = jnp.asarray([t, t+.13, t+.31, t+h])
    y, dy = _interpolate(query, t, jnp.asarray([t**4]), h, f)
    assert bool(jnp.max(jnp.abs(y[0]-query**4)) < 32*jnp.finfo(jnp.float64).eps)
    assert bool(jnp.max(jnp.abs(dy[0]-4*query**3)) < 64*jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize('direction,ends', [(0, .7), (1, .7), (-1, 2.)])
def test_terminal_direction(direction, ends):
    def event(t, y):
        return y[0]-.7, 1, direction
    sol = native_ode45(unit, [0., 2.], [0.], {'Events': event, 'MaxStep': .4})
    assert abs(float(sol['x'][-1])-ends) < 256*jnp.finfo(jnp.float64).eps
    assert sol['xe'].size == (0 if direction == -1 else 1)
    if sol['xe'].size:
        assert int(sol['ie'][0]) == 1
        assert bool(jnp.array_equal(sol['y'][:, -1], sol['ye'][:, -1]))
        assert abs(float(sol['ye'][0, 0])-.7) < 256*jnp.finfo(jnp.float64).eps


def test_backward_crossing():
    sol = native_ode45(unit, [2., 0.], [2.],
                      {'Events': lambda t, y: (y[0]-.7, 1, -1), 'MaxStep': .4})
    assert abs(float(sol['x'][-1])-.7) < 256*jnp.finfo(jnp.float64).eps
    assert bool(jnp.all(jnp.diff(sol['x']) < 0))


def test_initial_zero_does_not_stop():
    sol = native_ode45(unit, [0., 1.], [0.],
                      {'Events': lambda t, y: (y[0], 1, 0)})
    assert float(sol['x'][-1]) == 1.
    assert sol['xe'].size == 1
    assert float(sol['xe'][0]) < 256*jnp.finfo(jnp.float64).eps


def test_vector_nonterminal_then_terminal():
    def events(t, y):
        return jnp.asarray([y[0]-.3, y[0]-.7]), jnp.asarray([0, 1]), jnp.asarray([0, 0])
    sol = native_ode45(unit, [0., 2.], [0.],
                      {'Events': events, 'InitialStep': 1., 'MaxStep': 1.})
    assert bool(jnp.array_equal(sol['ie'], jnp.asarray([1, 2])))
    error = jnp.max(jnp.abs(sol['xe']-jnp.asarray([.3, .7])))
    assert float(error) < 256*jnp.finfo(jnp.float64).eps
    assert abs(float(sol['x'][-1])-.7) < 256*jnp.finfo(jnp.float64).eps


def test_nonlinear_event_on_quartic_state():
    def rhs(t, y):
        return jnp.asarray([4*t**3])
    sol = native_ode45(rhs, [.25, 1.], [.25**4],
                      {'Events': lambda t, y: (y[0]-.5**4, 1, 0),
                       'InitialStep': .6, 'MaxStep': .6})
    assert abs(float(sol['x'][-1])-.5) < 512*jnp.finfo(jnp.float64).eps
    assert abs(float(sol['y'][0, -1])-.5**4) < 512*jnp.finfo(jnp.float64).eps
