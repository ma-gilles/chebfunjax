"""Native wrapper routing controls; no independent MATLAB trajectory oracle.

Provenance
----------
MATLAB source: @chebfun/constructODEsol.m, @chebfun/ode113.m, @chebfun/ode78.m,
@chebfun/ode89.m. Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('name', ['ode113', 'ode78', 'ode89'])
def test_default_native_wrapper_does_not_call_scipy(name, monkeypatch):
    import scipy.integrate

    def forbidden(*args, **kwargs):
        raise AssertionError('native selection reached SciPy')

    monkeypatch.setattr(scipy.integrate, 'solve_ivp', forbidden)
    solver = getattr(cj, name)
    result = solver(lambda t, y: jnp.ones_like(y), (0., 1.), 0.)
    assert abs(float(result(.7)) - .7) < 1e-12
    assert getattr(cj.Chebfun, name) is solver
    assert getattr(cj.chebfun, name) is solver


@pytest.mark.parametrize('name', ['ode78', 'ode89'])
def test_native_events_are_explicitly_unsupported(name):
    with pytest.raises(NotImplementedError, match='Events'):
        getattr(cj, name)(lambda t, y: y, (0., 1.), 1.,
                          {'Events': lambda t, y: (y[0] - 2, 1, 1)})


@pytest.mark.parametrize('a,b', [(0., .4), (.4, 1.), (-1., -.4), (-.4, 0.)])
def test_source_forward_mapping_keeps_dense_callback_endpoints_exact(a, b):
    from chebfunjax.chebfun1d.chebfun import _Piece

    seen = []

    def strict_dense(x):
        assert bool(jnp.all((x >= a) & (x <= b)))
        seen.append(x)
        return x

    piece = _Piece.from_function(strict_dense, a, b)
    assert seen
    # Adaptive construction also makes interior-only happiness probes.
    # The canonical sampling batch contains both exact endpoints; every
    # callback invocation above is checked against the strict interval.
    assert any(float(x[0]) == a and float(x[-1]) == b for x in seen)
    assert bool(piece.ishappy)


def test_native113_public_terminal_event():
    result = cj.ode113(lambda t, y: jnp.ones_like(y), (0., 1.), 0.,
                      {'Events': lambda t, y: (y[0]-.5, 1, 1)})
    assert abs(float(result(.25))-.25) < 1e-12
    assert bool(jnp.isnan(result(.75)))
