"""Actual integrations through the native public output adapter."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


@pytest.fixture(scope='module')
def field():
    return Chebfun2v.from_functions(lambda x, y: 1+0*x, lambda x, y: 0*x,
                                   domain=(-2., 2., -2., 2.))


@pytest.mark.parametrize('events', [False, True])
@pytest.mark.parametrize('outputs', [1, 2, 5])
def test_actual_output_paths(field, monkeypatch, events, outputs):
    import chebfunjax.utils.native_ode45 as module
    original = module.native_ode45
    records = []
    def capture(*args, **kwargs):
        sol = original(*args, **kwargs)
        records.append(sol)
        return sol
    monkeypatch.setattr(module, 'native_ode45', capture)
    options = {'Events': lambda t, y: (y[0]-.7, 1, 0)} if events else None
    if not events and outputs == 5:
        with pytest.raises(KeyError, match='ie'):
            field.ode45([0., 1.], [0., 0.], options, outputs=outputs)
        assert len(records) == 1 and float(records[0]['x'][-1]) == 1.
        return
    result = field.ode45([0., 1.], [0., 0.], options, outputs=outputs)
    raw = records[0]
    if outputs == 1:
        T, Y = result['x'], result['y']
        assert result['stats'] is raw['stats']
        if events:
            assert result['xe'] is raw['xe'] and result['ye'] is raw['ye']
            assert result['ie'] is raw['ie']
        else:
            assert 'ie' not in result
    else:
        T, Y = result[:2]
        if outputs == 5:
            assert result[2] is raw['x'] and result[3] is raw['y']
            assert result[4] is raw['ie']
            assert result[2].size > raw['xe'].size
    assert isinstance(T, Chebfun) and isinstance(Y, Chebfun)
    end = .7 if events else 1.
    assert abs(Y.domain.b-end) < 512*jnp.finfo(jnp.float64).eps
    assert abs(complex(Y(Y.domain.b))-end) < 512*jnp.finfo(jnp.float64).eps
    assert not isinstance(raw['x'], Chebfun) and not isinstance(raw['y'], Chebfun)


def test_actual_event_present_no_crossing(field):
    result = field.ode45([0., 1.], [0., 0.],
                        {'Events': lambda t, y: (y[0]-5., 1, 0)}, outputs=5)
    assert result[4].size == 0
    assert float(result[2][-1]) == 1.
