"""Literal pinned wrapper output branches using an independent solver record."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


@pytest.fixture(scope='module')
def fields():
    return {n: Chebfun2v.from_functions(*([lambda x, y: 1+0*x]*n)) for n in (2, 3)}


def install(monkeypatch, components, *, events=True, crossing=True, nan=False):
    import chebfunjax.utils.native_ode45 as module
    x = jnp.asarray([0., .2, .7, 1.])
    y = jnp.stack([x*(i+1) for i in range(components)])
    if nan:
        y = y.at[0, 1].set(jnp.nan)
    metadata = {'unrelated': object()}
    sol = {'solver': 'ode45', 'x': x, 'y': y, 'stats': metadata, 'extdata': {'tag': object()}}
    if events:
        sol.update(xe=jnp.asarray([1.]) if crossing else jnp.empty((0,)),
                   ye=y[:, -1:] if crossing else jnp.empty((components, 0)),
                   ie=jnp.asarray([1]) if crossing else jnp.empty((0,), dtype=jnp.int32))
    calls = []
    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return sol
    monkeypatch.setattr(module, 'native_ode45', solve)
    return sol, calls, x, y


@pytest.mark.parametrize('components', [2, 3])
@pytest.mark.parametrize('outputs', [None, 0, 1, 2, 3, 4, 5])
def test_literal_outputs(fields, monkeypatch, components, outputs):
    sol, calls, x, y = install(monkeypatch, components)
    original_keys = tuple(sol)
    result = fields[components].ode45([0., 1.], [0.]*components, outputs=outputs)
    assert len(calls) == 1
    assert tuple(sol) == original_keys
    assert sol['x'] is x and sol['y'] is y
    if outputs == 0:
        assert result is None
        return
    if outputs == 1:
        assert isinstance(result, dict) and result is not sol
        assert tuple(result) == original_keys
        for key in sol:
            if key not in ('x', 'y'):
                assert result[key] is sol[key]
        T, Y = result['x'], result['y']
    else:
        assert isinstance(result, tuple)
        assert len(result) == (2 if outputs is None else outputs)
        T, Y = result[:2]
        if outputs is not None and outputs >= 3:
            assert result[2] is x
        if outputs is not None and outputs >= 4:
            assert result[3] is y
        if outputs == 5:
            assert result[4] is sol['ie']
    assert isinstance(T, Chebfun) and isinstance(Y, Chebfun)
    assert T.domain.breakpoints == Y.domain.breakpoints == (0., 1.)
    assert abs(float(T(1.))-1.) < 32*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('outputs', [1, 2, 3, 4, 5])
def test_noevent_field_presence(fields, monkeypatch, outputs):
    sol, calls, x, y = install(monkeypatch, 2, events=False)
    if outputs >= 3:
        with pytest.raises(KeyError, match='ie'):
            fields[2].ode45([0., 1.], [0., 0.], outputs=outputs)
    else:
        result = fields[2].ode45([0., 1.], [0., 0.], outputs=outputs)
        if outputs == 1:
            assert 'ie' not in result and 'xe' not in result
    assert len(calls) == 1 and sol['x'] is x and sol['y'] is y


def test_event_without_crossing_has_empty_ie(fields, monkeypatch):
    sol, _, x, y = install(monkeypatch, 2, crossing=False)
    result = fields[2].ode45([0., 1.], [0., 0.], outputs=5)
    assert result[2] is x and result[3] is y and result[4] is sol['ie']
    assert result[4].shape == (0,)


@pytest.mark.parametrize('outputs', [None, 0, 1, 2, 3, 4, 5])
def test_empty_before_inputs(monkeypatch, outputs):
    import chebfunjax.utils.native_ode45 as module
    def forbidden(*args, **kwargs):
        raise AssertionError('empty source must return before solver or input validation')
    monkeypatch.setattr(module, 'native_ode45', forbidden)
    field = Chebfun2v.empty()
    if outputs is not None and outputs > 1:
        with pytest.raises(ValueError, match='only one output'):
            field.ode45(None, None, outputs=outputs)
    else:
        result = field.ode45(None, None, outputs=outputs)
        if outputs == 0:
            assert result is None
        else:
            assert result.shape == (0,)


@pytest.mark.parametrize('outputs', [-1, 6, True, 1.5, '1'])
def test_invalid_output_selection(fields, monkeypatch, outputs):
    _, calls, _, _ = install(monkeypatch, 2)
    with pytest.raises(ValueError, match='integer from 0 to 5'):
        fields[2].ode45([0., 1.], [0., 0.], outputs=outputs)
    assert not calls


@pytest.mark.parametrize('outputs', [1, 2, 5])
def test_nan_source_error_precedes_output_selection(fields, monkeypatch, outputs):
    install(monkeypatch, 2, nan=True)
    with pytest.raises(RuntimeError, match='IVP returned NaN'):
        fields[2].ode45([0., 1.], [0., 0.], outputs=outputs)
