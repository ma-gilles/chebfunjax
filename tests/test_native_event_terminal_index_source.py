"""Installed R2025b odezero indexing and pinned Chebop maxnorm controls."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop
from chebfunjax.utils.native_ode_events import _event_values, locate_events


def locate(event):
    # Exact affine dense output: isolate source event indexing from integration.
    def dense(t):
        return jnp.asarray([t, 2*t])
    initial = _event_values(event, .25, dense(.25))[0]
    return locate_events(event, initial, .25, dense(.25), .75, dense(.75),
                         0., dense)


@pytest.mark.parametrize('which', ['none', 'first', 'second', 'both'])
def test_scalar_terminal_indexed_at_crossing(which):
    def event(t, y):
        value = {'none': [t+1, t+2], 'first': [t-.5, t+2],
                 'second': [t+1, t-.5], 'both': [t-.5, t-.5]}[which]
        return value, 1, 0
    if which in ('second', 'both'):
        with pytest.raises(IndexError, match='terminal flag index'):
            locate(event)
    else:
        times, _, indices, _, stopped = locate(event)
        if which == 'none':
            assert times == [] and indices == [] and not stopped
        else:
            np.testing.assert_allclose(times, [.5], rtol=0, atol=2e-14)
            assert indices == [1] and stopped


@pytest.mark.parametrize('disabled', [0, 1])
def test_mixed_infinite_disabled_components(disabled):
    limits = jnp.asarray([jnp.inf, 1.] if disabled == 0 else [.5, jnp.inf])
    def event(t, y):
        return jnp.abs(y)-limits, 1+0*limits, 0
    times, _, indices, _, stopped = locate(event)
    np.testing.assert_allclose(times, [.5], rtol=0, atol=2e-14)
    assert indices == [2 if disabled == 0 else 1] and stopped


def test_all_infinite_locator_has_no_crossing():
    def event(t, y):
        return jnp.full((2,), -jnp.inf), jnp.full((2,), jnp.nan), 0
    times, _, indices, _, stopped = locate(event)
    assert times == [] and indices == [] and not stopped


@pytest.mark.parametrize('value,terminal,direction', [
    ([jnp.nan], [1], 0), ([0.], [jnp.nan], 0), ([0.], [2], 0),
    ([0.], [1], 2), ([0.], [1], jnp.nan),
])
def test_existing_invalid_active_event_protections(value, terminal, direction):
    with pytest.raises(ValueError):
        _event_values(lambda t, y: (value, terminal, direction), 0., [0.])


def test_fixed_value_direction_shapes():
    with pytest.raises(ValueError, match='matching'):
        _event_values(lambda t, y: ([1., 2.], [1., 1.], [0., 0., 0.]), 0., [0.])
    with pytest.raises(ValueError, match='fixed'):
        _event_values(lambda t, y: ([1., 2.], [1., 1.], 0), 0., [0.], size=1)


@pytest.mark.parametrize('kind', ['none', 'first', 'second', 'mixed_first', 'mixed_second', 'all'])
def test_public_chebop_maxnorm_source(kind, monkeypatch):
    provider = importlib.import_module('chebfunjax.utils.native_ode113')
    original = provider.native_ode113
    calls = []
    def observed(fun, span, initial, options=None, **kwargs):
        result = original(fun, span, initial, options, **kwargs)
        calls.append((dict(options or {}), result))
        return result
    monkeypatch.setattr(provider, 'native_ode113', observed)
    speeds = [2., 1.] if kind in ('first', 'mixed_first') else [1., 2.]
    op = Chebop(lambda t, u, v: [u.diff()-speeds[0], v.diff()-speeds[1]], domain=[0., 1.])
    op.lbc = [0., 0.]
    op.maxnorm = {'none': 10., 'first': 1., 'second': 1.,
                  'mixed_first': [1., jnp.inf], 'mixed_second': [jnp.inf, 1.],
                  'all': [jnp.inf, jnp.inf]}[kind]
    if kind == 'second':
        with pytest.raises(IndexError, match='terminal flag index'):
            op.solve(0.)
        return
    u, v = op.solve(0.)
    np.testing.assert_allclose([float(u(.2)), float(v(.2))], np.asarray(speeds)*.2,
                               rtol=0, atol=1e-10)
    assert op._ivp_backend_used == 'native_ode113'
    options, result = calls[0]
    if kind in ('none', 'all'):
        np.testing.assert_allclose([float(u(1.)), float(v(1.))], speeds, rtol=0, atol=1e-10)
        assert result['xe'].size == 0
        assert ('Events' in options) == (kind == 'none')
    else:
        np.testing.assert_allclose(np.asarray(result['xe']), [.5], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(result['ie'], [2 if kind == 'mixed_second' else 1])
        assert np.isnan(float(u(.9))) and np.isnan(float(v(.9)))
