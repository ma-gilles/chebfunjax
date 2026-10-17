"""Public smooth Bndfun dispatch and canonical mapping source controls."""
import copy

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.fixture(autouse=True)
def isolated_session():
    saved = ChebfunPref._defaults
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref._defaults = saved


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('branch', ['unary', 'binary', 'data', 'pref', 'empty_g_list', 'empty_g_array'])
def test_nargin_dispatch(monkeypatch, Tech, branch):
    f = Bndfun.from_chebtech(Tech.from_coeffs(jnp.array([.1, .2])), Domain((-2., 7.)))
    other = Bndfun.from_chebtech(Tech.from_coeffs(jnp.array([.3, .4])), f.domain)
    ChebfunPref.setDefaults(minSamples=37)
    calls = []

    def compose(self, op, g=None, **kwargs):
        calls.append((self, op, g, kwargs))
        return self

    monkeypatch.setattr(Tech, 'compose', compose)
    if branch == 'unary':
        result = f.compose(jnp.square)
    elif branch == 'binary':
        result = f.compose(jnp.add, other)
    elif branch in ('data', 'empty_g_list', 'empty_g_array'):
        empty = [] if branch == 'empty_g_list' else jnp.array([]) if branch == 'empty_g_array' else None
        result = f.compose(jnp.square, empty, {})
    else:
        result = f.compose(jnp.square, None, [], {'minSamples': 41})
    assert result.domain is f.domain and type(result.onefun) is Tech
    _, _, actual_g, kw = calls[0]
    assert actual_g is (other.onefun if branch == 'binary' else None)
    if branch in ('unary', 'binary'):
        assert kw == {}
    else:
        assert kw['pref'].minSamples == (41 if branch == 'pref' else 37)
        assert kw['data'] == (None if branch == 'pref' else {})


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
def test_nonidentity_canonical_inverse_map(Tech):
    domain = Domain((-2., 7.))
    outer_domain = Domain((-5., 10.))
    f = Bndfun.from_chebtech(Tech.from_function(lambda t: .2+.1*domain.forward_map(t)), domain)
    outer = Bndfun.from_chebtech(Tech.from_function(lambda t: .5+.25*outer_domain.forward_map(t)), outer_domain)
    result = f.compose(outer)
    x = jnp.array([-1.9, -.3, 2.1, 6.8])
    np.testing.assert_allclose(result(x), .5+.25*(.2+.1*x), atol=3e-15, rtol=3e-15)
    assert result.domain is domain and type(result.onefun) is Tech
    # Direct canonical composition without inverse mapping is distinguishable.
    wrong = Bndfun.from_chebtech(f.onefun.compose(outer.onefun), domain)
    assert np.max(np.abs(np.asarray(wrong(x)-result(x)))) > .1


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('empty', [False, True])
def test_data_scales_reach_population(monkeypatch, Tech, empty):
    f = Bndfun.from_chebtech(Tech.from_coeffs(jnp.array([[.1, .2], [.2, .1]])), Domain((-3., 4.)))
    data = {'vscale': [], 'hscale': None} if empty else {'vscale': [1., 2.], 'hscale': 7.}
    before = copy.deepcopy(data)
    calls = []
    original = Tech._adaptive_construct

    def observe(cls, op, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(op, *args, **kwargs)

    monkeypatch.setattr(Tech, '_adaptive_construct', classmethod(observe))
    result = f.compose(jnp.square, None, data, {'minSamples': 33, 'sampleTest': True})
    kw = calls[0]
    np.testing.assert_array_equal(kw['vscale'], 0. if empty else [1., 2.])
    assert kw['hscale'] == (1. if empty else 7.) and kw['sample_test'] is False
    assert kw['min_samples'] == 33 and data == before
    x = jnp.array([-2.8, -.2, 1.1, 3.9])
    np.testing.assert_allclose(result(x), f(x)**2, atol=2e-14, rtol=2e-14)
