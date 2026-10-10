"""Preference/data ownership and native Singfun constructor branches."""
import copy

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun import _singfun_factory as factory
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('exponents,tol,extra', [((0., 0.), 2e-16, False), ((-.5, .2), 1e-14, True)])
def test_preference_forwarding_and_ownership(monkeypatch, tech, exponents, tol, extra):
    p = ChebfunPref({'tech': tech, 'chebfuneps': 2e-16, 'fixedLength': 19,
                    'sampleTest': False, 'maxLength': 257, 'minSamples': 33,
                    'refinementFunction': 'resampling', 'happinessCheck': 'standard'})
    original = ChebfunPref(p)
    data = {'exponents': exponents, 'vscale': 7., 'hscale': 11., 'unknown': 23}
    saved = copy.deepcopy(data)
    calls = []
    smooth = tech.from_coeffs(jnp.array([1.]))

    def construct(op, **kw):
        calls.append((op(jnp.array([.2])), kw))
        return smooth

    monkeypatch.setattr(tech, 'from_function', staticmethod(construct))
    f = Singfun.constructor(lambda x: (1+x)**exponents[0]*(1-x)**exponents[1], data, p)
    assert f.smoothPart is smooth and f.exponents == exponents
    _, kw = calls[0]
    assert kw == dict(n=19, tol=tol, turbo=False, check='standard', sample_test=False,
                      refinement_function='resampling', max_length=257, min_samples=33,
                      vscale=7., hscale=11., **({'extrapolate': extra} if tech is Chebtech2 else {}))
    assert float(calls[0][0][0]) == 1.
    assert p == original and data == saved


def test_turbo_keeps_requested_output_length(monkeypatch):
    smooth = Chebtech2.from_coeffs(jnp.array([1.]))
    calls = []
    monkeypatch.setattr(Chebtech2, 'from_function', staticmethod(lambda op, **kw: calls.append(kw) or smooth))
    Singfun.constructor(lambda x: 1+0*x, {'exponents': (0., 0.)}, {'fixedLength': 9, 'useTurbo': True})
    assert calls[0]['n'] == 9 and calls[0]['turbo'] is True


def test_default_hints_and_partial_exponents(monkeypatch):
    import importlib
    module = importlib.import_module('chebfunjax.fun.singfun')
    calls = []
    monkeypatch.setattr(module, '_find_sing_exponents', lambda op, hints: calls.append(hints) or (-2., -3.))
    monkeypatch.setattr(factory, '_smooth_part', lambda *args: Chebtech2.from_coeffs(jnp.array([1.])))
    f = Singfun.constructor(lambda x: x, {'exponents': (jnp.nan, .5), 'singType': []},
                           {'blowupPrefs': {'defaultSingType': 'pole'}})
    assert f.exponents == (-2., .5) and calls == [('pole', 'pole')]


def test_omitted_and_explicit_empty_identity():
    smooth = Chebtech2.from_coeffs(jnp.array([1., 1.]))
    f = Singfun(smooth, (-1., 0.))
    assert Singfun.constructor(f) is f and f.make(f) is f
    with pytest.raises(TypeError, match='badOp'):
        Singfun.constructor(f, {}, None)
    assert Singfun.constructor(smooth).smoothPart is smooth
    with pytest.raises(ValueError, match='unknownPref'):
        Singfun.constructor(smooth, {}, {'blowupPrefs': {'defaultSingType': 'invalid'}})


@pytest.mark.parametrize('op', [jnp.ones((2, 2)), lambda x: jnp.array([1., 2.])])
def test_array_rejection(op):
    with pytest.raises(ValueError, match='arrayValued'):
        Singfun.constructor(op, {'exponents': (0., 0.)})


def test_unknown_tech_no_fallback():
    with pytest.raises(ValueError, match='Unsupported public construction Tech'):
        Singfun.constructor(lambda x: x, {'exponents': (0., 0.)}, {'tech': 'not-a-tech'})


def test_legacy_and_native_make_preferences(monkeypatch):
    calls = []
    monkeypatch.setattr(factory, '_smooth_part', lambda op, data, pref: calls.append((data, pref)) or Chebtech2.from_coeffs(jnp.array([1.])))
    f = Singfun.empty()
    pref = ChebfunPref({'fixedLength': 11})
    f.make(lambda x: x, (0., 0.), ('none', 'none'), pref)
    f.make(lambda x: x, {'exponents': (0., 0.)}, pref)
    f.make(lambda x: x, [], ChebfunPref({'blowupPrefs': {'defaultSingType': 'none'}}))
    assert calls[0][1].fixedLength == calls[1][1].fixedLength == 11
    with pytest.raises(TypeError, match='cannot be combined'):
        f.make(lambda x: x, (0., 0.), data={})


def test_trig_receives_data_and_preferences(monkeypatch):
    smooth = Trigtech.from_coeffs(jnp.array([1.]))
    calls = []
    monkeypatch.setattr(Trigtech, 'from_function', staticmethod(lambda op, **kw: calls.append(kw) or smooth))
    f = Singfun.constructor(lambda x: x, {'exponents': (0., 0.), 'hscale': 8., 'vscale': 9., 'domain': [-4., 7.]},
                           {'tech': 'trigtech', 'fixedLength': 8})
    assert f.smoothPart is smooth
    assert calls[0]['data']['hscale'] == 8. and calls[0]['data']['domain'] == [-4., 7.]
    assert calls[0]['pref']['fixedLength'] == 8


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_actual_fixed_smooth_factor(tech):
    f = Singfun.constructor(lambda x: (1+x)**-.5*(2+x+x*x), {'exponents': (-.5, 0.)},
                           {'tech': tech, 'fixedLength': 17})
    x = jnp.linspace(-.9, .9, 31)
    assert isinstance(f.smoothPart, tech) and len(f.smoothPart) == 17
    assert float(jnp.max(jnp.abs(f(x)-(1+x)**-.5*(2+x+x*x)))) < 100*jnp.finfo(float).eps


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_actual_numeric_values_not_singular_samples(tech):
    values = jnp.array([2., 3., 4.])
    f = Singfun.constructor(values, {'exponents': (.5, -.5)}, {'tech': tech})
    reference = tech.from_values(values)
    assert f.smoothPart.isequal(reference) and f.exponents == (.5, -.5)


def test_actual_complex_callable_and_trig_factor():
    f = Singfun.constructor(lambda x: (1+2j)*(1+x), {'exponents': (0., 0.)})
    x = jnp.array([-.3, .6])
    assert float(jnp.max(jnp.abs(f(x)-(1+2j)*(1+x)))) < 100*jnp.finfo(float).eps
    t = Singfun.constructor(lambda x: 2+jnp.cos(jnp.pi*x), {'exponents': (0., 0.)},
                           {'tech': 'trigtech', 'fixedLength': 9})
    assert isinstance(t.smoothPart, Trigtech)
    assert float(jnp.max(jnp.abs(t(x)-(2+jnp.cos(jnp.pi*x))))) < 100*jnp.finfo(float).eps


def test_original_constructor_predicates_19_to_24():
    pref = ChebfunPref()
    f = Chebtech2.from_function(lambda x: jnp.sin(x))
    s = Singfun.constructor(f)
    assert bool((f-s.smoothPart).iszero())
    data = {'exponents': (-1.5, -1.), 'singType': ('sing', 'sing')}
    s = Singfun.constructor(f, data, pref)
    assert bool((f-s.smoothPart).iszero())
    assert float(jnp.max(jnp.abs(jnp.asarray(s.exponents)-jnp.array([-1.5, -1.])))) < pref.blowupPrefs.exponentTol
    f = Singfun.constructor(42.)
    assert bool((f-42.).iszero())
    f = Singfun.constructor(42., {'exponents': (1.5, 1.), 'singType': ('sing', 'sing')}, pref)
    g = Singfun.constructor(lambda x: 42.+0*x)
    g = Singfun(g.smoothPart, (1.5, 1.))
    assert bool((f-g).iszero())
    # Literal native pass24 references earlier s, not f.
    assert float(jnp.max(jnp.abs(jnp.asarray(s.exponents)-jnp.array([-1.5, -1.])))) < pref.blowupPrefs.exponentTol
