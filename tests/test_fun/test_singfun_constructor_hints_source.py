"""Native @singfun constructor126-145 and findSingExponents dispatch7574c77."""
import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

module = importlib.import_module('chebfunjax.fun.singfun')


@pytest.mark.parametrize('hints,expected,kinds', [
    (('pole', 'pole'), (-2., -3.), ('pole', 'pole')),
    (('sing', 'root'), (-.2, .3), ('sing', 'sing')),
    (('ROOT', 'PoLe'), (-.2, -3.), ('sing', 'pole')),
    (('none', 'none'), (0., 0.), ()),
    (('none', 'root'), (0., .3), ('sing',)),
])
def test_native_detection_dispatch(monkeypatch, hints, expected, kinds):
    calls = []
    def pole(op, end):
        calls.append(('pole', end))
        return -2. if end == 'left' else -3.
    def sing(op, end):
        calls.append(('sing', end))
        return -.2 if end == 'left' else .3
    monkeypatch.setattr(module, '_find_pole_order', pole)
    monkeypatch.setattr(module, '_find_sing_order', sing)
    assert module._find_sing_exponents(lambda x: x, hints) == expected
    assert tuple(kind for kind, _ in calls) == kinds


def test_numeric_detection_returns_before_hint_dispatch():
    assert module._find_sing_exponents(jnp.array([1., 2.]), ('invalid', 'invalid')) == (0., 0.)


def test_unknown_detection_hint_source_identifier():
    with pytest.raises(ValueError, match='CHEBFUN:SINGFUN:findSingExponents:unknownPref'):
        module._find_sing_exponents(lambda x: x, ('bad', 'none'))


@pytest.mark.parametrize('supplied,expected,detect', [
    (None, (-2., -3.), True), ([], (-2., -3.), True),
    ((jnp.nan, .4), (-2., .4), True),
    ((.3, jnp.nan), (.3, -3.), True),
    ((jnp.nan, jnp.nan), (-2., -3.), True),
    ((.3, .4), (.3, .4), False),
])
def test_constructor_resolves_only_missing_exponents(monkeypatch, supplied, expected, detect):
    calls = []
    def detection(op, hints):
        calls.append(hints)
        return (-2., -3.)
    monkeypatch.setattr(module, '_find_sing_exponents', detection)
    tech = Chebtech2.from_coeffs(jnp.array([1.]))
    monkeypatch.setattr(Chebtech2, 'from_function', staticmethod(lambda *a, **k: tech))
    f = Singfun.from_function(lambda x: x, supplied, sing_type=('pole', 'none'))
    assert f.exponents == expected
    assert calls == ([('pole', 'none')] if detect else [])


@pytest.mark.parametrize('hints,exponents', [(('pole', 'pole'), (-3., -4.)),
                                           (('pole', 'none'), (-3., 0.))])
def test_actual_integer_pole_constructor(hints, exponents):
    def operator(x):
        return (1+x)**exponents[0]*(1-x)**exponents[1]
    f = Singfun.from_function(operator, sing_type=hints)
    reference = Singfun.from_function(operator, exponents)
    assert f.exponents == exponents
    assert f.isequal(reference)
