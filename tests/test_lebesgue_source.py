"""Literal native lebesgue test slots; Chebfun7574c77 tests/misc/test_lebesgue.m."""
import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.lebesgue_source import lebesgue
from chebfunjax.utils.quadrature import chebpts, legpts, trigpts


@pytest.mark.parametrize("slot", [1, 2, 3, 4, 5])
def test_native_polynomial_small(slot):
    tol = float(ChebfunPref().chebfuneps)
    if slot == 1:
        _, value = lebesgue(chebpts(3), return_constant=True)
        assert abs(value-5/4) < 10*tol
    elif slot == 2:
        nodes, _ = legpts(3)
        _, value = lebesgue(nodes, [-1., 1.], return_constant=True)
        assert abs(value-7/3) < 11*tol
    elif slot == 3:
        _, value = lebesgue(jnp.linspace(5., 9., 3), 5., 9., return_constant=True)
        assert abs(value-5/4) < 10*tol
    elif slot == 4:
        L = lebesgue([1., 2.], [0., 7.])
        assert abs(L.norm(jnp.inf)-11) < 100*tol
    else:
        L, value = lebesgue([-1., 0., 1.], [-1., 2.], return_constant=True)
        assert jnp.max(jnp.abs(jnp.asarray([value, L.max()[1]])-7)) < 100*tol


@pytest.mark.parametrize("slot", [8, 9, 10, 11])
def test_native_trigonometric_small(slot):
    tol = float(ChebfunPref().chebfuneps)
    if slot == 8:
        nodes, _ = trigpts(3)
        _, value = lebesgue(nodes, "trig", return_constant=True)
    elif slot == 9:
        nodes = jnp.linspace(-jnp.pi, jnp.pi, 4)[:-1]
        _, value = lebesgue(nodes, [-jnp.pi, jnp.pi], "trig", return_constant=True)
    elif slot == 10:
        nodes = jnp.linspace(-jnp.pi, jnp.pi, 4)
        _, value = lebesgue(nodes, -jnp.pi, jnp.pi, "trig", return_constant=True)
    else:
        nodes, _ = trigpts(4)
        with pytest.raises(ValueError, match="CHEBFUN:lebesgue:trigLebesgue:evenLengthGrid"):
            lebesgue(nodes, "trig")
        return
    assert abs(value-5/3) < 10*tol


@pytest.mark.parametrize("slot", [6, 7])
def test_native_polynomial_large(slot):
    if slot == 6:
        s = jnp.linspace(.25, 1., 17)
        L, _ = lebesgue(jnp.concatenate((-s, s)), return_constant=True)
        assert L.min()[1] > .999
    else:
        L = lebesgue(jnp.linspace(-1., 1., 40))
        assert len(L) < 1560


@pytest.mark.parametrize("technology", ["chebtech1", Chebtech1])
@pytest.mark.parametrize("trig", [False, True])
def test_session_tech_and_tolerance_source_contract(monkeypatch, technology, trig):
    constructor = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    original = constructor.chebfun
    captured = []

    def observe(*args, **kwargs):
        captured.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(constructor, "chebfun", observe)
    saved = ChebfunPref()
    tolerance = 2.**-30
    try:
        ChebfunPref.setDefaults("tech", technology)
        ChebfunPref.setDefaults("chebfuneps", tolerance)
        nodes = trigpts(3)[0] if trig else chebpts(3)
        L = lebesgue(nodes, *(('trig',) if trig else ()))
        assert all(isinstance(piece.tech, Chebtech1) for piece in L.funs)
        assert captured[0]["tech"] == technology
        assert captured[0]["eps"] == tolerance
        if not trig:
            assert captured[0]["n"] == 3
            assert captured[0]["sample_test"] is False
        assert float(jnp.max(jnp.abs(L(nodes)-1))) < 10*tolerance
    finally:
        ChebfunPref.setDefaults(saved)


@pytest.mark.parametrize("args,identifier", [
    ((True,), "badArg1"),
    (({"a": 0},), "badArg1"),
    (([-1., 1.], "other"), "badArg2"),
    ((-1., 1., "other"), "badArg3"),
    ((-1., 1., "trig", 0), "tooManyArgs"),
    (([[ -1.], [1.]],), "badDom"),
    (([1., -1.],), "badDom"),
    ((-.5, .5), "pointsOutsideDomain"),
])
def test_native_parse_errors(args, identifier):
    with pytest.raises(ValueError, match="CHEBFUN:lebesgue:parseInputs:"+identifier):
        lebesgue([-1., 0., 1.], *args)


@pytest.mark.parametrize("trig", [False, True])
def test_native_weight_matrix_error(trig):
    nodes = jnp.linspace(-.9, .9, 6).reshape(3, 2)
    identifier = "trigBaryWts" if trig else "baryWeights"
    with pytest.raises(ValueError, match="CHEBFUN:"+identifier+":matrix"):
        lebesgue(nodes, *(('trig',) if trig else ()))


@pytest.mark.parametrize("technology", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("trig", [False, True])
@pytest.mark.parametrize("fixed", [None, 17])
def test_session_preferences_reach_actual_technology(monkeypatch, technology, trig, fixed):
    original = technology.from_function
    calls = []

    def observe(cls, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(technology, "from_function", classmethod(observe))
    saved = ChebfunPref()
    tolerance = 2.**-24
    try:
        ChebfunPref.setDefaults("tech", technology)
        for name, value in [("chebfuneps", tolerance), ("minSamples", 33),
                            ("maxLength", 129), ("fixedLength", fixed),
                            ("refinementFunction", "resampling"),
                            ("sampleTest", True), ("happinessCheck", "strict")]:
            ChebfunPref.setDefaults(name, value)
        nodes = trigpts(3)[0] if trig else chebpts(3)
        L = lebesgue(nodes, *(('trig',) if trig else ()))
        assert calls
        for call in calls:
            assert call["tol"] == tolerance
            assert call["n"] == (fixed if trig else 3)
            assert call["min_samples"] == 33
            assert call["max_length"] == 129
            assert call["refinement_function"] == "resampling"
            assert call["sample_test"] is trig
            assert call["check"] == "strict"
        assert all(isinstance(piece.tech, technology) for piece in L.funs)
        assert float(jnp.max(jnp.abs(L(nodes)-1))) < 10*tolerance
    finally:
        ChebfunPref.setDefaults(saved)


def test_public_entry_returns_continuous_function_and_constant():
    import chebfunjax as cj
    tol = float(ChebfunPref().chebfuneps)
    assert cj.lebesgue is lebesgue
    L, constant = cj.lebesgue(chebpts(3), return_constant=True)
    assert isinstance(L, cj.Chebfun)
    assert abs(constant-5/4) < 10*tol
