"""Source constructor -> bndfun -> chebtech preference forwarding.

Chebfun7574c77 @chebfun/constructor.m, @bndfun/bndfun.m,
@chebtech/chebtech.m, populate.m and happinessCheck.m.
Controls are independent adapters, not additional literal native test slots.
"""
import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def observe_technology(monkeypatch, tech):
    original = tech.from_function
    calls = []

    def observe(cls, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(tech, "from_function", classmethod(observe))
    return calls


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("check", ["standard", "strict"])
def test_factory_forwards_to_actual_technology(monkeypatch, tech, check):
    calls = observe_technology(monkeypatch, tech)
    saved = ChebfunPref()
    tolerance = 2.**-24
    try:
        ChebfunPref.setDefaults("happinessCheck", check)
        f = chebfun(jnp.exp, tech=tech, eps=tolerance, n=9)
        assert isinstance(f.funs[0].tech, tech)
        assert calls[0]["tol"] == tolerance
        assert calls[0]["check"] == check
        assert calls[0]["n"] == 9
    finally:
        ChebfunPref.setDefaults(saved)


@pytest.mark.parametrize("enabled", [False, True])
def test_c1_turbo_forwarding(monkeypatch, enabled):
    calls = observe_technology(monkeypatch, Chebtech1)
    f = chebfun(jnp.exp, tech=Chebtech1, turbo=enabled, n=16)
    assert calls[0]["turbo"] is enabled
    assert isinstance(f.funs[0].tech, Chebtech1)
    points = jnp.linspace(-1., 1., 65)
    assert float(jnp.max(jnp.abs(f(points)-jnp.exp(points)))) < 128*jnp.finfo(jnp.float64).eps*math.e


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_omitted_tol_and_factory_check_unchanged(monkeypatch, tech):
    calls = observe_technology(monkeypatch, tech)
    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults("factory")
        chebfun(jnp.exp, tech=tech, n=9)
        assert calls[0]["tol"] is None
        assert calls[0]["check"] == "standard"
    finally:
        ChebfunPref.setDefaults(saved)


def test_direct_public_from_function_check(monkeypatch):
    calls = observe_technology(monkeypatch, Chebtech2)
    f = Chebfun.from_function(jnp.exp, Domain((-1., 1.)), check="strict")
    assert calls[0]["check"] == "strict"
    points = jnp.linspace(-1., 1., 65)
    assert float(jnp.max(jnp.abs(f(points)-jnp.exp(points)))) < 64*jnp.finfo(jnp.float64).eps*math.e


def test_c1_tolerance_changes_adaptive_resolution():
    # Independent analytic exp truth. Factor16 times relative tolerance
    # and sup(exp)=e allows a modest interpolation/chopping envelope; this
    # does not replace any native acceptance bound or claim a uniform theorem.
    loose_tol, tight_tol = 2.**-12, 2.**-35
    loose = chebfun(jnp.exp, tech=Chebtech1, eps=loose_tol)
    tight = chebfun(jnp.exp, tech=Chebtech1, eps=tight_tol)
    assert len(loose) < len(tight)
    points = jnp.linspace(-1., 1., 129)
    for f, tol in [(loose, loose_tol), (tight, tight_tol)]:
        assert f.funs[0].tech.ishappy
        assert float(jnp.max(jnp.abs(f(points)-jnp.exp(points)))) < 16*tol*math.e
