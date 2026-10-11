"""Supplemental ANGLE source/API controls, native predicate count zero.

Source @chebfun/angle.m at7574c77 always uses atan2(imag,real,pref).
Fixed100eps bounds below are supplemental, never claimed native bounds.
No standalone tests/chebfun/test_angle.m exists at the source pin.
"""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)
SITES = jnp.asarray(np.linspace(-0.99, 0.99, 100))


def check_phase(f, h, name):
    expected = jnp.angle(f(SITES))
    actual = h(SITES)
    assert expected.shape == actual.shape == (100,)
    error = float(jnp.linalg.norm(actual - expected, ord=jnp.inf))
    print({"supplement": name, "error": error, "bound": 100 * EPS}, flush=True)
    assert error < 100 * EPS
    assert set(f.domain.breakpoints).issubset(h.domain.breakpoints)
    expected_points = jnp.angle(f(jnp.asarray(h.domain.breakpoints)))
    assert expected_points.shape == h.point_values.shape
    assert bool(jnp.all(h.point_values == expected_points))


def test_angle_source_explicit_pref_forwarding(monkeypatch):
    mod = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    original = mod.atan2
    calls = []

    def observe(y, x, pref=None):
        calls.append(pref)
        return original(y, x, pref)

    monkeypatch.setattr(mod, "atan2", observe)
    pref = ChebfunPref()
    pref.chebfuneps = 2 * EPS
    f = cj.chebfun(lambda x: jnp.exp(1j * jnp.pi * x / 3))
    h = f.angle(pref)
    assert calls == [pref]
    check_phase(f, h, "explicit_pref_smooth_phase")


@pytest.mark.parametrize("explicit_pref", [False, True])
def test_angle_source_later_complex_fun(explicit_pref):
    f = cj.chebfun([1, lambda x: jnp.exp(1j * jnp.pi * x / 3)], domain=(-1, 0, 1))
    assert not jnp.iscomplexobj(f.funs[0].tech.coeffs)
    assert jnp.iscomplexobj(f.funs[1].tech.coeffs)
    h = f.angle(ChebfunPref()) if explicit_pref else f.angle()
    check_phase(f, h, "first_real_later_complex_" + str(explicit_pref))


@pytest.mark.parametrize("explicit_pref", [False, True])
def test_angle_source_isolated_complex_point(explicit_pref):
    f = cj.chebfun([1, 1], domain=(-1, 0, 1))
    assert all(not jnp.iscomplexobj(p.tech.coeffs) for p in f.funs)
    f = f.set_point_values(jnp.asarray([1, 1j, 1]))
    h = f.angle(ChebfunPref()) if explicit_pref else f.angle()
    check_phase(f, h, "real_FUNs_complex_stored_point_" + str(explicit_pref))
    assert float(h(jnp.asarray(0.0))) == float(jnp.pi / 2)
    assert float(h(jnp.asarray(0.25))) == 0


def test_angle_source_real_negative_control():
    f = cj.chebfun(-1)
    h = f.angle()
    check_phase(f, h, "real_negative_constant")
