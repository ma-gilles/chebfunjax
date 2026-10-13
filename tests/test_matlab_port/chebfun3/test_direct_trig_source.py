"""Direct periodic construction controls and original technology clauses.

Provenance
----------
MATLAB source: tests/chebfun3/test_techs.m, clauses 5--6;
              @chebfun3/constructor.m; @chebfun3/chebfun3f.m
Chebfun commit: 7574c77

Off-grid, complex/nonunit-domain and dispatch controls supplement the
original near-zero origin predicate; they do not replace it.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech


@pytest.fixture(autouse=True)
def no_postconversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("direct construction called change_tech")
    monkeypatch.setattr(Chebfun3, 'change_tech', forbidden)
    original = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref.setDefaults(original)


def test_original_techs_5_6_and_native_off_grid():
    pref = ChebfunPref()
    pref.tech = 'trigtech'
    op = lambda x, y, z: jnp.cos(jnp.pi*z)*jnp.sin(jnp.pi*y)*jnp.sin(jnp.pi*x)
    f = chebfun3(op, pref=pref)
    tol = 100 * pref.cheb3Prefs.chebfun3eps
    assert isinstance(f.cols[0], Trigtech)
    assert abs(float(f(0, 0, 0))-float(op(0., 0., 0.))) < 100*tol
    fixture = json.loads((Path(__file__).parent/'fixtures/trig_source_queries.json').read_text())
    xyz = jnp.asarray(fixture['probes'])
    actual = f(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert float(jnp.max(jnp.abs(actual-jnp.asarray(fixture['values'])))) < 100*tol
    assert float(jnp.max(jnp.abs(actual-op(xyz[:, 0], xyz[:, 1], xyz[:, 2])))) < 100*tol
    assert f.length() == (3, 3, 3)
    assert pref.tech == 'trigtech'


def test_complex_asymmetric_modes_nonunit_domain():
    domain = (-2., 4., 1., 5., -3., 7.)
    def op(x, y, z):
        a, b, c = (x+2)/6, (y-1)/4, (z+3)/10
        return ((1+.2j)*jnp.exp(2j*jnp.pi*(2*a+b))
                + .3*jnp.cos(2*jnp.pi*(a-3*c)) + .07*jnp.sin(14*jnp.pi*a))
    f = chebfun3(op, domain, periodic=True)
    q = jnp.asarray([.071, .239, .467, .691, .883])
    x, y, z = -2+6*q, 1+4*q[::-1], -3+10*q**2
    assert f.domain == domain
    assert all(isinstance(t, Trigtech) for group in (f.cols, f.rows, f.tubes) for t in group)
    assert float(jnp.max(jnp.abs(f(x, y, z)-op(x, y, z)))) < 10000*2**-52


def test_zero_and_periodic_sample_grid():
    pref = ChebfunPref()
    pref.tech = 'trigtech'
    f = chebfun3(lambda x, y, z: 0*x, pref=pref)
    assert isinstance(f.cols[0], Trigtech)
    assert float(f(.13, .37, -.21)) == 0.
    shapes = []
    class Sampled(Exception):
        pass
    def op(x, y, z):
        if x.ndim == 3:
            shapes.append(np.asarray(x[:, 0, 0]))
            raise Sampled
        return x+y+z
    with pytest.raises(Sampled):
        chebfun3(op, pref=pref)
    np.testing.assert_array_equal(shapes[0], -1+np.arange(17)/17*2)


def test_sample_test_preference_and_session_tech():
    seen = []
    def op(x, y, z):
        seen.append(x.shape)
        return jnp.cos(jnp.pi*x)+0*y+0*z
    pref = ChebfunPref()
    pref.tech = 'trigtech'
    pref.cheb3Prefs.sampleTest = False
    ChebfunPref.setDefaults(pref)
    f = chebfun3(op)
    assert isinstance(f.cols[0], Trigtech)
    assert (30,) not in seen
    assert pref.cheb3Prefs.sampleTest is False
    assert abs(float(f(.17, .21, .43))-float(jnp.cos(.17*jnp.pi))) < 10000*2**-52


@pytest.mark.parametrize("value", [0., 1.+2j])
def test_numeric_scalar_periodic_uses_session_technology(value):
    from chebfunjax.tech.chebtech import Chebtech2

    # Native chebfun3double recurses without the parsed local preference for
    # scalar values, so the default session technology wins over `trig`.
    f = chebfun3(value, trig=True)
    assert all(isinstance(t, Chebtech2) for group in
               (f.cols, f.rows, f.tubes) for t in group)
    assert abs(complex(f(.13, -.21, .37))-value) < 10000*2**-52


def test_numeric_scalar_drops_call_local_preferences_but_uses_session_default():
    ChebfunPref.setDefaults("tech", "trigtech")
    from chebfunjax.tech.chebtech import Chebtech2

    explicit_c2 = ChebfunPref(tech="chebtech2")
    for f in (
        chebfun3(3., tech="chebtech2"),
        Chebfun3.from_values(jnp.asarray(3.), tech=Chebtech2),
        chebfun3(3., trig=False, eps=1e-8),
        chebfun3(3., pref=explicit_c2),
        chebfun3(3., coeffs=True, pref=explicit_c2),
    ):
        assert all(isinstance(t, Trigtech) for group in
                   (f.cols, f.rows, f.tubes) for t in group)
        assert abs(float(f(.13, -.21, .37))-3.) < 100*np.finfo(float).eps*3


def test_explicit_chebtech2_preference_original_3_4():
    from chebfunjax.tech.chebtech import Chebtech2
    pref = ChebfunPref()
    pref.tech = "chebtech2"
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z), pref=pref)
    assert isinstance(f.cols[0], Chebtech2)
    assert abs(float(f(0., 0., 0.))-1.) < 100*pref.cheb3Prefs.chebfun3eps


def test_constructor_preference_dispatch_and_identity():
    pref = ChebfunPref()
    pref.cheb3Prefs.constructor = "classic"
    ChebfunPref.setDefaults(pref)
    with pytest.raises(NotImplementedError, match="chebfun3f"):
        chebfun3(lambda x, y, z: x)
    class Sampled(Exception):
        pass
    def stop(x, y, z):
        raise Sampled
    with pytest.raises(Sampled):
        chebfun3(stop, chebfun3f=True)
    existing = Chebfun3.empty()
    assert chebfun3(existing) is existing
    pref.cheb3Prefs.constructor = "unrecognized_native_fallback"
    with pytest.raises(Sampled):
        chebfun3(stop, pref=pref)


@pytest.mark.parametrize("field,value", [
    ("tech", "custom_unsupported_tech"),
    ("happinessCheck", "classic"),
])
def test_explicit_unsupported_preferences(field, value):
    pref = ChebfunPref()
    setattr(pref, field, value)
    with pytest.raises(NotImplementedError):
        chebfun3(lambda x, y, z: x+y+z, pref=pref)


def test_scalar_periodic_flag_is_dropped_with_call_local_preference():
    from chebfunjax.tech.chebtech import Chebtech2

    pref = ChebfunPref()
    pref.tech = "chebtech1"
    f = chebfun3(1., pref=pref, trig=True)
    # The native scalar recursion drops both the explicit preference and the
    # trig flag, then uses the current default C2 technology.
    assert isinstance(f.cols[0], Chebtech2)
    assert pref.tech == "chebtech1"


def test_fixed_length_scalar_callback_also_resets_local_technology():
    from chebfunjax.tech.chebtech import Chebtech2

    f = chebfun3(lambda x, y, z: 3., n=(5, 4, 3), trig=True)
    assert f.length() == (1, 1, 1)
    assert all(isinstance(t, Chebtech2) for group in
               (f.cols, f.rows, f.tubes) for t in group)
