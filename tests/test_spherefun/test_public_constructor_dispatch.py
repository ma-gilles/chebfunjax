"""Independent constructor dispatch, representation and source-order controls.

Provenance
----------
MATLAB source : @spherefun/{spherefun,constructor,coeffs2spherefun}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
New analytic controls supplement the unchanged thirty original predicates.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun import Spherefun, spherefun
from chebfunjax.spherefun._constructor import fix_rank
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("value", [None, [], jnp.zeros((0, 3))])
def test_empty_precedes_option_errors(value):
    assert spherefun(value, -1, "eps", -1).isempty()


@pytest.mark.parametrize("rank", [-1, 1.5])
def test_arity_precedes_rank_error(rank):
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:CONSTRUCTOR:toFewInputArgs"):
        spherefun(lambda x: x, rank)


def test_string_variables_precede_rank_error():
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:constructor:str2op:depvars"):
        spherefun("x.*y.*z.*w", -1)


@pytest.mark.parametrize("disabled", [False, True])
def test_scalar_only_cartesian_vectorization(disabled):
    def scalar_only(x, y, z):
        assert jnp.ndim(z) == 0
        return math.cos(float(z))
    with jax.disable_jit(disabled):
        f = spherefun(scalar_only, "vectorize")
    lam = jnp.asarray([-.7, .1, 1.2])
    theta = jnp.asarray([.2, .9, 2.2])
    np.testing.assert_allclose(f(lam, theta), jnp.cos(jnp.cos(theta)),
                               rtol=0, atol=2e3*jnp.finfo(jnp.float64).eps)


def test_automatic_scalar_vectorization():
    def scalar_only(lam, theta):
        assert jnp.ndim(theta) == 0
        return math.cos(float(theta))
    with pytest.warns(UserWarning, match="constructor:vectorize"):
        f = spherefun(scalar_only)
    np.testing.assert_allclose(f(jnp.asarray([.1, .2]), jnp.asarray([.4, .7])),
                               jnp.cos(jnp.asarray([.4, .7])), rtol=0,
                               atol=2e3*jnp.finfo(jnp.float64).eps)


def test_copy_rank_prefix_padding_and_metadata():
    def constant(x):
        return Trigtech(coeffs=jnp.asarray([x], dtype=jnp.complex128),
                        is_real=True, ishappy=True)
    f = Spherefun(cols=[constant(2), constant(3)], rows=[constant(5), constant(7)],
                  pivots=jnp.asarray([11., 13.]), idx_plus=(0,), idx_minus=(1,),
                  pivot_locations=((.1, .2), (.3, .4)), nonzero_poles=True)
    assert spherefun(f) is f
    one = spherefun(f, 1)
    np.testing.assert_array_equal(one.pivots, [11.])
    assert one.idx_plus == (0,) and one.idx_minus == ()
    assert one.pivot_locations == f.pivot_locations and one.nonzero_poles
    padded = spherefun(f, 4)
    assert len(padded.cols) == 4 and padded.idx_plus == (0,) and padded.idx_minus == (1,)
    np.testing.assert_array_equal(padded.pivots, [11., 13., 0., 0.])
    assert all(bool(jnp.all(c.coeffs == 0)) for c in padded.cols[2:])
    zero = fix_rank(f, 0)
    assert zero.idx_plus == () and zero.idx_minus == (0,) and zero.nonzero_poles
    np.testing.assert_array_equal(zero.pivots, [jnp.inf])
    assert zero.pivot_locations == f.pivot_locations


def test_eps_forwarding_clamp_and_preference_ownership(monkeypatch):
    recorded = {}
    def fake(cls, op, **kwargs):
        recorded.update(kwargs)
        return Spherefun.empty()
    monkeypatch.setattr(Spherefun, "from_function", classmethod(fake))
    pref = ChebfunPref()
    original = pref.cheb2Prefs.chebfun2eps
    spherefun(lambda l, t: jnp.cos(t), pref, "eps", 1e-5)
    assert recorded["tol"] == 1e-5
    assert recorded["max_rank"] == pref.cheb2Prefs.maxRank
    assert recorded["max_sample"] == 2**16
    assert pref.cheb2Prefs.chebfun2eps == original
    spherefun(lambda l, t: jnp.cos(t), "eps", 0)
    assert recorded["tol"] == original


@pytest.mark.parametrize("disabled", [False, True])
def test_numeric_dimensions_cdr_and_coefficient_roundtrip(disabled):
    lam = -jnp.pi + 2*jnp.pi*jnp.arange(8)/8
    theta = jnp.linspace(0, jnp.pi, 5)
    xx, yy = jnp.meshgrid(lam, theta)
    values = jnp.cos(xx)*jnp.sin(yy)
    with jax.disable_jit(disabled):
        f = spherefun(values)
        assert all(c.coeffs.shape == (8,) for c in f.cols)
        assert all(r.coeffs.shape == (8,) for r in f.rows)
        g = spherefun(f.coeffs2(), "coeffs")
        cols, d, rows = f.cdr()
        cvals = jnp.stack([c(theta/jnp.pi) for c in cols], axis=1)
        rvals = jnp.stack([r(lam/jnp.pi) for r in rows], axis=1)
    tol = 2e3*jnp.finfo(jnp.float64).eps
    np.testing.assert_allclose(cvals@d@rvals.T, values, rtol=0, atol=tol)
    np.testing.assert_allclose(g.fevalm(lam, theta), values, rtol=0, atol=tol)


def test_fixed_lengths_grid_contract(monkeypatch):
    captured = {}
    def from_values(cls, values, **kwargs):
        captured["values"] = values
        return Spherefun.empty()
    monkeypatch.setattr(Spherefun, "from_values", classmethod(from_values))
    spherefun(lambda l, t: jnp.cos(l)*jnp.sin(t), [3, 4])
    assert captured["values"].shape == (4, 8)
    x = -jnp.pi + 2*jnp.pi*jnp.arange(8)/8
    y = jnp.linspace(0, jnp.pi, 4)
    np.testing.assert_allclose(captured["values"], jnp.sin(y[:, None])*jnp.cos(x[None, :]),
                               rtol=0, atol=4*jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("options", [(), (-1,), ("eps", -1)])
def test_empty_string_precedes_option_errors(options):
    assert spherefun("", *options).isempty()


@pytest.mark.parametrize("value,identifier", [
    (float("inf"), "inf"), (float("nan"), "nan"),
])
def test_numeric_scalar_recursion_retains_source_error(value, identifier):
    with pytest.raises(ValueError,
                       match=f"CHEBFUN:SPHEREFUN:constructor:{identifier}"):
        spherefun(value)


def test_numeric_scalar_recursion_resets_preferences(monkeypatch):
    recorded = {}
    def fake(cls, op, **kwargs):
        recorded.update(kwargs)
        np.testing.assert_array_equal(op(jnp.zeros((2, 2)), jnp.zeros((2, 2))),
                                      jnp.ones((2, 2)))
        return Spherefun.empty()
    monkeypatch.setattr(Spherefun, "from_function", classmethod(fake))
    pref = ChebfunPref()
    pref.cheb2Prefs.maxRank = 7
    pref.cheb2Prefs.sampleTest = False
    spherefun(1., pref, "eps", 1e-5)
    defaults = ChebfunPref()
    assert recorded["tol"] == defaults.cheb2Prefs.chebfun2eps
    assert recorded["max_rank"] == defaults.cheb2Prefs.maxRank
    assert recorded["sample_test"] == defaults.cheb2Prefs.sampleTest


def test_sample_test_preference_forwarding(monkeypatch):
    recorded = {}
    def fake(cls, op, **kwargs):
        recorded.update(kwargs)
        return Spherefun.empty()
    monkeypatch.setattr(Spherefun, "from_function", classmethod(fake))
    pref = ChebfunPref()
    pref.cheb2Prefs.sampleTest = False
    spherefun(lambda lam, theta: jnp.cos(theta), pref)
    assert recorded["sample_test"] is False


def test_disabled_sample_test_omits_constructed_function_probes(monkeypatch):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("sampleTest=false must not probe constructed Spherefun")
    monkeypatch.setattr(Spherefun, "__call__", forbidden)
    pref = ChebfunPref()
    pref.cheb2Prefs.sampleTest = False
    f = spherefun(lambda lam, theta: 1 + 0 * lam, pref)
    assert not f.isempty()
    assert all(bool(jnp.all(jnp.isfinite(c.coeffs))) for c in f.cols + f.rows)
