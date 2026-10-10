"""Public source predicates and explicitly labeled empty-Tech adapters."""
import equinox as eqx
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_strict_exponent_boundary(tech):
    tol = ChebfunPref().blowupPrefs.exponentTol
    above = float(jnp.nextafter(jnp.float64(-tol), jnp.inf))
    below = float(jnp.nextafter(jnp.float64(-tol), -jnp.inf))
    for exponent, expected in [(0., True), (.5, True), (-1.05e-11, True),
                               (-tol, False), (above, True), (below, False)]:
        f = Singfun(tech.from_coeffs(jnp.array([1.])), (exponent, 0.))
        assert bool(f.isfinite()) is expected
        assert bool(f.isinf()) is not expected


def test_current_preference_not_approximate_constant(monkeypatch):
    f = Singfun(Chebtech2.from_coeffs(jnp.array([1.])), (-1.5e-8, 0.))
    assert not f.isfinite()
    pref = ChebfunPref({'blowupPrefs': {'exponentTol': 2e-8}})
    monkeypatch.setattr(ChebfunPref, '_defaults', pref)
    assert f.isfinite()
    assert not f.isinf()


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('value', [jnp.nan, jnp.inf])
def test_nonfinite_factor_complement(tech, value):
    f = Singfun(tech.from_coeffs(jnp.array([value])), (0., 0.))
    assert not f.isfinite()
    assert f.isinf()  # Source complement includes NaN, unlike numeric isinf.
    assert f.isreal()
    assert bool(f.any()) is not bool(jnp.isnan(value))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_complex_zero_storage_and_metadata(tech):
    f = Singfun(tech.from_coeffs(jnp.array([2.+0.j]), ishappy=False), (0., 0.))
    assert not f.isreal()
    assert f.isfinite()
    assert not f.ishappy
    out = f.any(2)
    assert isinstance(out, tech)
    assert not out.ishappy
    assert bool(jnp.array_equal(out.coeffs, jnp.array([True])))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_any_delegation_and_dimension_error(tech):
    f = Singfun(tech.from_coeffs(jnp.array([2., 1.])), (-.5, 0.))
    assert f.any()
    assert f.any(1)
    out = f.any(2)
    assert isinstance(out, tech)
    assert bool(jnp.array_equal(out.coeffs, jnp.array([True])))
    with pytest.raises(ValueError, match='CHEBTECH:any:dim'):
        f.any(3)
    zero = Singfun(tech.from_coeffs(jnp.array([0.])), (-1., -1.))
    assert not zero.any()
    assert bool(jnp.array_equal(zero.any(2).coeffs, jnp.array([False])))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_chebtech_predicates_under_jit(tech):
    evaluate = eqx.filter_jit(lambda f: (f.isfinite(), f.isinf(), f.isreal(), f.any()))
    result = evaluate(Singfun(tech.from_coeffs(jnp.array([2., 1.])), (-1.05e-11, 0.)))
    assert tuple(bool(v) for v in result) == (True, False, True, True)


def test_isnan_endpoint_route_not_coefficient_proxy():
    # Smooth factor is finite, but zero times an endpoint pole is NaN.
    f = Singfun(Chebtech2.from_coeffs(jnp.array([1., 1.])), (-1., 0.))
    assert not f.smoothPart.isnan()
    assert f.isnan()


def test_python_empty_tech_adapter():
    # Native no-input uses numeric[]; these qualify the Python storage adapter.
    f = Singfun.constructor()
    assert f.isfinite()
    assert not f.isinf()
    assert f.isreal()
    assert not f.any()
    out = f.any(2)
    assert isinstance(out, Chebtech2)
    assert bool(jnp.array_equal(out.coeffs, jnp.array([False])))


def test_trig_factor_delegation():
    smooth = Trigtech(coeffs=jnp.array([[2.+0.j]]), is_real=False, ishappy=False)
    f = Singfun(smooth, (0., 0.))
    assert f.isfinite()
    assert not f.isinf()
    assert not f.isreal()
    assert f.any()
    assert isinstance(f.any(2), Trigtech)
    assert not f.any(2).ishappy
