"""Omitted scalar simplify tolerance: @chebfun/simplify.m36, pin7574c77."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def restore_session_preferences():
    saved = ChebfunPref()
    try:
        yield
    finally:
        ChebfunPref.setDefaults(saved)


@pytest.mark.parametrize("tech", ["chebtech1", "chebtech2"])
@pytest.mark.parametrize("transposed", [False, True])
def test_scalar_omitted_simplify_uses_current_session(tech, transposed):
    # Construct at factory precision BEFORE changing the session default.
    # An existing rational function has a genuine decaying coefficient tail.
    ChebfunPref.setDefaults("factory")
    f = cj.chebfun(lambda x: 1 / (1 + 10 * (x - 0.1) ** 2), domain=(-1.0, 0.0, 1.0), tech=tech)
    f = f.set_point_values(jnp.asarray([7.0, -3.0, 8.0]))
    if transposed:
        f = f.T
    original = [p.tech.coeffs for p in f.funs]
    ChebfunPref.setDefaults("chebfuneps", 1e-6)
    explicit = f.simplify(1e-6)
    actual = f.simplify()
    assert len(explicit) < len(f), "The fixture must expose a real chopped tail"
    assert [len(p.tech.coeffs) for p in actual.funs] == [len(p.tech.coeffs) for p in explicit.funs]
    for a, b, old, saved in zip(actual.funs, explicit.funs, f.funs, original):
        assert jnp.array_equal(a.tech.coeffs, b.tech.coeffs)
        assert jnp.array_equal(old.tech.coeffs, saved)
        assert type(a.tech) is type(old.tech)
        assert a.ishappy == old.ishappy
        assert a.interval == old.interval
    assert actual.domain == f.domain
    assert actual.is_transposed == f.is_transposed
    assert jnp.array_equal(actual.point_values, f.point_values)
    # Compare away from stored isolated point values; no point-value coercion.
    x = jnp.linspace(-0.99, 0.99, 202)
    assert float(jnp.max(jnp.abs(actual(x) - f(x)))) < 10e-6 * float(f.vscale)


@pytest.mark.parametrize("tech", ["chebtech1", "chebtech2"])
def test_explicit_scalar_simplify_tolerance_is_not_replaced(tech):
    ChebfunPref.setDefaults("factory")
    f = cj.chebfun(lambda x: 1 / (1 + 10 * (x - 0.1) ** 2), tech=tech)
    expected = f.simplify(1e-12)
    ChebfunPref.setDefaults("chebfuneps", 1e-3)
    actual = f.simplify(1e-12)
    assert jnp.array_equal(actual.funs[0].tech.coeffs, expected.funs[0].tech.coeffs)
