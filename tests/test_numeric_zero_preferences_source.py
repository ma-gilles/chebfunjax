"""Native numeric zero becomes an operator with resolved technology prefs.

Source7574c77: unbndfun97–131, onefun52–71, smoothfun45–57 and
chebfun parseInputs739–744. Independent controls, not native fixture slots.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts


@pytest.fixture(autouse=True)
def reset_preferences():
    saved = ChebfunPref._defaults
    ChebfunPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebfunPref._defaults = saved


@pytest.mark.parametrize("tech,kind", [(Chebtech1, 1), (Chebtech2, 2)])
@pytest.mark.parametrize("explicit", [False, True])
def test_actual_zero_refinement_preferences_and_precedence(tech, kind, explicit):
    calls = []

    def refine(op, previous, prefs):
        calls.append(prefs.copy())
        assert previous is None
        x = chebpts(prefs["minSamples"], kind=kind)
        return op(x), False

    def unused(*args):
        raise AssertionError("Explicit refinement must override session callback")

    ChebfunPref.setDefaults({"tech": tech.__name__, "techPrefs": {
        "chebfuneps": 2.**-24, "minSamples": 33, "maxLength": 129,
        "sampleTest": True, "happinessCheck": "strict", "useTurbo": False,
        "refinementFunction": unused if explicit else refine,
        "extrapolate": True,
    }})
    kwargs = ({"eps": 2.**-30, "min_samples": 17, "max_length": 65,
               "sample_test": False, "refinement_function": refine,
               "turbo": False, "extrapolate": False} if explicit else {})
    f = chebfun(jnp.zeros((3, 2), dtype=jnp.complex128), domain=(0., float("inf")), **kwargs)
    assert len(calls) == 1
    prefs = calls[0]
    assert prefs["chebfuneps"] == 2.**(-30 if explicit else -24)
    assert prefs["minSamples"] == (17 if explicit else 33)
    assert prefs["maxLength"] == (65 if explicit else 129)
    assert prefs["sampleTest"] is (not explicit)
    assert prefs["happinessCheck"] == "strict"
    assert prefs["refinementFunction"] is refine
    assert prefs["extrapolate"] is (kind == 2 and not explicit)
    assert isinstance(f.funs[0].tech, tech)
    assert f.n_columns == 2 and f.funs[0].tech.ishappy
    assert f.funs[0].tech.coeffs.dtype == jnp.float64
    assert bool(jnp.all(f(jnp.asarray([0., 1., float("inf")])) == 0))


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_fixed_length_explicit_false_and_finite_numeric_transform(tech, monkeypatch):
    original = tech.from_function
    calls = []

    def observe(cls, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(tech, "from_function", classmethod(observe))
    ChebfunPref.setDefaults({"tech": tech.__name__, "techPrefs": {
        "fixedLength": 17, "useTurbo": True, "extrapolate": True,
    }})
    f = chebfun(0., domain=(-float("inf"), 0.), n=9, turbo=False, extrapolate=False)
    assert calls[0]["n"] == 9 and calls[0]["turbo"] is False
    if tech is Chebtech2:
        assert calls[0]["extrapolate"] is False
    assert len(f.funs[0].tech) == 9
    calls.clear()
    values = jnp.asarray([1., 2., 3.])
    g = chebfun(values, tech=tech, n=3, sample_test=False, refinement_function="resampling")
    assert not calls
    assert bool(jnp.all(g.funs[0].tech.coeffs == tech.from_values(values).coeffs))


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_callable_omission_retains_prior_false_defaults(tech):
    ChebfunPref.setDefaults("useTurbo", True)
    ChebfunPref.setDefaults("extrapolate", True)
    omitted = chebfun(jnp.exp, tech=tech, n=9)
    explicit = chebfun(jnp.exp, tech=tech, n=9, turbo=False, extrapolate=False)
    assert bool(jnp.all(omitted.funs[0].tech.coeffs == explicit.funs[0].tech.coeffs))


def test_trig_adaptive_options_remain_explicitly_unsupported():
    with pytest.raises(ValueError, match="not yet supported"):
        chebfun(0., domain=(0., float("inf")), trig=True, sample_test=False)


@pytest.mark.parametrize("tech,kind", [(Chebtech1, 1), (Chebtech2, 2)])
def test_session_turbo_keeps_fixed_length_construction_adaptive(tech, kind):
    calls = []

    def refine(op, previous, prefs):
        calls.append(prefs.copy())
        assert previous is None
        return op(chebpts(prefs["minSamples"], kind=kind)), False

    ChebfunPref.setDefaults({"tech": tech.__name__, "techPrefs": {
        "fixedLength": 9, "useTurbo": True, "refinementFunction": refine,
    }})
    f = chebfun(0., domain=(0., float("inf")))
    assert len(calls) == 1
    assert calls[0]["fixedLength"] == 9 and calls[0]["useTurbo"] is True
    assert len(f.funs[0].tech) == 9
    assert bool(jnp.all(f.funs[0].tech.coeffs == 0))
