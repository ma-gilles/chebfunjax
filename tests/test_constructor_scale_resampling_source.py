"""Constructor scale/refinement modifiers from Chebfun native source.

Provenance
----------
MATLAB source: @chebfun/chebfun.m parseInputs642–644,671–676;
@chebfun/constructor.m48–50 and getFun.
Chebfun commit: 7574c77.
Manufactured polynomial controls use 10eps; modifiers introduce no new bound.
"""

from unittest.mock import patch

import jax.numpy as jnp

from chebfunjax.chebfun1d._construction import prepare_preferences
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def test_supplied_scale_data():
    for scale in [7.0, jnp.array([7.0, 9.0])]:
        (_, _, data, _) = prepare_preferences((-1.0, 1.0), keywords={"vscale": scale})
        assert jnp.array_equal(jnp.asarray(data["vscale"]), jnp.asarray(scale))


def test_empty_or_omitted_scale_default():
    for keywords in [{}, {"vscale": None}, {"vscale": []}, {"vscale": 0.0}]:
        (_, _, data, _) = prepare_preferences((-1.0, 1.0), keywords=keywords)
        assert data["vscale"] == 0


def test_explicit_off_overrides_pref_without_mutation():
    pref = ChebfunPref({"refinementFunction": "resampling"})
    (private, _, _, _) = prepare_preferences((-1.0, 1.0), pref, keywords={"resampling": False})
    assert private.refinementFunction == "nested"
    assert pref.refinementFunction == "resampling"
    assert prepare_preferences((-1.0, 1.0), pref)[0].refinementFunction == "resampling"


def test_explicit_on_overrides_pref_without_mutation():
    pref = ChebfunPref({"refinementFunction": "nested"})
    (private, _, _, _) = prepare_preferences((-1.0, 1.0), pref, keywords={"resampling": True})
    assert private.refinementFunction == "resampling" and pref.refinementFunction == "nested"


def test_public_scalar_scale_forwarding():
    for tech in [Chebtech1, Chebtech2]:
        observations = []
        original = tech.from_function

        def observe(cls, op, **kwargs):
            observations.append(kwargs.copy())
            return original(op, **kwargs)

        with patch.object(tech, "from_function", classmethod(observe)):
            f = chebfun(lambda x: x * x, tech=tech, vscale=7.0)
        assert jnp.array_equal(jnp.asarray(observations[0]["vscale"]), jnp.asarray(7.0))
        x = jnp.array([-0.5, 0.0, 0.5])
        assert jnp.max(jnp.abs(f(x) - x * x)) < 10 * jnp.finfo(jnp.float64).eps


def test_public_column_scale_forwarding():
    observations = []
    original = Chebtech2.from_function

    def observe(cls, op, **kwargs):
        observations.append(kwargs.copy())
        return original(op, **kwargs)

    with patch.object(Chebtech2, "from_function", classmethod(observe)):
        f = chebfun(lambda x: jnp.stack((x, 2 * x), axis=-1), vscale=jnp.array([7.0, 9.0]))
    assert jnp.array_equal(jnp.asarray(observations[0]["vscale"]), jnp.array([7.0, 9.0]))
    x = jnp.array([-0.5, 0.0, 0.5])
    expected = jnp.stack((x, 2 * x), axis=-1)
    assert jnp.max(jnp.abs(f(x) - expected)) < 10 * jnp.finfo(jnp.float64).eps


def test_public_off_forwarding_from_session_defaults():
    original_pref = ChebfunPref()
    try:
        ChebfunPref.setDefaults("refinementFunction", "resampling")
        for tech in [Chebtech1, Chebtech2]:
            observations = []
            original = tech.from_function

            def observe(cls, op, **kwargs):
                observations.append(kwargs.copy())
                return original(op, **kwargs)

            with patch.object(tech, "from_function", classmethod(observe)):
                f = chebfun(lambda x: x * x, tech=tech, resampling=False)
            assert observations[0]["refinement_function"] == "nested"
            assert ChebfunPref().refinementFunction == "resampling"
            assert abs(f(0.5) - 0.25) < 10 * jnp.finfo(jnp.float64).eps
    finally:
        ChebfunPref.setDefaults(original_pref)


def test_unknown_keyword_still_rejected():
    try:
        chebfun(lambda x: x, nonexistent_native_modifier=True)
    except TypeError as e:
        assert "unexpected keyword" in str(e)
    else:
        raise AssertionError("unknown keyword accepted")
